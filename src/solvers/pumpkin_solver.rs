//! A solver that uses [pumpkin](https://docs.rs/pumpkin-solver), a pure rust solver.

use std::collections::HashMap;

use pumpkin_solver::constraints;
use pumpkin_solver::results::{ProblemSolution, SatisfactionResult, Solution as PumpkinSolution};
use pumpkin_solver::termination::Indefinite;
use pumpkin_solver::variables::TransformableVariable;
use pumpkin_solver::Solver as PumpkinSolver;

use crate::variable::UnsolvedProblem;
use crate::{
    constraint::ConstraintReference,
    solvers::{ObjectiveDirection, ResolutionError, Solution, SolverModel},
};
use crate::{Constraint, Variable};

/// Scaling factor for continuous variables to preserve some precision
/// while keeping domain sizes manageable. (Pumpkin only supports integer vars.)
const SCALING_FACTOR: f64 = 100.0;

/// Create a Pumpkin-based solver model from an UnsolvedProblem.
pub fn pumpkin(to_solve: UnsolvedProblem) -> PumpkinProblem {
    let UnsolvedProblem {
        objective,
        direction,
        variables,
    } = to_solve;

    let mut pumpkin = PumpkinSolver::default();
    let mut var_map = HashMap::new();
    let mut objective_coeffs = HashMap::new();

    // Create variables in Pumpkin
    for (var, def) in variables.iter_variables_with_def() {
        let pumpkin_var = if def.is_integer {
            // For integer variables, we use the exact integer bounds
            pumpkin.new_bounded_integer(def.min as i32, def.max as i32)
        } else {
            // For continuous variables, scale them into integer range
            let scaled_min = (def.min * SCALING_FACTOR) as i32;
            let scaled_max = (def.max * SCALING_FACTOR) as i32;
            pumpkin.new_bounded_integer(scaled_min, scaled_max)
        };
        var_map.insert(var, pumpkin_var);
    }

    // Convert objective coefficients to a HashMap
    for (var, coef) in objective.linear.coefficients {
        objective_coeffs.insert(var, coef);
    }

    PumpkinProblem {
        pumpkin,
        var_map,
        objective_coeffs,
        objective_direction: direction,
        constraints_added: 0,
    }
}

/// A Pumpkin model
pub struct PumpkinProblem {
    pumpkin: PumpkinSolver,
    var_map: HashMap<Variable, pumpkin_solver::variables::DomainId>,
    objective_coeffs: HashMap<Variable, f64>,
    objective_direction: ObjectiveDirection,
    constraints_added: usize,
}

impl SolverModel for PumpkinProblem {
    type Solution = PumpkinSolverSolution;
    type Error = ResolutionError;

    fn solve(mut self) -> Result<Self::Solution, Self::Error> {
        // If there's no objective, just do one feasibility solve
        if self.objective_coeffs.is_empty() {
            let mut brancher = self
                .pumpkin
                .default_brancher_over_all_propositional_variables();
            let mut termination = Indefinite;
            let result = self.pumpkin.satisfy(&mut brancher, &mut termination);

            return match result {
                SatisfactionResult::Satisfiable(solution) => Ok(PumpkinSolverSolution {
                    solution,
                    var_map: self.var_map,
                }),
                SatisfactionResult::Unsatisfiable => Err(ResolutionError::Infeasible),
                SatisfactionResult::Unknown => Err(ResolutionError::Other(
                    "Solver terminated without finding a solution",
                )),
            };
        }

        // Otherwise, do iterative optimization
        let mut best_solution: Option<PumpkinSolution> = None;
        // We'll store an integer representation of the best objective in "scaled form"
        // (since we scaled variables). We'll handle sign flips for min vs max.
        let mut best_obj_scaled: i64 = match self.objective_direction {
            ObjectiveDirection::Maximisation => i64::MIN,
            ObjectiveDirection::Minimisation => i64::MAX,
        };

        // Build the integer expression for the objective. Each var's
        // scaled coefficient is (coeff * SCALING_FACTOR).
        // We'll store them in a Vec<ScaledLiteral>, so we can add constraints easily.
        let objective_expr = {
            let mut expr = Vec::new();
            for (var, coeff) in &self.objective_coeffs {
                let pid = self.var_map[var];
                // Multiply by SCALING_FACTOR again so we do everything in integer domain.
                let scaled_coeff = (*coeff * SCALING_FACTOR) as i32;
                if scaled_coeff != 0 {
                    expr.push(pid.scaled(scaled_coeff));
                }
            }
            expr
        };

        // For maximization, we might prefer to use a "negated" expression so we only add
        // "less_than_or_equals" constraints. We'll do that below.
        let negate_for_max = self.objective_direction == ObjectiveDirection::Maximisation;
        // Pre-build a negated version of the objective expression for ease of adding cuts
        let neg_objective_expr: Vec<_> = objective_expr.iter().map(|lit| lit.scaled(-1)).collect();

        // We'll keep re-solving until unsatisfiable or unknown
        let mut brancher = self
            .pumpkin
            .default_brancher_over_all_propositional_variables();
        let mut termination = Indefinite;

        loop {
            let result = self.pumpkin.satisfy(&mut brancher, &mut termination);
            match result {
                SatisfactionResult::Satisfiable(solution) => {
                    // Evaluate objective in *floating* terms to do normal math,
                    // but we also store an integer scaled version for the cut.
                    let objective_val =
                        compute_objective_value(&solution, &self.var_map, &self.objective_coeffs);
                    let scaled_obj_val = (objective_val * SCALING_FACTOR) as i64;

                    // Check if it's better than our current best
                    // - For max, better means scaled_obj_val > best_obj_scaled
                    // - For min,  scaled_obj_val < best_obj_scaled
                    let is_better = match self.objective_direction {
                        ObjectiveDirection::Maximisation => scaled_obj_val > best_obj_scaled,
                        ObjectiveDirection::Minimisation => scaled_obj_val < best_obj_scaled,
                    };

                    if is_better {
                        best_obj_scaled = scaled_obj_val;
                        best_solution = Some(solution.clone());
                    }

                    // Now add a “cut” that excludes all solutions *at least as bad*
                    // For Minimization: objective_expr <= best_obj_scaled - 1
                    // For Maximization: objective_expr >= best_obj_scaled + 1
                    // but we only have "less_than_or_equals", so:
                    //
                    // - Minimization cut: objective_expr <= best_obj_scaled - 1
                    // - Maximization cut: -objective_expr <= - (best_obj_scaled + 1)
                    //   i.e. neg_objective_expr <= -(best_obj_scaled + 1)
                    //
                    // That forces the solver to find a strictly better solution next time.

                    let cut_val = match self.objective_direction {
                        ObjectiveDirection::Minimisation => best_obj_scaled - 1,
                        ObjectiveDirection::Maximisation => -(best_obj_scaled + 1),
                    } as i32;

                    let expr_to_cut = if negate_for_max {
                        &neg_objective_expr
                    } else {
                        &objective_expr
                    };

                    // If adding the cut fails, we just stop
                    let cut_result = self
                        .pumpkin
                        .add_constraint(constraints::less_than_or_equals(
                            expr_to_cut.to_vec(),
                            cut_val,
                        ))
                        .post();

                    if let Err(e) = cut_result {
                        // If we fail to post the constraint, we can't continue
                        eprintln!("Error adding cut constraint: {:?}", e);
                        break;
                    }
                }
                SatisfactionResult::Unsatisfiable => {
                    // No more solutions better than the last best found => we're optimal
                    break;
                }
                SatisfactionResult::Unknown => {
                    // The solver gave up or timed out
                    break;
                }
            }
        }

        // Return the best solution found, or Infeasible if we never found any
        match best_solution {
            Some(sol) => Ok(PumpkinSolverSolution {
                solution: sol,
                var_map: self.var_map,
            }),
            None => Err(ResolutionError::Infeasible),
        }
    }

    fn add_constraint(&mut self, constraint: Constraint) -> ConstraintReference {
        let index = self.constraints_added;
        self.constraints_added += 1;

        // Build a Pumpkin linear expression
        let mut linear_terms = Vec::new();
        for (var, coeff) in constraint.expression.linear.coefficients {
            let pumpkin_var = self.var_map[&var];
            let scaled_coeff = (coeff * SCALING_FACTOR) as i32;
            if scaled_coeff != 0 {
                linear_terms.push(pumpkin_var.scaled(scaled_coeff));
            }
        }

        let constant = (constraint.expression.constant * SCALING_FACTOR) as i32;

        // Post the constraint
        if constraint.is_equality {
            if let Err(e) = self
                .pumpkin
                .add_constraint(constraints::equals(linear_terms, -constant))
                .post()
            {
                eprintln!("Warning: Could not add equality constraint: {:?}", e);
            }
        } else if let Err(e) = self
            .pumpkin
            .add_constraint(constraints::less_than_or_equals(linear_terms, -constant))
            .post()
        {
            eprintln!("Warning: Could not add inequality constraint: {:?}", e);
        }

        ConstraintReference { index }
    }

    fn name() -> &'static str {
        "Pumpkin"
    }
}

/// Compute the floating-point objective from a Pumpkin solution
fn compute_objective_value(
    solution: &PumpkinSolution,
    var_map: &HashMap<Variable, pumpkin_solver::variables::DomainId>,
    coeffs: &HashMap<Variable, f64>,
) -> f64 {
    let mut obj_val = 0.0;
    for (var, coeff) in coeffs {
        let pid = var_map[var];
        let var_val = solution.get_integer_value(pid) as f64 / SCALING_FACTOR;
        obj_val += var_val * coeff;
    }
    obj_val
}

/// A Pumpkin solution
pub struct PumpkinSolverSolution {
    solution: PumpkinSolution,
    var_map: HashMap<Variable, pumpkin_solver::variables::DomainId>,
}

impl Solution for PumpkinSolverSolution {
    fn value(&self, variable: Variable) -> f64 {
        let pid = self.var_map[&variable];
        let val = self.solution.get_integer_value(pid);
        val as f64 / SCALING_FACTOR
    }
}
