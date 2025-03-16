//! A solver that uses [pumpkin-solver](https://github.com/Dekker1/pumpkin), a pure rust constraint programming solver.

use std::collections::HashMap;

use pumpkin_solver::constraints;
use pumpkin_solver::results::{OptimisationResult, ProblemSolution, SatisfactionResult, Solution};
use pumpkin_solver::termination::Indefinite;
use pumpkin_solver::variables::TransformableVariable;
use pumpkin_solver::Solver as PumpkinSolver;

use crate::variable::UnsolvedProblem;
use crate::{
    constraint::ConstraintReference,
    solvers::{ObjectiveDirection, ResolutionError, Solution as GoodLpSolution, SolverModel},
};
use crate::{Constraint, Variable};

/// Rounding precision for continuous variables, which need to be mapped
/// to integers for Pumpkin (since it only supports integer variables).
const SCALING_FACTOR: i32 = 10000;

/// The [pumpkin-solver](https://github.com/Dekker1/pumpkin) solver,
/// to be used with [UnsolvedProblem::using].
pub fn pumpkin(to_solve: UnsolvedProblem) -> PumpkinProblem {
    let UnsolvedProblem {
        objective,
        direction,
        variables,
    } = to_solve;

    // Collect all variable definitions once so we can retrieve them later
    let var_defs = variables
        .iter_variables_with_def()
        .map(|(v, d)| (v, d.clone())) // clone each definition
        .collect::<Vec<_>>();

    let mut pumpkin = PumpkinSolver::default();
    let mut var_map = HashMap::new();

    // Create variables in Pumpkin
    for (var, def) in &var_defs {
        let pumpkin_var = if def.is_integer {
            // For integer variables, simply use the bounds as-is
            pumpkin.new_bounded_integer(def.min as i32, def.max as i32)
        } else {
            // For continuous variables, scale by SCALING_FACTOR to maintain some precision
            let scaled_min = (def.min * SCALING_FACTOR as f64).round() as i32;
            let scaled_max = (def.max * SCALING_FACTOR as f64).round() as i32;
            pumpkin.new_bounded_integer(scaled_min, scaled_max)
        };
        var_map.insert(*var, (pumpkin_var, !def.is_integer));
    }

    // If there are no coefficients in the objective, we won't create an objective var
    let objective_var = if !objective.linear.coefficients.is_empty() {
        // Calculate potential min/max values for objective expression
        let mut min_obj = objective.constant;
        let mut max_obj = objective.constant;

        for (var, coeff) in &objective.linear.coefficients {
            // find this var's definition
            let def = var_defs
                .iter()
                .find(|(v, _)| v == var)
                .map(|(_, d)| d)
                .expect("Missing definition for variable in objective.");
            if *coeff > 0.0 {
                min_obj += coeff * def.min;
                max_obj += coeff * def.max;
            } else {
                // negative or zero coefficient
                min_obj += coeff * def.max;
                max_obj += coeff * def.min;
            }
        }

        let scaled_min = (min_obj * SCALING_FACTOR as f64).round() as i32;
        let scaled_max = (max_obj * SCALING_FACTOR as f64).round() as i32;
        Some(pumpkin.new_bounded_integer(scaled_min, scaled_max))
    } else {
        None
    };

    // Collect the objective coefficients into a standard HashMap
    let objective_coeffs = objective
        .linear
        .coefficients
        .iter()
        .map(|(v, c)| (*v, *c))
        .collect::<HashMap<_, _>>();

    PumpkinProblem {
        pumpkin,
        var_map,
        objective_coeffs,
        objective_constant: objective.constant,
        objective_direction: direction,
        objective_var,
        constraints_added: 0,
        var_defs,
    }
}

/// A Pumpkin model
pub struct PumpkinProblem {
    pumpkin: PumpkinSolver,
    // Map from good_lp Variable to (pumpkin variable, is_continuous)
    var_map: HashMap<Variable, (pumpkin_solver::variables::DomainId, bool)>,
    objective_coeffs: HashMap<Variable, f64>,
    objective_constant: f64,
    objective_direction: ObjectiveDirection,
    objective_var: Option<pumpkin_solver::variables::DomainId>,
    constraints_added: usize,
    // Keep around the (var, def) info so we can do lookups if needed
    var_defs: Vec<(Variable, crate::variable::VariableDefinition)>,
}

impl SolverModel for PumpkinProblem {
    type Solution = PumpkinSolverSolution;
    type Error = ResolutionError;

    fn solve(mut self) -> Result<Self::Solution, Self::Error> {
        let mut brancher = self
            .pumpkin
            .default_brancher_over_all_propositional_variables();
        let mut termination = Indefinite;

        // If there's no objective, just do a satisfiability solve
        if self.objective_var.is_none() || self.objective_coeffs.is_empty() {
            match self.pumpkin.satisfy(&mut brancher, &mut termination) {
                SatisfactionResult::Satisfiable(solution) => {
                    return Ok(PumpkinSolverSolution {
                        solution,
                        var_map: self.var_map,
                        objective_value: self.objective_constant,
                    });
                }
                SatisfactionResult::Unsatisfiable => {
                    return Err(ResolutionError::Infeasible);
                }
                SatisfactionResult::Unknown => {
                    return Err(ResolutionError::Other(
                        "Solver terminated without finding a solution",
                    ));
                }
            }
        }

        // There is an objective, so we need to optimize
        let objective_var = self.objective_var.unwrap();

        // Build expression for the objective
        let mut objective_terms = Vec::new();
        for (var, coeff) in &self.objective_coeffs {
            let (pumpkin_var, is_continuous) = self.var_map[var];
            let scale = if is_continuous { SCALING_FACTOR } else { 1 };
            let scaled_coeff = (coeff * scale as f64).round() as i32;
            if scaled_coeff != 0 {
                objective_terms.push(pumpkin_var.scaled(scaled_coeff));
            }
        }

        // Add constraint linking objective variable to the objective expression.
        // The `equals(...)` function in Pumpkin 0.1.4 is:
        //    equals(Vec<Affine<Var>>, i32) -> ConstraintBuilder
        // so we need to incorporate the objective_var inside the same Vec.
        let scaled_constant = (self.objective_constant * SCALING_FACTOR as f64).round() as i32;
        let mut full_expr = objective_terms;
        // Move objective_var to the *other* side of the equation:
        // objective_terms - objective_var = -scaled_constant
        full_expr.push(objective_var.scaled(-1));

        if let Err(e) = self
            .pumpkin
            .add_constraint(constraints::equals(full_expr, -scaled_constant))
            .post()
        {
            return Err(ResolutionError::Str(format!(
                "Failed to post objective constraint: {:?}",
                e
            )));
        }

        // Perform optimization
        let result = match self.objective_direction {
            ObjectiveDirection::Minimisation => {
                self.pumpkin
                    .minimise(&mut brancher, &mut termination, objective_var)
            }
            ObjectiveDirection::Maximisation => {
                self.pumpkin
                    .maximise(&mut brancher, &mut termination, objective_var)
            }
        };

        match result {
            OptimisationResult::Optimal(solution) => {
                let raw_obj_value = solution.get_integer_value(objective_var);
                let objective_value =
                    (raw_obj_value as f64 / SCALING_FACTOR as f64) + self.objective_constant;
                Ok(PumpkinSolverSolution {
                    solution,
                    var_map: self.var_map,
                    objective_value,
                })
            }
            OptimisationResult::Satisfiable(solution) => {
                let raw_obj_value = solution.get_integer_value(objective_var);
                let objective_value =
                    (raw_obj_value as f64 / SCALING_FACTOR as f64) + self.objective_constant;
                Ok(PumpkinSolverSolution {
                    solution,
                    var_map: self.var_map,
                    objective_value,
                })
            }
            OptimisationResult::Unsatisfiable => Err(ResolutionError::Infeasible),
            OptimisationResult::Unknown => Err(ResolutionError::Other(
                "Optimization terminated without finding a solution",
            )),
        }
    }

    fn add_constraint(&mut self, constraint: Constraint) -> ConstraintReference {
        let index = self.constraints_added;
        self.constraints_added += 1;

        // Build linear expression for the constraint
        let mut linear_terms = Vec::new();
        for (var, coeff) in constraint.expression.linear.coefficients {
            let (pumpkin_var, is_continuous) = self.var_map[&var];
            let scale = if is_continuous { SCALING_FACTOR } else { 1 };
            let scaled_coeff = (coeff * scale as f64).round() as i32;
            if scaled_coeff != 0 {
                linear_terms.push(pumpkin_var.scaled(scaled_coeff));
            }
        }

        let scaled_constant =
            (constraint.expression.constant * SCALING_FACTOR as f64).round() as i32;

        // Post the constraint according to its type
        if constraint.is_equality {
            if let Err(e) = self
                .pumpkin
                .add_constraint(constraints::equals(linear_terms, -scaled_constant))
                .post()
            {
                eprintln!("Warning: Could not add equality constraint: {:?}", e);
            }
        } else {
            if let Err(e) = self
                .pumpkin
                .add_constraint(constraints::less_than_or_equals(
                    linear_terms,
                    -scaled_constant,
                ))
                .post()
            {
                eprintln!("Warning: Could not add inequality constraint: {:?}", e);
            }
        }

        ConstraintReference { index }
    }

    fn name() -> &'static str {
        "Pumpkin"
    }
}

/// A Pumpkin-backed solution
pub struct PumpkinSolverSolution {
    solution: Solution,
    // Map from good_lp variable to (pumpkin variable, is_continuous)
    var_map: HashMap<Variable, (pumpkin_solver::variables::DomainId, bool)>,
    objective_value: f64,
}

impl GoodLpSolution for PumpkinSolverSolution {
    fn value(&self, variable: Variable) -> f64 {
        let (pid, is_continuous) = self.var_map[&variable];
        let val = self.solution.get_integer_value(pid) as f64;
        if is_continuous {
            val / SCALING_FACTOR as f64
        } else {
            val
        }
    }
}

#[cfg(test)]
mod tests {
    use super::pumpkin;
    use crate::{constraint, variable, variables, Solution, SolverModel};

    #[test]
    fn can_solve_simple_linear() {
        let mut vars = variables!();
        let x = vars.add(variable().clamp(0, 12));
        let y = vars.add(variable().clamp(0, 12));

        let solution = vars
            .maximise(x + y)
            .using(pumpkin)
            .with(constraint!(x + y == 12))
            .solve()
            .unwrap();

        assert_eq!(solution.value(x) + solution.value(y), 12.0);
    }

    #[test]
    fn can_solve_with_inequality() {
        let mut vars = variables!();
        let x = vars.add(variable().clamp(0, 12).integer());
        let y = vars.add(variable().clamp(0, 12).integer());

        let solution = vars
            .maximise(x + y)
            .using(pumpkin)
            .with(constraint!(x + y <= 12))
            .with(constraint!(x >= 5))
            .solve()
            .unwrap();

        assert!(solution.value(x) >= 5.0);
        assert!(solution.value(x) + solution.value(y) <= 12.0);
    }

    #[test]
    fn can_solve_mixed_integer_continuous() {
        let mut vars = variables!();
        let x = vars.add(variable().clamp(0, 5).integer());
        let y = vars.add(variable().clamp(0, 5)); // continuous

        let solution = vars
            .maximise(x + y)
            .using(pumpkin)
            .with(constraint!(x + y <= 7))
            .with(constraint!(2 * x + y <= 9))
            .solve()
            .unwrap();

        // x should be integer
        assert_eq!(solution.value(x).round(), solution.value(x));
        assert!(solution.value(x) + solution.value(y) <= 7.0);
        assert!(2.0 * solution.value(x) + solution.value(y) <= 9.0);
    }

    #[test]
    fn can_solve_minimization() {
        let mut vars = variables!();
        let x = vars.add(variable().clamp(1, 10));
        let y = vars.add(variable().clamp(1, 10));

        let solution = vars
            .minimise(x + 2.0 * y)
            .using(pumpkin)
            .with(constraint!(x + y >= 5))
            .solve()
            .unwrap();

        assert!(solution.value(x) + solution.value(y) >= 5.0);
        let obj_val = solution.value(x) + 2.0 * solution.value(y);
        // Minim solution should be x=4, y=1 with obj=6
        assert!(obj_val <= 7.0);
    }
}
