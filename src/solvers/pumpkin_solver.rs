//! A solver that uses [Pumpkin](https://github.com/ConSol-Lab/Pumpkin), a pure rust solver.

use std::collections::HashMap;

use pumpkin_solver::constraints;
use pumpkin_solver::results::{SatisfactionResult, OptimisationResult, Solution as PumpkinSolution, ProblemSolution};
use pumpkin_solver::termination::Indefinite;
use pumpkin_solver::variables::TransformableVariable;
use pumpkin_solver::Solver as PumpkinSolver;

use crate::variable::UnsolvedProblem;
use crate::{
    constraint::ConstraintReference,
    solvers::{ObjectiveDirection, ResolutionError, Solution, SolverModel},
};
use crate::{Constraint, Variable};

/// The [Pumpkin](https://github.com/ConSol-Lab/Pumpkin) solver,
/// to be used with [UnsolvedProblem::using].
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
            // For integer variables, we use the exact bounds
            pumpkin.new_bounded_integer(def.min as i32, def.max as i32)
        } else {
            // For continuous variables, we need to convert to a reasonable integer range
            // Pumpkin only supports integer variables, so we'll need to scale appropriately
            let scaled_min = (def.min * 1000.0) as i32;
            let scaled_max = (def.max * 1000.0) as i32;
            pumpkin.new_bounded_integer(scaled_min, scaled_max)
        };
        var_map.insert(var, pumpkin_var);
    }

    // Convert objective coefficients to standard HashMap
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
        // Create an objective variable if there are objective coefficients
        let objective_var = if !self.objective_coeffs.is_empty() {
            // Create the objective variable with a reasonable range
            let obj_var = self.pumpkin.new_bounded_integer(i32::MIN / 2, i32::MAX / 2);
            
            // Create a linear combination for the objective
            let mut linear_terms = Vec::new();
            for (var, coeff) in &self.objective_coeffs {
                // Scale the coefficient for better precision
                let scaled_coeff = (*coeff * 1000.0) as i32;
                if scaled_coeff != 0 {
                    let pumpkin_var = self.var_map[var];
                    linear_terms.push(pumpkin_var.scaled(scaled_coeff));
                }
            }
            
            // Add the constraint linking the objective variable to the linear combination
            if !linear_terms.is_empty() {
                linear_terms.push(obj_var.scaled(-1));
                match self.pumpkin
                    .add_constraint(constraints::equals(linear_terms, 0))
                    .post() {
                    Ok(_) => {}
                    Err(_) => return Err(ResolutionError::Other("Failed to add objective constraint")),
                }
            }
            
            Some(obj_var)
        } else {
            None
        };

        // Create the brancher and termination condition
        let mut brancher = self.pumpkin.default_brancher_over_all_propositional_variables();
        let mut termination = Indefinite;

        // Solve the problem based on whether it's an optimization or satisfaction problem
        if let Some(obj_var) = objective_var {
            let opt_result = match self.objective_direction {
                ObjectiveDirection::Minimisation => {
                    self.pumpkin.minimise(&mut brancher, &mut termination, obj_var)
                }
                ObjectiveDirection::Maximisation => {
                    self.pumpkin.maximise(&mut brancher, &mut termination, obj_var)
                }
            };

            match opt_result {
                OptimisationResult::Optimal(solution) | OptimisationResult::Satisfiable(solution) => {
                    Ok(PumpkinSolverSolution {
                        solution,
                        var_map: self.var_map,
                    })
                }
                OptimisationResult::Unsatisfiable => {
                    Err(ResolutionError::Infeasible)
                }
                OptimisationResult::Unknown => {
                    Err(ResolutionError::Other("Solver terminated without finding a solution"))
                }
            }
        } else {
            // Simple satisfaction problem
            match self.pumpkin.satisfy(&mut brancher, &mut termination) {
                SatisfactionResult::Satisfiable(solution) => {
                    Ok(PumpkinSolverSolution {
                        solution,
                        var_map: self.var_map,
                    })
                }
                SatisfactionResult::Unsatisfiable => {
                    Err(ResolutionError::Infeasible)
                }
                SatisfactionResult::Unknown => {
                    Err(ResolutionError::Other("Solver terminated without finding a solution"))
                }
            }
        }
    }

    fn add_constraint(&mut self, constraint: Constraint) -> ConstraintReference {
        let index = self.constraints_added;
        self.constraints_added += 1;

        // Extract linear terms from the constraint
        let mut linear_terms = Vec::new();
        for (var, coeff) in constraint.expression.linear.coefficients {
            let pumpkin_var = self.var_map[&var];
            // Scale by 1000 to handle floating point coefficients
            let scaled_coeff = (coeff * 1000.0) as i32;
            if scaled_coeff != 0 {
                linear_terms.push(pumpkin_var.scaled(scaled_coeff));
            }
        }

        // Scale the constant term
        let constant = (constraint.expression.constant * 1000.0) as i32;

        // Add the appropriate constraint to Pumpkin
        if constraint.is_equality {
            self.pumpkin
                .add_constraint(constraints::equals(linear_terms, -constant))
                .post()
                .expect("Failed to add equality constraint");
        } else {
            self.pumpkin
                .add_constraint(constraints::less_than_or_equals(linear_terms, -constant))
                .post()
                .expect("Failed to add inequality constraint");
        }

        ConstraintReference { index }
    }

    fn name() -> &'static str {
        "Pumpkin"
    }
}

/// A Pumpkin solution
pub struct PumpkinSolverSolution {
    solution: PumpkinSolution,
    var_map: HashMap<Variable, pumpkin_solver::variables::DomainId>,
}

impl Solution for PumpkinSolverSolution {
    fn value(&self, variable: Variable) -> f64 {
        // Get the Pumpkin variable
        let pumpkin_var = self.var_map[&variable];
        
        // Get the integer value and convert it back to float (dividing by 1000.0 to undo scaling)
        let value = self.solution.get_integer_value(pumpkin_var);
        value as f64 / 1000.0
    }
}

#[cfg(test)]
mod tests {
    use crate::{constraint, variable, variables, Solution, SolverModel};
    use super::pumpkin;

    #[test]
    fn can_solve_with_inequality() {
        let mut vars = variables!();
        let x = vars.add(variable().clamp(0, 2));
        let y = vars.add(variable().clamp(1, 3));
        let solution = vars
            .maximise(x + y)
            .using(pumpkin)
            .with((2 * x + y) << 4)
            .solve()
            .unwrap();
        
        // Note: due to the scaling/rounding of floats to integers, 
        // we may need to allow for some imprecision
        let x_val = solution.value(x);
        let y_val = solution.value(y);
        
        assert!((x_val - 0.5).abs() < 0.01);
        assert!((y_val - 3.0).abs() < 0.01);
    }

    #[test]
    fn can_solve_with_equality() {
        let mut vars = variables!();
        let x = vars.add(variable().clamp(0, 2).integer());
        let y = vars.add(variable().clamp(1, 3).integer());
        let solution = vars
            .maximise(x + y)
            .using(pumpkin)
            .with(constraint!(2 * x + y == 4))
            .with(constraint!(x + 2 * y <= 5))
            .solve()
            .unwrap();
        
        let x_val = solution.value(x);
        let y_val = solution.value(y);
        
        assert!((x_val - 1.0).abs() < 0.01);
        assert!((y_val - 2.0).abs() < 0.01);
    }
}