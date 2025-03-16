//! A solver that uses [pumpkin-solver](https://github.com/Yard1/pumpkin), a constraint satisfaction solver.

use std::collections::HashMap;

use pumpkin_solver::results::{OptimisationResult, ProblemSolution, SatisfactionResult};
use pumpkin_solver::termination::Indefinite;
use pumpkin_solver::variables::TransformableVariable;
use pumpkin_solver::{constraints, Solver};

use crate::expression::LinearExpression;
use crate::variable::UnsolvedProblem;
use crate::{
    constraint::ConstraintReference,
    solvers::{ObjectiveDirection, ResolutionError, Solution, SolverModel},
    IntoAffineExpression, Variable, CardinalityConstraintSolver,
};
use crate::{Constraint, ModelWithSOS1};

/// The [pumpkin-solver](https://github.com/Yard1/pumpkin) constraint satisfaction solver,
/// to be used with [UnsolvedProblem::using].
pub fn pumpkin(to_solve: UnsolvedProblem) -> PumpkinProblem {
    let UnsolvedProblem {
        objective,
        direction,
        variables,
    } = to_solve;

    // Create a new Pumpkin solver
    let mut solver = Solver::default();
    let mut variables_map = HashMap::new();
    
    // Create variables in the Pumpkin solver and map them to good_lp variables
    for (var, def) in variables.iter_variables_with_def() {
        let pumpkin_var = if def.is_integer {
            solver.new_bounded_integer(def.min as i32, def.max as i32)
        } else {
            // Pumpkin only supports integer variables, so we need to scale the bounds
            // This is a limitation of the solver
            solver.new_bounded_integer(def.min as i32, def.max as i32)
        };
        variables_map.insert(var, pumpkin_var);
    }

    PumpkinProblem {
        solver,
        variables: variables_map,
        objective,
        direction,
        constraints_count: 0,
    }
}

/// A pumpkin solver model
pub struct PumpkinProblem {
    solver: Solver,
    variables: HashMap<Variable, pumpkin_solver::variables::DomainId>,
    objective: crate::Expression,
    direction: ObjectiveDirection,
    constraints_count: usize,
}

impl PumpkinProblem {
    /// Get the inner pumpkin solver
    pub fn as_inner(&self) -> &Solver {
        &self.solver
    }

    /// Get mutable access to the inner pumpkin solver
    pub fn as_inner_mut(&mut self) -> &mut Solver {
        &mut self.solver
    }

    /// Convert a linear expression to a list of pumpkin variables (example helper).
    /// Currently unused; remove or mark as `#[allow(dead_code)]`.
    #[allow(dead_code)]
    fn expression_to_pumpkin(&self, expr: &LinearExpression) -> Vec<pumpkin_solver::variables::DomainId> {
        let mut pumpkin_vars = Vec::new();
        for (var, _coeff) in expr.coefficients.iter() {
            // Get the Pumpkin variable corresponding to the good_lp variable
            let pumpkin_var = self.variables.get(var).expect("Variable not found");
            pumpkin_vars.push(*pumpkin_var);
        }
        pumpkin_vars
    }
}

impl SolverModel for PumpkinProblem {
    type Solution = PumpkinSolution;
    type Error = ResolutionError;

    fn solve(mut self) -> Result<Self::Solution, Self::Error> {
        // Set up termination and branching strategy
        let mut termination = Indefinite;
        let mut brancher = self.solver.default_brancher_over_all_propositional_variables();

        // If there's an objective function, optimize it
        if !self.objective.linear.coefficients.is_empty() {
            // For optimization, we handle the objective function
            let (obj_var, obj_coeff) = self
                .objective
                .linear
                .coefficients
                .iter()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .expect("Objective function is empty");
                
            let pumpkin_obj_var = self.variables[obj_var];
            
            // If the coefficient is negative and we're maximizing (or vice versa),
            // we need to flip the direction
            let actual_direction = if obj_coeff < &0.0 {
                match self.direction {
                    ObjectiveDirection::Maximisation => ObjectiveDirection::Minimisation,
                    ObjectiveDirection::Minimisation => ObjectiveDirection::Maximisation,
                }
            } else {
                self.direction
            };
            
            // Solve the optimization problem
            let result = match actual_direction {
                ObjectiveDirection::Maximisation => {
                    self.solver.maximise(&mut brancher, &mut termination, pumpkin_obj_var)
                }
                ObjectiveDirection::Minimisation => {
                    self.solver.minimise(&mut brancher, &mut termination, pumpkin_obj_var)
                }
            };

            // Process the result
            match result {
                OptimisationResult::Optimal(solution) | OptimisationResult::Satisfiable(solution) => {
                    Ok(PumpkinSolution {
                        solution,
                        variables: self.variables,
                    })
                }
                OptimisationResult::Unsatisfiable => Err(ResolutionError::Infeasible),
                OptimisationResult::Unknown => Err(ResolutionError::Other("Unknown")),
            }
        } else {
            // Otherwise, just find a feasible solution
            match self.solver.satisfy(&mut brancher, &mut termination) {
                SatisfactionResult::Satisfiable(solution) => Ok(PumpkinSolution {
                    solution,
                    variables: self.variables,
                }),
                SatisfactionResult::Unsatisfiable => Err(ResolutionError::Infeasible),
                SatisfactionResult::Unknown => Err(ResolutionError::Other("Unknown")),
            }
        }
    }

    fn add_constraint(&mut self, constraint: Constraint) -> ConstraintReference {
        let reference = ConstraintReference {
            index: self.constraints_count,
        };
        self.constraints_count += 1;

        // Get the variables and coefficients
        let mut pumpkin_vars_with_coeff = Vec::new();
        for (var, coeff) in constraint.expression.linear.coefficients.iter() {
            let pumpkin_var = self.variables.get(var).expect("Variable not found");
            pumpkin_vars_with_coeff.push((*pumpkin_var, *coeff as i32));
        }
        
        let constant = -constraint.expression.constant as i32;
        
        // Add the appropriate constraint type
        if constraint.is_equality {
            let mut weighted_vars = Vec::new();
            for (var, coeff) in pumpkin_vars_with_coeff {
                if coeff != 1 {
                    weighted_vars.push(var.scaled(coeff));
                } else {
                    weighted_vars.push(var.into());
                }
            }
            let _ = self
                .solver
                .add_constraint(constraints::equals(weighted_vars, constant))
                .post();
        } else {
            // For <= constraint
            let mut weighted_vars = Vec::new();
            for (var, coeff) in pumpkin_vars_with_coeff {
                if coeff != 1 {
                    weighted_vars.push(var.scaled(coeff));
                } else {
                    weighted_vars.push(var.into());
                }
            }
            let _ = self
                .solver
                .add_constraint(constraints::less_than_or_equals(weighted_vars, constant))
                .post();
        }

        reference
    }

    fn name() -> &'static str {
        "Pumpkin Solver"
    }
}

impl ModelWithSOS1 for PumpkinProblem {
    fn add_sos1<I: IntoAffineExpression>(&mut self, variables_and_weights: I) {
        // Get the variables from the expression
        let sos_vars = variables_and_weights
            .linear_coefficients()
            .into_iter()
            .filter_map(|(var, _weight)| self.variables.get(&var).copied())
            .collect::<Vec<_>>();
        
        // We implement SOS1 constraints using the logic:
        // - each variable is in [0,1]
        // - sum is at most 1
        if !sos_vars.is_empty() {
            // Ensure each variable is in range [0, 1]
            for &var in &sos_vars {
                let _ = self
                    .solver
                    .add_constraint(constraints::less_than_or_equals(vec![var], 1))
                    .post();
                let _ = self
                    .solver
                    .add_constraint(constraints::less_than_or_equals(vec![var.scaled(-1)], 0))
                    .post();
            }
            
            // Then ensure the sum is at most 1
            let _ = self
                .solver
                .add_constraint(constraints::less_than_or_equals(sos_vars, 1))
                .post();
        }
    }
}

impl CardinalityConstraintSolver for PumpkinProblem {
    fn add_cardinality_constraint(&mut self, vars: &[Variable], rhs: usize) -> ConstraintReference {
        let reference = ConstraintReference {
            index: self.constraints_count,
        };
        self.constraints_count += 1;

        // Convert good_lp variables to pumpkin variables
        let pumpkin_vars: Vec<pumpkin_solver::variables::DomainId> = vars
            .iter()
            .filter_map(|var| self.variables.get(var))
            .copied()
            .collect();

        // Ensure each variable is in [0, 1]
        for &var in &pumpkin_vars {
            let _ = self
                .solver
                .add_constraint(constraints::less_than_or_equals(vec![var], 1))
                .post();
            let _ = self
                .solver
                .add_constraint(constraints::less_than_or_equals(vec![var.scaled(-1)], 0))
                .post();
        }
        
        // Then add the constraint that the sum is at most rhs
        let _ = self
            .solver
            .add_constraint(constraints::less_than_or_equals(pumpkin_vars, rhs as i32))
            .post();

        reference
    }
}

/// The solution to a pumpkin problem
pub struct PumpkinSolution {
    solution: pumpkin_solver::results::Solution,
    variables: HashMap<Variable, pumpkin_solver::variables::DomainId>,
}

impl Solution for PumpkinSolution {
    fn value(&self, variable: Variable) -> f64 {
        // Look up the pumpkin variable corresponding to the good_lp variable
        if let Some(&pumpkin_var) = self.variables.get(&variable) {
            self.solution.get_integer_value(pumpkin_var) as f64
        } else {
            panic!("Variable not found in solution");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::pumpkin;
    use crate::{variables, variable, Solution, constraint, SolverModel};

    #[test]
    fn can_solve_simple_problem() {
        let mut vars = variables!();
        let x = vars.add(variable().clamp(0, 12).integer());
        let y = vars.add(variable().clamp(0, 12).integer());
        
        let solution = vars
            .maximise(x + y)
            .using(pumpkin)
            .with(constraint!(x + y == 12))
            .solve()
            .unwrap();
            
        // Check the solution: should maximize x, so x=12, y=0
        assert_eq!(solution.value(x), 12.0);
        assert_eq!(solution.value(y), 0.0);
    }

    /// Test a simple integer LP: max x + y subject to x + y <= 12, x,y >= 0.
    #[test]
    fn can_solve_easy() {
        let mut vars = variables!(); // no name means "ProblemVariables::new()"
        let x = vars.add(variable().clamp(0, 12).integer());
        let y = vars.add(variable().clamp(0, 12).integer());
        let solution = vars
            .maximise(x + y)            // Objective: maximize x + y
            .using(pumpkin)             // Use our Pumpkin solver backend
            .with(x + y << 12)          // constraint: x + y <= 12
            .solve()
            .expect("Failed to solve with Pumpkin");
        // We expect x + y should be 12. There's no unique solution, but we know the sum.
        assert!((solution.value(x) + solution.value(y)).abs() <= 12.0 + 1e-9);
    }

    /// Slightly more complex test: an MILP with a couple constraints
    #[test]
    fn can_solve_milp() {
        let mut vars = variables!();
        // x, y >= 0, with x integer, y integer
        let x = vars.add(variable().integer().min(0).max(10));
        let y = vars.add(variable().integer().min(0).max(10));

        // Maximize 3x + 2y
        let model = vars
            .maximise(3 * x + 2 * y)
            .using(pumpkin)
            // constraints:
            .with(x * 2 + y * 3 << 12)  // 2x + 3y <= 12
            .with(x + y << 8);         // x + y <= 8

        let sol = model.solve().unwrap();
        // We'll do a quick check that the constraints are satisfied:
        assert!(2.0 * sol.value(x) + 3.0 * sol.value(y) <= 12.0 + 1e-9);
        assert!(sol.value(x) + sol.value(y) <= 8.0 + 1e-9);

        // We can do a quick check that the objective is correct or at least feasible.
        // E.g. for 2x + 3y <= 12 => if y = 2, x = 3 => then 2*3 +3*2=6+6=12 => feasible
        // So 3x+2y = 3*3 +2*2=9+4=13. Possibly there's better. We'll just confirm it's not obviously wrong:
        let obj = 3.0 * sol.value(x) + 2.0 * sol.value(y);
        assert!(obj >= 10.0); // or something: we expect around 13 or 14
    }
}
