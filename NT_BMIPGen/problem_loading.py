"""
Bi-level Optimization Problem Loader

This module loads and constructs a bi-level optimization problem using Pyomo.
It handles both continuous and binary variables, constraints, and objective functions
for upper and lower level problems.

Dependencies:
    - pyomo: For mathematical optimization modeling
    - pandas: For data input operations
    - numpy: For numerical operations
    - json: For metadata handling
    - os: For file operations
"""
#%%
from pyomo.environ import (
    Var, Binary, Reals, ConcreteModel, ConstraintList,
    Objective, minimize, SolverFactory, Param
)
import pandas as pd
import numpy as np
import json
import os
#%%
def load_problem(folder: str, model: ConcreteModel, default_bounds: tuple = (-10, 10)) -> None:
    """
    Load and construct a bi-level optimization problem from files.
    
    This function reads problem data from files and constructs the corresponding
    optimization model including variables, constraints, and objectives for both
    upper and lower level problems.
    
    Args:
        folder (str): Directory containing problem files
        model (ConcreteModel): Pyomo model to be populated
        default_bounds (tuple): Default bounds for variables (min, max)
    
    Returns:
        None: The model is modified in place
    
    Files required:
        - metadata.json: Problem dimensions and parameters
        - {constraint_type}_A.csv: Constraint matrices
        - {constraint_type}_b.csv: Constraint right-hand sides
        - {objective_type}.csv: Objective function coefficients
    """
    
    # Load problem dimensions from metadata
    with open(os.path.join(folder, 'metadata.json'), 'r') as f:
        parameters = json.load(f)

    # Define decision variables
    # Upper level variables (ud: upper decision, uc: upper constraint)
    model.x_ud = Var(range(parameters["x_ud"]), domain=Reals, bounds=default_bounds)
    model.x_uc = Var(range(parameters["x_uc"]), domain=Reals, bounds=default_bounds)
    # Lower level variables (ld: lower decision, lc: lower constraint)
    model.x_ld = Var(range(parameters["x_ld"]), domain=Reals, bounds=default_bounds)
    model.x_lc = Var(range(parameters["x_lc"]), domain=Reals, bounds=default_bounds)
    
    # Binary variables for both levels
    model.y_ud = Var(range(parameters["y_ud"]), domain=Binary, bounds=default_bounds)
    model.y_uc = Var(range(parameters["y_uc"]), domain=Binary, bounds=default_bounds)
    model.y_ld = Var(range(parameters["y_ld"]), domain=Binary, bounds=default_bounds)
    model.y_lc = Var(range(parameters["y_lc"]), domain=Binary, bounds=default_bounds)

    # Initialize constraint list
    model.constraints = ConstraintList()

    def add_constraints(xs: list, g_file: str):
        """
        Helper function to add constraints to the model.
        
        Args:
            xs (list): List of variable types involved in constraints
            g_file (str): Base name for constraint files
        """
        if sum([parameters[i] for i in xs]) != 0:
            # Load constraint matrix and vector
            A = pd.read_csv(os.path.join(folder, f"{g_file}_A.csv")).values
            b = pd.read_csv(os.path.join(folder, f"{g_file}_b.csv")).values.flatten()
            size = b.size
            k = np.cumsum(np.array([parameters[i] for i in xs]))
            
            # Add constraints to model
            for i in range(size):
                exprs = []
                # Build expressions for each variable type
                for idx in range(len(xs)):
                    start = k[idx-1] if idx > 0 else 0
                    var_name = f"model.{xs[idx]}"
                    expr = sum(A[i, j] * eval(var_name)[j - start] 
                             for j in range(start, k[idx]))
                    exprs.append(expr)
                # Add combined constraint
                model.constraints.add(expr = sum(exprs) <= b[i])

    # Add different types of constraints
    # Upper level decoupled constraints
    add_constraints(["x_ud", "y_ud"], 'G_ud')
    # Lower level decoupled constraints
    add_constraints(["x_ld", "y_ld"], 'g_ld')
    # Upper level coupled constraints
    add_constraints(["x_ud", "y_ud", "x_uc", "y_uc"], 'G_uc')
    # Lower level coupled constraints
    add_constraints(["x_ld", "y_ld", "x_lc", "y_lc"], 'g_lc')
    # Global constraints
    add_constraints(["x_uc", "y_uc", "x_lc", "y_lc"], 'g_g')

    def build_objective(xs: list, obj_file: str) -> float:
        """
        Helper function to build objective terms.
        
        Args:
            xs (list): List of variable types in objective
            obj_file (str): Base name for objective file
            
        Returns:
            float: Objective term value
        """
        if sum([parameters[i] for i in xs]) == 0:
            return 0
            
        o = pd.read_csv(os.path.join(folder, f"{obj_file}.csv")).values.flatten()
        k = np.cumsum(np.array([parameters[i] for i in xs]))
        
        exprs = []
        for idx in range(len(xs)):
            start = k[idx-1] if idx > 0 else 0
            var_name = f"model.{xs[idx]}"
            expr = sum(o[j] * eval(var_name)[j - start] 
                      for j in range(start, k[idx]))
            exprs.append(expr)
        return sum(exprs)

    # Build objective components
    F_u = build_objective(["x_ud", "y_ud"], 'F_u')  # Upper variables upper level
    F_l = build_objective(["x_ld", "y_ld"], 'F_l')  # Lower variables upper level
    F_c = build_objective(["x_uc", "y_uc", "x_lc", "y_lc"], 'F_c')  # Coupled variables upper level
    ff_l = build_objective(["x_ld", "y_ld"], 'ff_l')  # Lower variables lower level
    ff_c = build_objective(["x_uc", "y_uc", "x_lc", "y_lc"], 'ff_c')  # coupled variables lower level

    # Combine objectives
    model.upper_objective = F_u + F_l + F_c  # Total upper level objective
    model.lower_objective = ff_l + ff_c  # Total lower level objective
    
    # Set the model's objective
    model.objective = Objective(expr=model.upper_objective, sense=minimize)
#%%
"""
Helper Functions for Bi-level Optimization Model Management

This module provides utility functions for managing variable states and objectives
in a bi-level optimization model, specifically handling the fixing and unfixing
of upper-level variables and objective switching.
"""

def fixed_upper(model: 'ConcreteModel') -> None:
    """
    Fix upper-level variables to their current values.
    
    This function temporarily fixes all upper-level variables (both continuous and binary)
    to their current solved values. This is typically used when solving the lower-level
    problem while keeping the upper-level decisions constant.
    
    Args:
        model (ConcreteModel): The Pyomo optimization model containing the variables
        
    Notes:
        - Affects both decision (ud) and constraint (uc) variables
        - Only fixes variables that exist in the model (non-zero length)
        - Uses the current value of each variable as the fixed value
    """
    # List of all upper-level variable sets
    upper_params = [
        model.x_uc,  # Upper-level continuous coupled variables
        model.y_uc,  # Upper-level binary coupled variables
        model.x_ud,  # Upper-level continuous decoupled variables
        model.y_ud   # Upper-level binary decoupled variables
    ]
    
    # Fix each variable to its current value
    for upper_param in upper_params:
        if len(upper_param) != 0:  # Only process non-empty variable sets
            for i in upper_param:
                upper_param[i].fix(upper_param[i].value)
#%%
def setlowerobj(model: 'ConcreteModel') -> None:
    """
    Switch the model's objective to the lower-level objective.
    
    This function changes the model's objective function to optimize for the
    lower-level problem. This is typically used in bi-level optimization when
    solving the lower-level problem with fixed upper-level variables.
    
    Args:
        model (ConcreteModel): The Pyomo optimization model to modify
        
    Notes:
        - Modifies the existing objective in-place
        - Uses the pre-defined lower_objective expression
    """
    model.objective.set_value(model.lower_objective)
#%%
def unfixed_upper(model: 'ConcreteModel') -> None:
    """
    Unfix all upper-level variables.
    
    This function releases all upper-level variables from their fixed states,
    allowing them to be optimized again. This is typically used after solving
    the lower-level problem to resume upper-level optimization.
    
    Args:
        model (ConcreteModel): The Pyomo optimization model containing the variables
        
    Notes:
        - Affects both decision (ud) and constraint (uc) variables
        - Only processes variables that exist in the model (non-zero length)
        - Counterpart to fixed_upper() function
    """
    # List of all upper-level variable sets
    upper_params = [
        model.x_uc,  # Upper-level continuous coupled variables
        model.y_uc,  # Upper-level binary coupled variables
        model.x_ud,  # Upper-level continuous decoupled variables
        model.y_ud   # Upper-level binary decoupled variables
    ]
    
    # Unfix each variable
    for upper_param in upper_params:
        if len(upper_param) != 0:  # Only process non-empty variable sets
            for i in upper_param:
                upper_param[i].unfix()