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
def load_problem(folder, model, default_bounds=(-10, 10)):
    """
    Load and configure an optimization problem from files.
    
    Args:
        folder: Path to folder containing problem definition files
        model: Pyomo model instance to configure
        default_bounds: Default bounds for variables (min, max)
    """
    # Load parameters from metadata
    with open(os.path.join(folder, 'metadata.json'), 'r') as f:
        parameters = json.load(f)
    
    # Initialize variables
    var_types = {
        'x_ud': Reals, 'x_uc': Reals, 'x_ld': Reals, 'x_lc': Reals,
        'y_ud': Binary, 'y_uc': Binary, 'y_ld': Binary, 'y_lc': Binary
    }
    
    for var_name, domain in var_types.items():
        setattr(model, var_name, 
                Var(range(parameters[var_name]), 
                    domain=domain, 
                    bounds=default_bounds))
    
    # Initialize constraints list
    model.constraints = ConstraintList()
    
    def add_constraints(var_groups, file_prefix):
        """Helper function to add constraints for a group of variables"""
        if sum(parameters[x] for x in var_groups) == 0:
            return
            
        A = pd.read_csv(os.path.join(folder, f"{file_prefix}_A.csv")).values
        b = pd.read_csv(os.path.join(folder, f"{file_prefix}_b.csv")).values.flatten()
        k = np.cumsum([parameters[x] for x in var_groups])
        
        for i in range(b.size):
            expr = 0
            start_idx = 0
            for j, var_name in enumerate(var_groups):
                var = getattr(model, var_name)
                end_idx = k[j]
                cols = range(start_idx, end_idx)
                expr += sum(A[i, j] * var[j - start_idx] for j in cols)
                start_idx = end_idx
            model.constraints.add(expr <= b[i])
    
    def calculate_objective(var_groups, file_name):
        """Helper function to calculate objective terms"""
        if sum(parameters[x] for x in var_groups) == 0:
            return 0
            
        o = pd.read_csv(os.path.join(folder, f"{file_name}.csv")).values.flatten()
        k = np.cumsum([parameters[x] for x in var_groups])
        
        expr = 0
        start_idx = 0
        for j, var_name in enumerate(var_groups):
            var = getattr(model, var_name)
            end_idx = k[j]
            cols = range(start_idx, end_idx)
            expr += sum(o[j] * var[j - start_idx] for j in cols)
            start_idx = end_idx
        return expr
    
    # Add constraints
    constraint_groups = [
        (['x_ud', 'y_ud'], 'G_ud'),
        (['x_ld', 'y_ld'], 'g_ld'),
        (['x_ud', 'y_ud', 'x_uc', 'y_uc'], 'G_uc'),
        (['x_ld', 'y_ld', 'x_lc', 'y_lc'], 'g_lc'),
        (['x_uc', 'y_uc', 'x_lc', 'y_lc'], 'g_g')
    ]
    
    for var_groups, file_prefix in constraint_groups:
        add_constraints(var_groups, file_prefix)
    
    # Calculate objectives
    objective_groups = [
        (['x_ud', 'y_ud'], 'F_u', 'F_pu'),
        (['x_ld', 'y_ld'], 'F_l', 'F_pl'),
        (['x_uc', 'y_uc', 'x_lc', 'y_lc'], 'F_c', 'F_mu'),
        (['x_ld', 'y_ld'], 'ff_l', 'ff_pl'),
        (['x_uc', 'y_uc', 'x_lc', 'y_lc'], 'ff_c', 'ff_ml')
    ]
    
    objectives = {}
    for var_groups, file_name, obj_name in objective_groups:
        objectives[obj_name] = calculate_objective(var_groups, file_name)
    
    # Set final objectives
    model.upper_objective = (objectives['F_pu'] + 
                           objectives['F_pl'] + 
                           objectives['F_mu'])
    model.lower_objective = (objectives['ff_pl'] + 
                           objectives['ff_ml'])
    
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