#%%
import numpy as np
import os
import json
import pandas as pd
#%%
def generate_problem(folder, parameters):
    """
    Generate and save a bi-level optimization problem instance.
    
    This function creates a complete problem instance including constraint matrices,
    objective vectors, and associated parameters for both upper and lower level
    optimization problems.
    
    Args:
        folder (str): Directory path where the problem instance will be saved
        parameters (dict): Dictionary containing problem dimensions with keys:
            - x_ud, y_ud: Dimensions for upper-level decoupled (continuous/binary) variables
            - x_uc, y_uc: Dimensions for upper-level coupled (continuous/binary) variables
            - x_ld, y_ld: Dimensions for lower-level decoupled (continuous/binary) variables
            - x_lc, y_lc: Dimensions for lower-level coupled (continuous/binary) variables
    
    Returns:
        None: Files are saved directly to the specified folder
    
    Files generated:
        - metadata.json: Contains all problem parameters
        - {matrix_type}_A.csv: Constraint matrices
        - {matrix_type}_b.csv: Constraint vectors
        - {objective_type}.csv: Objective function vectors
    """

    # Create folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    
    # Save parameters to metadata.json
    with open(os.path.join(folder, 'metadata.json'), 'w') as f:
        json.dump(parameters, f, indent=4)

    # Helper function to create random matrix/vector with specified ranges
    def create_random_matrix(size, low, high,decimal=6):
        """
        Generate a random matrix/vector with specified dimensions and value ranges.
        
        Args:
            size (tuple): Dimensions of the matrix/vector to generate
            low (float): Lower bound for random values
            high (float): Upper bound for random values
            decimal (int): Float64 if True, Float 16 if False
        Returns:
            np.ndarray: Generated random matrix/vector
        """
        return np.round(np.random.uniform(low, high, size), int(decimal))
        
    # Variables and constraints sizes from parameters
    num_vars = {
        'x_ud': parameters['x_ud'],
        'y_ud': parameters['y_ud'],
        'x_uc': parameters['x_uc'],
        'y_uc': parameters['y_uc'],
        'x_ld': parameters['x_ld'],
        'y_ld': parameters['y_ld'],
        'x_lc': parameters['x_lc'],
        'y_lc': parameters['y_lc']
    }

    # Save random matrices and vectors for each constraint type
    constraints = {
        'G_ud': (num_vars['x_ud'] + num_vars['y_ud']),
        'g_ld': (num_vars['x_ld'] + num_vars['y_ld']),
        'G_uc': (num_vars['x_ud'] + num_vars['y_ud'] + num_vars['x_uc'] + num_vars['y_uc']),
        'g_lc': (num_vars['x_ld'] + num_vars['y_ld'] + num_vars['x_lc'] + num_vars['y_lc']),
        'g_g': (num_vars['x_uc'] + num_vars['y_uc'] + num_vars['x_lc'] + num_vars['y_lc'])
    }

    # Generate constraint matrices and vectors and save
    for c_type, size in constraints.items():
        A = create_random_matrix((size, size), -10, 10)
        b = create_random_matrix(size, 0, 10)
        pd.DataFrame(A).to_csv(os.path.join(folder, f"{c_type}_A.csv"), index=False)
        pd.DataFrame(b).to_csv(os.path.join(folder, f"{c_type}_b.csv"), index=False)

    # Generate objective vectors for upper and lower levels
    objectives = {
        'F_u': (num_vars['x_ud'] + num_vars['y_ud']),
        'F_l': (num_vars['x_ld'] + num_vars['y_ld']),
        'F_c': (num_vars['x_uc'] + num_vars['y_uc'] + num_vars['x_lc'] + num_vars['y_lc']),
        'ff_l': (num_vars['x_ld'] + num_vars['y_ld']),
        'ff_c': (num_vars['x_uc'] + num_vars['y_uc'] + num_vars['x_lc'] + num_vars['y_lc'])
    }
    # Generate and save objective vectors
    for o_type, size in objectives.items():
        # print(f"Generating file for {o_type} with size {size}")
        obj_vector = create_random_matrix(size, -10, 10)
        file_path = os.path.join(folder, f"{o_type}.csv")
        # print(f"Saving file at: {file_path}")
        pd.DataFrame(obj_vector).to_csv(file_path, index=False)
        # print(f"File saved: {file_path}")
#%%
# Example usage
'''
A. Variables

x_ud: Number of decoupled upper level continuous variable
y_ud: Number of decoupled upper level binary variable
x_uc: Number of dcoupled upper level continuous variable
y_uc: Number of dcoupled upper level binary variable
x_ld: Number of decoupled lower level continuous variable
y_ld: Number of decoupled lower level binary variable
x_lc: Number of dcoupled lower level continuous variable
y_lc: Number of dcoupled lower level binary variable

B. Constraints
G_ud: Number of decoupled upper constraints, function of x_ud, y_ud
g_ld: Number of decoupled lower constraints, function of x_ld, y_ld
G_uc: Number of coupled upper constraints, function of x_ud, y_ud, x_uc, y_uc
g_lc: Number of dcoupled lower constraints, function of x_ld, y_ld, x_lc, y_lc
g_g: Number of general constraints, function of x_uc, y_uc, x_lc, y_lc

C. Objectives
F_ud: decoupled upper variables upper objective, function of x_ud, y_ud
F_ld: decoupled lower variables upper objective, function of x_ld, y_ld
F_uc: dcoupled variables upper objective, function of x_uc, y_uc, x_lc, y_lc
f_ld: decoupled lower variables lower objective, function of x_ld, y_ld
f_lc: dcoupled variables lower objective, function of x_uc, y_uc, x_lc, y_lc

The upper objective F is defined by F_ud + F_ld + F_uc. 
The lower objective f is defined by f_ld + f_lc
'''
#%%
"""parameters = {
    'x_ud': 3, 'y_ud': 2, 'x_uc': 3, 'y_uc': 1,
    'x_ld': 3, 'y_ld': 2, 'x_lc': 3, 'y_lc': 1,
    'G_ud': 2, 'g_ld': 2, 'G_uc': 3, 'g_lc': 3, 'g_g': 2
}
generate_problem('problem_1', parameters)"""
# %%
