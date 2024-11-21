"""
Bilevel Mixed-Integer Programming (BMIP) Problem Generator and Analysis Module

This module provides functionality for generating and analyzing bilevel mixed-integer
programming problems, with specific focus on evaluating problem triviality and
generating non-trivial test cases.

The module includes utilities for:
- Calculating triviality metrics for generated problems
- Generating sets of non-trivial BMIP problems
- Managing problem files and metadata

Dependencies:
    - pyomo.environ: For mathematical modeling
    - tqdm: For progress tracking
    - json: For metadata handling
    - shutil: For file operations
    - os: For directory management
"""
#%%
from .problem_loading import load_problem, fixed_upper, setlowerobj
from .problem_generator import generate_problem
from pyomo.environ import ConcreteModel, SolverFactory
from tqdm import tqdm
import json
import shutil
import os
#%%
def Triviality_calculate(parameters, problems_name="problems_folder", N_eval=30, solver_name="gurobi"):
    """
    Calculate the triviality percentage of generated BMIP problems.
    
    A problem is considered trivial if the gap between its highpoint relaxation optimization (RO)
    objective and fixed upper-level (RF) objective is negligible (< 1e-6).
    
    Args:
        parameters (dict): Problem generation parameters
        problems_name (str, optional): Directory name for storing problems. Defaults to "problems_folder"
        N_eval (int, optional): Number of problems to evaluate. Defaults to 30
        solver_name (str, optional): Name of the solver to use. Defaults to "gurobi"
    
    Returns:
        float: Percentage of problems that were found to be trivial
    
    Side Effects:
        - Creates a directory specified by problems_name if it doesn't exist
        - Generates problem files in the specified directory
        - Removes trivial problem instances
        - Updates metadata.json for non-trivial problems
    """
    count = 0
    
    # Create and change to problems directory
    if not os.path.exists(problems_name):
        os.makedirs(problems_name)
    os.chdir(problems_name)

    # Evaluate problems with progress tracking
    for i in tqdm(range(N_eval), desc="Evaluating Problems"):
        problem_name = f"problem_{i - count + 1}"
        
        # Problem generation and model setup
        generate_problem(problem_name, parameters)
        mymodel = ConcreteModel()
        load_problem(problem_name, model=mymodel)

        # Solve relaxed optimization
        solver = SolverFactory(solver_name)
        solver.solve(mymodel)
        RO_Obj = mymodel.objective()

        # Solve fixed upper-level problem
        fixed_upper(model=mymodel)
        setlowerobj(model=mymodel)
        solver.solve(mymodel)
        RF_Obj = mymodel.upper_objective()

        # Calculate optimality gap
        gap = RF_Obj - RO_Obj

        # Handle trivial cases (gap < 1e-6)
        if abs(gap) < 1e-6:
            if os.path.exists(problem_name):
                shutil.rmtree(problem_name)
            count += 1
            continue

        # Update metadata for non-trivial problems
        metadata_file = os.path.join(problem_name, "metadata.json")
        metadata = {}
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as file:
                metadata = json.load(file)
        
        metadata.update({
            "RO_Obj": RO_Obj,
            "RF_Obj": RF_Obj,
            "Gap": gap
        })
        
        with open(metadata_file, "w") as file:
            json.dump(metadata, file, indent=4)

    os.chdir("..")
    
    trivial_percentage = count / N_eval * 100
    print(f"Trivial %: {trivial_percentage:.2f}%")
    return trivial_percentage

def nontrivial_BMIP_generator(parameters, N_gen=10, I=3, problems_name="problems_folder", solver_name="gurobi"):
    """
    Generate a specified number of non-trivial BMIP problems.
    
    This function continues generating problems until either the desired number of
    non-trivial cases is reached or the maximum number of attempts is exceeded.
    
    Args:
        parameters (dict): Problem generation parameters
        N_gen (int, optional): Number of non-trivial problems to generate. Defaults to 10
        I (int, optional): Multiplier for maximum attempts (max_attempts = I * N_gen). Defaults to 3
        problems_name (str, optional): Directory name for storing problems. Defaults to "problems_folder"
        solver_name (str, optional): Name of the solver to use. Defaults to "gurobi"
    
    Returns:
        tuple: (trivial_percentage, count_nontrivial)
            - trivial_percentage (float): Percentage of generated problems that were trivial
            - count_nontrivial (int): Number of non-trivial problems successfully generated
    
    Side Effects:
        - Creates a directory specified by problems_name if it doesn't exist
        - Generates problem files in the specified directory
        - Removes trivial problem instances
        - Updates metadata.json for non-trivial problems
    
    Notes:
        - A problem is considered trivial if |RF_Obj - RO_Obj| < 1e-6
        - The function will stop if either N_gen non-trivial problems are generated
          or if the maximum number of attempts (I * N_gen) is reached
    """
    count_trivial = 0
    count_nontrivial = 0
    max_iterations = I * N_gen

    # Setup problems directory
    if not os.path.exists(problems_name):
        os.makedirs(problems_name)
    os.chdir(problems_name)

    # Generate problems with progress tracking
    with tqdm(total=N_gen, desc="Generating Non-Trivial Cases") as pbar:
        for iteration in range(max_iterations):
            problem_name = f"problem_{count_nontrivial+1}"
            
            # Generate and solve problem
            generate_problem(problem_name, parameters)
            mymodel = ConcreteModel()
            load_problem(problem_name, model=mymodel)

            # Solve relaxed optimization
            solver = SolverFactory(solver_name)
            solver.solve(mymodel)
            RO_Obj = mymodel.objective()

            # Solve fixed upper-level problem
            fixed_upper(model=mymodel)
            setlowerobj(model=mymodel)
            solver.solve(mymodel)
            RF_Obj = mymodel.upper_objective()

            gap = RF_Obj - RO_Obj

            # Process based on triviality
            if abs(gap) < 1e-6:
                if os.path.exists(problem_name):
                    shutil.rmtree(problem_name)
                count_trivial += 1
            else:
                # Update metadata for non-trivial problems
                metadata_file = os.path.join(problem_name, "metadata.json")
                metadata = {}
                if os.path.exists(metadata_file):
                    with open(metadata_file, "r") as file:
                        metadata = json.load(file)
                
                metadata.update({
                    "RO_Obj": RO_Obj,
                    "RF_Obj": RF_Obj,
                    "Gap": gap
                })
                
                with open(metadata_file, "w") as file:
                    json.dump(metadata, file, indent=4)
                
                count_nontrivial += 1
                pbar.update(1)

            # Check termination conditions
            if count_nontrivial >= N_gen:
                break
            if iteration + 1 >= max_iterations:
                print(f"Reached maximum iterations ({max_iterations}).")
                break

    os.chdir("..")
    
    # Calculate and return statistics
    trivial_percentage = count_trivial / iteration * 100 if iteration > 0 else 0
    print(f"Trivial %: {trivial_percentage:.2f}%")
    print(f"Non-Trivial Cases Generated: {count_nontrivial}/{N_gen}")
    return trivial_percentage, count_nontrivial