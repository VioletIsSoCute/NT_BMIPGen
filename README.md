# NT_BMIPGen

NT_BMIPGen is a Python library for generating bilevel mixed integer problems.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install NT_BMIPGen.

```bash
pip install NT_BMIPGen
```

## Parameters Define

| Category | Symbol | Description | Function of |
|----------|--------|-------------|-------------|
| **Variables - Upper Level** | x_ud | Decoupled continuous variables | - |
| | y_ud | Decoupled binary variables | - |
| | x_uc | Coupled continuous variables | - |
| | y_uc | Coupled binary variables | - |
| **Variables - Lower Level** | x_ld | Decoupled continuous variables | - |
| | y_ld | Decoupled binary variables | - |
| | x_lc | Coupled continuous variables | - |
| | y_lc | Coupled binary variables | - |
| **Constraints** | G_ud | Decoupled upper constraints | x_ud, y_ud |
| | g_ld | Decoupled lower constraints | x_ld, y_ld |
| | G_uc | Coupled upper constraints | x_ud, y_ud, x_uc, y_uc |
| | g_lc | Coupled lower constraints | x_ld, y_ld, x_lc, y_lc |
| | g_g | General constraints | x_uc, y_uc, x_lc, y_lc |
| **Objectives** | F_ud | Decoupled upper variables upper objective | x_ud, y_ud |
| | F_ld | Decoupled lower variables upper objective | x_ld, y_ld |
| | F_uc | Coupled variables upper objective | x_uc, y_uc, x_lc, y_lc |
| | f_ld | Decoupled lower variables lower objective | x_ld, y_ld |
| | f_lc | Coupled variables lower objective | x_uc, y_uc, x_lc, y_lc |

**Note:**
- Upper objective (F) = F_ud + F_ld + F_uc
- Lower objective (f) = f_ld + f_lc

## Usage

Function:

1. nontrivial_BMIP_generator make nontrivial problem folders, including the number of problems you want to generate, return trivial percent and number of constructed problems.

- N_gen (int, optional): Number of non-trivial problems to generate. Defaults to 10

- I (int, optional): Multiplier for maximum attempts (max_attempts = I * N_gen). Defaults to 3

- problems_name (str, optional): Directory name for storing problems. Defaults to "problems_folder"

- solver_name (str, optional): Name of the solver to use. Defaults to "gurobi"



```python
import NT_BMIPGen

# Make nontrivial problem folders, return trivial percent and number of constructed problems
parameters = {
    'x_ud':0, 'y_ud': 0, 'x_uc': 20, 'y_uc': 0,
    'x_ld': 0, 'y_ld': 0, 'x_lc': 20, 'y_lc': 0,
    'G_ud': 0, 'g_ld': 0, 'G_uc': 0, 'g_lc': 0, 'g_g': 20
}
NT_BMIPGen.nontrivial_BMIP_generator(parameters, N_gen=5, I=5, problems_name = "problems_folder", solver_name="gurobi")
```
## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Group website

[Avraamidou group](https://avraamidougroup.che.wisc.edu)

## License

[MIT](https://choosealicense.com/licenses/mit/)