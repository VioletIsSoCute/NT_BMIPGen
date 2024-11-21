#%%
from triviality import nontrivial_BMIP_generator, Triviality_calculate
import numpy as np
import matplotlib.pyplot as plt

#%%
parameters = {
    'x_ud':0, 'y_ud': 0, 'x_uc': 20, 'y_uc': 0,
    'x_ld': 0, 'y_ld': 0, 'x_lc': 20, 'y_lc': 0,
    'G_ud': 0, 'g_ld': 0, 'G_uc': 0, 'g_lc': 0, 'g_g': 20
}
nontrivial_BMIP_generator(parameters, N_gen=5, I=5, problems_name = "problems_folder", solver_name="gurobi")
# %% Heatmap

# Step 1: Define the function f(a, b)
def f(a, b):
    parameters = {
    'x_ud':0, 'y_ud': 0, 'x_uc': a, 'y_uc': 0,
    'x_ld': 0, 'y_ld': 0, 'x_lc': b, 'y_lc': 0,
    'C_ud': 0, 'C_ld': 0, 'C_uc': 0, 'C_lc': 0, 'C_g': 20
}
    return  Triviality_calculate(parameters,N_eval=30) # Example function, replace with your function

# Step 2: Create a range of values for a and b
a_values = np.linspace(2, 20, 10)
b_values = np.linspace(2, 20, 10)

# Step 3: Create a grid and initialize Z for results
Z = np.zeros((len(a_values), len(b_values)))

# Evaluate f(a, b) point by point to avoid passing arrays
for i, a in enumerate(a_values):
    for j, b in enumerate(b_values):
        Z[j, i] = f(int(a), int(b))  # Store each result in Z
#%%
# Step 4: Plot the heatmap
# Assuming Z has already been computed
plt.figure(figsize=(8, 6))
plt.imshow(Z, extent=[2, 20, 2, 20], origin='lower', cmap='viridis', aspect='auto')

# Colorbar with larger font size
cbar = plt.colorbar(label="Trivial %")
cbar.ax.tick_params(labelsize=14)
cbar.set_label("Trivial %", fontsize=16)
cbar.mappable.set_clim(0, 100)

# Increase font size for labels and title
plt.xlabel("Number of coupled upper level variables", fontsize=16)
plt.ylabel("Number of coupled lower level variables", fontsize=16)
plt.title("Heatmap of Trivial Percentage", fontsize=18)

# Display the plot
plt.show()
# %%
