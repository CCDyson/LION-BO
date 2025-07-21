import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

response_functions = []  # Initialize as list
for stack in range(7):
    # open a pickle file located in a 'stack-response-Functions' directory
    # the filename follows the pattern 'stack_{identifier}_response_function.pkl'
    with open(f"../LinOps/stack_response_functions/stack_{stack+1}_response_function.pkl", "rb") as file:
        response_func = pickle.load(file)
        response_functions.append(response_func)
        
# Test for response functions. Will delete later
# Define a kinetic energy range for testing (adjust as needed)
ke_test = np.linspace(0, 20, 500)  # energies from 0 to 20 MeV

stack_fn_cut = [3., 5.821401202856385, 7.997094322613386, 9.658781341633036, 
                    11.148286563497997, 12.435592394828737, 13.723555017136873]

print('--->Getting histogram data')

histograms = []

for i, cut in enumerate(stack_fn_cut):
    # Filter particles above threshold
    mask = ke_test > cut
    if not np.any(mask):
        histograms.append(np.zeros((50, 50)))
        continue
    
    energy_dict = {
    0: 'a_en_dep',
    1: 'b_en_dep',
    2: 'c_en_dep',
    3: 'd_en_dep',
    4: 'e_en_dep',
    5: 'f_en_dep',
    6: 'g_en_dep'
    }
    
    csv = energy_dict.get(i)
    csv_path = f"../LinOps/stack_response_functions/stack_response/{csv}.csv"
    df = pd.read_csv(csv_path)
    x_csv = df.iloc[:, 0]
    y_csv = df.iloc[:, 1]

    # Interpolation for CSV data
    interp_func = interp1d(x_csv, y_csv, kind='linear')
    energies = np.linspace(float(np.min(x_csv)), 25, 2000)
    deposited_energies_csv = interp_func(energies)
    

    ke_filtered = ke_test[mask]
    deposited_energy = response_functions[i](ke_filtered)
    plt.figure(figsize=(10, 6))
    plt.plot(ke_filtered, deposited_energy, label=f'New')
    plt.plot(energies, deposited_energies_csv, label='Original Data', color='orange')
    plt.xlabel('Kinetic Energy (MeV)')
    plt.ylabel('Deposited Energy (arb. units)')
    plt.title('Response Functions for Each Stack Layer')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'99-Scratch/response_functions_plot_{i}.png')
    
