import numpy as np
import argparse 
import matplotlib.pyplot as plt 
""" 
Input Parameters: 
nlayers: (int) 
    Number of atmospheric layers (N)
emissivity: (float)
    Atmospheric emissivity (ε), 0-1
solar_constant: (float)
    Solar constant S₀ (W/m²), default 1350
albedo: (float)
    Planetary albedo (α), default 0.3
debug: (bool)
    Printing debug info

Return: 
temps: (array)
    Array of temperatures
"""
def energy_balance_model(nlayers, emissivity, solar_constant=1350, albedo=0.3, debug=False):

    # solar flux
    S = solar_constant * (1 - albedo) / 4
    sigma = 5.67e-8  # Stefan-Boltzmann constant
    
    
    # Create array of coefficients, an N+1 x N+1 array:
    A = np.zeros([nlayers+1, nlayers+1])
    b = np.zeros(nlayers+1)
    
    # Transmittance 
    size = nlayers+1

    for i in range(size):
        for j in range(size):
            if i == j: 
                if i == 0: # if own layer is surface
                    A[i][j] = -1
                else: # own layer is atomosphere layer
                    A[i][j] = -2
            elif j < i: # the below layer
                A[i][j] = emissivity * np.power((1-emissivity), (i-j-1))
            elif j > i: # the above layer
                A[i][j] = emissivity * np.power((1-emissivity), (j-i-1)) 
    # if in layer 0, then remove the emissivity
    A[0][1:] = A[0][1:] / emissivity

    b[0] = -S
    
    A_inv = np.linalg.inv(A)
    fluxes = np.matmul(A_inv, b)
    
    temperatures = np.zeros(size)

    # surface temperature
    temperatures[0] = np.power((fluxes[0] / sigma), 1/4)

    # layer temperature
    temperatures[1:] = np.power((fluxes[1:] / (sigma * emissivity)), 1/4)
    
    if debug:
        print("A:")
        print(A)
        print("b:")
        print(b)
        print("fluxes")
        print(fluxes)
        
    return temperatures

# To answer SQ 1, conducting the first experiment 
def run_emissivity_experiment1():
    emissivities = np.linspace(0.05, 0.95, 50)
    temps = []
    for ems in emissivities:
        # collect surface temperature
        temps.append(energy_balance_model(nlayers=1, emissivity=ems)[0])

    idx_288 = np.argmin(np.abs(np.array(temps) - 288))
    eps_288 = emissivities[idx_288]
    print(f'Emissivity when surface temperature reaches 288: {eps_288:.3f}')
    
    plt.figure(figsize=(10, 6))
    plt.plot(emissivities, temps, 'b-', linewidth=2, label='Model Prediction')
    plt.axhline(y=288, color='r', linestyle='--', alpha=0.7, label='Earth average temperature (288K)')
    plt.axvline(x=0.255, color='g', linestyle='--', alpha=0.7, label='Other studies (0.255)')
    plt.axvline(x=eps_288, color='r', linestyle='--', alpha=0.7, label=f'Emissivity at 288K ({eps_288:.3f})')
    
    plt.xlabel('Emissivity')
    plt.ylabel('Surface temperature (K)')
    plt.title('Surface Temperature vs Emissivity in one layer model')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/experiment1_emissivity_vs_temp.png', dpi=300)
    plt.show()

def run_emissivity_experiment2():
    sample_emissivity = 0.255
    temps = []
    nlayers_range = range(1, 21)
    for n in nlayers_range:
        temps.append(energy_balance_model(nlayers=n, emissivity=sample_emissivity)[0])
    differences = np.abs(np.array(temps) - 288)
    best_idx = np.argmin(differences)
    best_layers = nlayers_range[best_idx]
    print(f"The number of layers that the surface temperature closest to 288K: {best_layers}")
    plt.figure(figsize=(10, 6))
    plt.plot(nlayers_range, temps, 'bo-', linewidth=2, markersize=6)
    plt.axhline(y=288, color='r', linestyle='--', alpha=0.7, label='Earth Average temperature (288K)')
    plt.axvline(x=best_layers, color='r', linestyle='--', alpha=0.7, 
                label=f'Best layers ({best_layers}, {temps[best_idx]:.1f}K)')
    
    plt.xlabel('Number of Layers')
    plt.ylabel('Surface Temperature (K)')
    plt.title('Surface Temperature vs Number of Layers at emissivity = 0.255')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/experiment2_layers_vs_temp.png', dpi=300)
    plt.show()

def run_Venus_experiment():
    S0 = 2600 # solar constant
    albedo_venus = 0.75
    target_temp = 700
    n_layers_range = range(1, 100)
    temps = []
    for n in n_layers_range:
        temps.append(energy_balance_model(nlayers=n, emissivity=1, solar_constant= S0, albedo = albedo_venus)[0])

    differences = np.abs(np.array(temps) - target_temp)
    best_idx = np.argmin(differences)
    best_layers = n_layers_range[best_idx]
    print(f"The number of layers that the Venus surface temperature closest to 700K: {best_layers}")
    plt.figure(figsize=(10, 6))
    plt.plot(n_layers_range, temps, 'ro-', linewidth=2, markersize=6, label='Model Prediction')
    plt.axhline(y=target_temp, color='b', linestyle='--', alpha=0.7, 
                label=f'Target Venus Surface Temperature ({target_temp}K)')
    plt.axvline(x=best_layers, color='r', linestyle='--', alpha=0.7, 
                label=f'Best Layers ({best_layers}, {temps[best_idx]:.1f}K)')
    
    plt.xlabel('Atmosphere Layers (N)')
    plt.ylabel('Surface Temperature (K)')
    plt.title('Venus, Surface Temperature vs atmosphere layer at emissivity = 1')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/venus_layers_experiment.png', dpi=300)
    plt.show()
    
def run_nuclear_winter_experiment():
    """
    This model will assume the top layer absorb all incoming solar flux
    """
    def nuclear_winter_model(nlayers, emissivity, solar_constant=1350, albedo=0.3, debug=False):
        S = solar_constant * (1 - albedo) / 4
        sigma = 5.67e-8  # Stefan-Boltzmann constant
        
        
        # Create array of coefficients, an N+1 x N+1 array:
        A = np.zeros([nlayers+1, nlayers+1])
        b = np.zeros(nlayers+1)
        
        # Transmittance 
        size = nlayers+1

        for i in range(size):
            for j in range(size):
                if i == j: 
                    if i == 0: # if own layer is surface
                        A[i][j] = -1
                    else: # own layer is atomosphere layer
                        A[i][j] = -2
                elif j < i: # the below layer
                    A[i][j] = emissivity * np.power((1-emissivity), (i-j-1))
                elif j > i: # the above layer
                    A[i][j] = emissivity * np.power((1-emissivity), (j-i-1)) 
        # if in layer 0, then remove the emissivity
        A[0][1:] = A[0][1:] / emissivity

        b[-1] = -S
        
        A_inv = np.linalg.inv(A)
        fluxes = np.matmul(A_inv, b)
        
        temperatures = np.zeros(size)

        # surface temperature
        temperatures[0] = np.power((fluxes[0] / sigma), 1/4)

        # layer temperature
        temperatures[1:] = np.power((fluxes[1:] / (sigma * emissivity)), 1/4)
        
        if debug:
            print("A:")
            print(A)
            print("b:")
            print(b)
            print("fluxes")
            print(fluxes)
            
        return temperatures
    
    surface_temp = nuclear_winter_model(nlayers=5, emissivity=0.5, debug=True)
    
    plt.figure(figsize=(10, 8))
    plt.plot(surface_temp, range(len(surface_temp)), 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Temperature (K)')
    plt.ylabel('Altitude')
    plt.title('Nuclear Winter Altitude vs Temperature')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/nuclear_winter_temperature_profile.png', dpi=300)
    plt.show()
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=int, default=1)
    args = parser.parse_args()
    if args.experiment == 1:
        run_emissivity_experiment1()
    elif args.experiment == 2:
        run_emissivity_experiment2()
    elif args.experiment == 3: 
        run_Venus_experiment()
    elif args.experiment == 4:
        run_nuclear_winter_experiment()
    
if __name__ == "__main__":
    main()
