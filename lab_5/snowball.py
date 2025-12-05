#!/usr/bin/env python3
"""
Lab 5 - Snowball Earth Model
Energy Balance Model (EBM) with 1D diffusion, spherical correction, and radiative forcing.

To reproduce the figures used in the report, run:
    python snowball_lab.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
from scipy.sparse import diags
import os


from snowball_functions import temp_warm, insolation


S0 = 1370.0              # Solar Flux [W/m^2]
C_WATER = 4.2e6          # Heat Capacity of Seawater [J m^-3 K^-1]
RHO_WATER = 1020.0       # Density of Seawater [kg/m^3]
R_EARTH = 6378e3         # Earth Radius [m] (Converted from km)
DZ = 50.0                # Mixed-layer depth [m]
SIGMA = 5.67e-8          # Stefan-Boltzmann constant [J m^-2 s^-1 K^-4]
DIFFUSIVITY = 100.0      # Thermal diffusivity (lambda) [m^2 s^-1]
EMISSIVITY = 1.0         # Emissivity (epsilon) [0-1]
ALBEDO_WATER = 0.3       # Albedo for water/ground
ALBEDO_ICE = 0.6         # Albedo for ice/snow
DT = 365 * 24 * 3600.0   # Time step: 1 year in seconds

class SnowballModel:
    """
    represent the 1D Snowball Earth Energy Balance Model (EBM).
    """

    def __init__(self, npoints=18):
        """
        initialize the model
        
        Parameters
        ----------
        npoints : int
            number of latitude bands
        """
        self.npoints = npoints
        
        # grid setup (0 = South, 180 = North)
        self.dlat = 180.0 / npoints
        self.lats = np.linspace(self.dlat/2, 180 - self.dlat/2, npoints)
        
        # dy = R * dtheta (radians)
        self.dy = R_EARTH * (self.dlat * np.pi / 180.0)
        
        # Axz: Cross-sectional area of the latitudinal ring
        # Axz ~ circumference * depth = 2*pi*R*cos(lat)*dz
        lat_rad = (self.lats - 90.0) * np.pi / 180.0
        self.Axz = 2 * np.pi * R_EARTH * np.cos(lat_rad) * DZ
        self._build_matrices()

    def _build_matrices(self):
        """
        This function constructs the implicit diffusion matrix (L) and first derivative matrix (B).
        """
        N = self.npoints
        
        # centered difference: (T_i+1 - 2T_i + T_i-1) / dy^2
        # neumann BCs: T_-1 = T_1 (South), T_N = T_N-2 (North)
        
        main_diag = -2.0 * np.ones(N)
        off_diag = np.ones(N-1)
        
        self.A = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
        
        # apply BCs to the corners of A
        self.A[0, 1] = 2.0   
        self.A[-1, -2] = 2.0 
        
        self.A /= (self.dy ** 2)

        # centered difference: (T_i+1 - T_i-1) / 2dy
        # used to calculate dAxz/dy and dT/dy
        
        off_diag_B = np.ones(N-1)
        self.B = np.zeros((N, N))
        self.B += np.diag(off_diag_B, k=1)
        self.B -= np.diag(off_diag_B, k=-1)
        
        self.B[0, 1] = 0.0
        self.B[-1, -2] = 0.0
        self.B /= (2 * self.dy)

        # L = I - lambda * dt * A
        self.L = np.eye(N) - DIFFUSIVITY * DT * self.A
        
        # Pre-compute inverse of L for the solver
        self.L_inv = np.linalg.inv(self.L)

    def solve(self, t_max_years=10000, use_spherical=True, use_radiative=True, 
              albedo_const=None, initial_temp=None, dynamic_albedo=True):
        """
        run the simulation.
        
        Parameters
        ----------
        t_max_years : int
            Number of years to simulate.
        use_spherical : bool
            Enable spherical correction term.
        use_radiative : bool
            Enable solar/blackbody radiative forcing.
        albedo_const : float or None
            If set, use this constant albedo everywhere. If None, use dynamic.
        initial_temp : array or None
            Initial temperature profile. If None, uses temp_warm().
        dynamic_albedo : bool
            If True and albedo_const is None, use ice/water albedo feedback.
            
        Returns
        -------
        temp : numpy array
            Final temperature profile.
        """
        
        # 1. initialize conditions
        if initial_temp is not None:
            T = initial_temp.copy()
        else:
            T = temp_warm(self.lats)
            
        # 2. compute insolation
        S = insolation(S0, self.lats)
        
        # 3. time loop
        steps = int(t_max_years) 
        

        # calcualte d(Axz)/dy
        dAxz_dy = self.B @ self.Axz 

        for _ in range(steps):
            
            alpha = np.zeros_like(T)
            if albedo_const is not None:
                alpha[:] = albedo_const
            elif dynamic_albedo:
                # Dynamic Albedo: Ice if <= -10C, else Water
                loc_ice = T <= -10.0
                alpha[loc_ice] = ALBEDO_ICE
                alpha[~loc_ice] = ALBEDO_WATER
            else:
                alpha[:] = ALBEDO_WATER # Default fallback

            spherical_corr = np.zeros_like(T)
            if use_spherical:
                # Calculate dT/dy
                dT_dy = self.B @ T
            
                spherical_corr = (DIFFUSIVITY * DT / self.Axz) * dT_dy * dAxz_dy

            radiative_forcing = np.zeros_like(T)
            if use_radiative:
                # Gain: Solar, Loss: Sigma * T^4 (Kelvin)
                # T is in Celsius, convert to Kelvin for Stefan-Boltzmann
                T_K = T + 273.15
                energy_balance = S * (1 - alpha) - EMISSIVITY * SIGMA * (T_K**4)
                
                radiative_forcing = (DT / (RHO_WATER * C_WATER * DZ)) * energy_balance

            # T_new = L_inv * (T_old + spherical + radiative)
            rhs = T + spherical_corr + radiative_forcing
            T_new = self.L_inv @ rhs
            
            T = T_new
            
        return T

def validation():
    """
    Validation routine to reproduce Figure 1 from the Lab Manual.
    Runs the model in 3 stages:
    1. Basic Diffusion (Red Line)
    2. Diff + Spherical Correction (Gold Line)
    3. Diff + SphCorr + Radiative (Green Line)
    """
    print("Running Validation Steps...")
    
    # Initialize Model
    model = SnowballModel(npoints=18)
    
    # Initial Condition (Blue Line)
    T_initial = temp_warm(model.lats)
    
    # Case 1: Basic Diffusion Only
    T_step1 = model.solve(t_max_years=10000, 
                          use_spherical=False, 
                          use_radiative=False, 
                          albedo_const=0.3)

    # Case 2: Spherical Correction
    T_step2 = model.solve(t_max_years=10000, 
                          use_spherical=True, 
                          use_radiative=False, 
                          albedo_const=0.3)
    
    # Case 3: Full Model (Green Line)
    T_step3 = model.solve(t_max_years=10000, 
                          use_spherical=True, 
                          use_radiative=True, 
                          albedo_const=0.3)

    if not os.path.exists("results"):
        os.makedirs("results")
        
    plt.figure(figsize=(8, 6))
    
    # 1. Initial Condition (Blue)
    plt.plot(model.lats, T_initial, label='Initial Condition', linewidth=2)
    
    # 2. Basic Diffusion (Red) - Interpreted as Cylindrical Radiative Balance
    plt.plot(model.lats, T_step1, color='red', label='Basic Diffusion', linewidth=2)
    
    # 3. Spherical Correction (Gold) - Interpreted as Spherical Radiative Balance
    plt.plot(model.lats, T_step2, color='gold', label='Diff + Spherical Correction', linewidth=2)
    
    # 4. Full Model (Green)
    plt.plot(model.lats, T_step3, color='green', label='Diff + SphCorr + Radiative', linewidth=2)
    
    plt.xlabel("Latitude")
    plt.ylabel(r"Temperature ($^\circ$C)")
    plt.title("Validation: Reproduction of Manual Figure 1")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, 180)
    
    output_path = "results/validation_figure.png"
    plt.savefig(output_path, dpi=150)
    print(f"Validation plot saved to {output_path}")
    plt.close()

def experiment1():
    """
    Experiment 1: Parameter Tuning.
    Fixed SyntaxWarning by using raw strings (r"") for LaTeX symbols.
    """
    print("\n--- Running Experiment 1: Parameter Tuning ---")
    
    if not os.path.exists("results"):
        os.makedirs("results")
        
    model = SnowballModel(npoints=18)
    T_warm_ref = temp_warm(model.lats)


    print("Performing Diffusivity Sweep...")
    diffusivities = [0, 50, 100, 150]
    fixed_epsilon = 1.0
    
    plt.figure(figsize=(8, 6))
    plt.plot(model.lats, T_warm_ref, 'k--', linewidth=2, label='Target (Warm Earth)')
    
    for D in diffusivities:
        global DIFFUSIVITY 
        DIFFUSIVITY = D
        model._build_matrices() 
        
        global EMISSIVITY
        EMISSIVITY = fixed_epsilon
        
        T_out = model.solve(t_max_years=4000, albedo_const=0.3)
        plt.plot(model.lats, T_out, label=rf'$\lambda$={D}')
        
    plt.title(rf"Effect of Diffusivity ($\lambda$) with $\epsilon$={fixed_epsilon}")
    plt.xlabel("Latitude")
    plt.ylabel(r"Temperature ($^\circ$C)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("results/exp1_diffusivity_sweep.png", dpi=150)
    plt.close()

    print("Performing Emissivity Sweep...")
    emissivities = [0.4, 0.6, 0.8, 1.0]
    fixed_diffusivity = 100
    
    plt.figure(figsize=(8, 6))
    plt.plot(model.lats, T_warm_ref, 'k--', linewidth=2, label='Target (Warm Earth)')
    
    for eps in emissivities:
        DIFFUSIVITY = fixed_diffusivity
        EMISSIVITY = eps
        model._build_matrices() 
        
        T_out = model.solve(t_max_years=4000, albedo_const=0.3)
        plt.plot(model.lats, T_out, label=rf'$\epsilon$={eps}')
        
    plt.title(rf"Effect of Emissivity ($\epsilon$) with $\lambda$={fixed_diffusivity}")
    plt.xlabel("Latitude")
    plt.ylabel(r"Temperature ($^\circ$C)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("results/exp1_emissivity_sweep.png", dpi=150)
    plt.close()

    print("Performing Grid Search for Best Parameters...")
    
    d_values = np.arange(0, 151, 10)       
    e_values = np.arange(0.5, 1.05, 0.05)  
    
    best_rms = float('inf')
    best_params = (None, None)
    best_T = None
    
    for D in d_values:
        for eps in e_values:
            DIFFUSIVITY = D
            EMISSIVITY = eps
            model._build_matrices()
            
            T_out = model.solve(t_max_years=4000, albedo_const=0.3)
            rms = np.sqrt(np.mean((T_out - T_warm_ref)**2))
            
            if rms < best_rms:
                best_rms = rms
                best_params = (D, eps)
                best_T = T_out.copy()
    
    print(f"Best Fit Found: Lambda={best_params[0]}, Epsilon={best_params[1]:.2f}")
    print(f"RMS Error: {best_rms:.4f} C")
    
    plt.figure(figsize=(8, 6))
    plt.plot(model.lats, T_warm_ref, 'k--', linewidth=2, label='Target (Warm Earth)')
    plt.plot(model.lats, best_T, 'r-', linewidth=2, 
             label=rf'Model ($\lambda$={best_params[0]}, $\epsilon$={best_params[1]:.2f})')
    
    plt.title("Experiment 1: Best Fit Tuning")
    plt.xlabel("Latitude")
    plt.ylabel(r"Temperature ($^\circ$C)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("results/exp1_best_fit.png", dpi=150)
    plt.close()

    # Set globals to best params
    global BEST_LAMBDA, BEST_EPSILON
    BEST_LAMBDA, BEST_EPSILON = best_params
    
    DIFFUSIVITY = BEST_LAMBDA
    EMISSIVITY = BEST_EPSILON
    model._build_matrices()

def experiment2():
    """
    Experiment 2: 
    Tests stability of Warm vs Snowball states using dynamic albedo.
    
    Scenarios:
    1. Hot Earth: Start at 60C everywhere (Dynamic Albedo).
    2. Cold Earth: Start at -60C everywhere (Dynamic Albedo).
    3. Flash Freeze: Start at Warm Earth Temp, but FORCE Albedo=0.6 (Fixed).
    """
    print("\n--- Running Experiment 2: Sensitivity to Initial Conditions ---")
    
    # Use best params from Exp 1, or default if not run
    if 'BEST_LAMBDA' in globals():
        D = BEST_LAMBDA
        E = BEST_EPSILON
    else:
        D = 30
        E = 0.70
        print(f"Warning: Exp 1 not run. Using default tuned values: Lambda={D}, Epsilon={E}")

    # initialize Model with Tuned Parameters
    global DIFFUSIVITY, EMISSIVITY
    DIFFUSIVITY = D
    EMISSIVITY = E
    model = SnowballModel(npoints=18)
    model._build_matrices()
    
    # hot earth
    print("Simulating Case 1: Hot Earth (60C)...")
    T_hot_init = np.ones(model.npoints) * 60.0
    T_hot_final = model.solve(t_max_years=10000, 
                              initial_temp=T_hot_init, 
                              dynamic_albedo=True)

    # cold earth
    print("Simulating Case 2: Cold Earth (-60C)...")
    T_cold_init = np.ones(model.npoints) * -60.0
    T_cold_final = model.solve(t_max_years=10000, 
                               initial_temp=T_cold_init, 
                               dynamic_albedo=True)

   
    print("Simulating Case 3: Flash Freeze (Warm Earth Init, Forced Albedo 0.6)...")
    T_warm_start = temp_warm(model.lats)
    # Force Albedo to 0.6 (Ice) and KEEP it there (Fixed)
    T_flash_final = model.solve(t_max_years=10000, 
                                initial_temp=T_warm_start, 
                                albedo_const=0.6)

    plt.figure(figsize=(8, 6))
    
    plt.axhline(y=-10, color='gray', linestyle=':', alpha=0.5, label=r'Freezing Threshold (-10$^\circ$C)')
    plt.plot(model.lats, temp_warm(model.lats), 'k--', alpha=0.3, label='Modern Warm Earth Ref')

    # Plot Experiment Results
    plt.plot(model.lats, T_hot_final, 'r-', linewidth=2, label=r'Start Hot (60$^\circ$C)')
    plt.plot(model.lats, T_cold_final, 'b-', linewidth=2, label=r'Start Cold (-60$^\circ$C)')
    plt.plot(model.lats, T_flash_final, 'c--', linewidth=2, label='Flash Freeze (Fixed $\\alpha$=0.6)')

    plt.title(rf"Sensitivity to Initial Conditions ($\lambda$={D}, $\epsilon$={E})")
    plt.xlabel("Latitude")
    plt.ylabel(r"Temperature ($^\circ$C)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-100, 60) 
    
    output_path = "results/exp2_initial_conditions.png"
    plt.savefig(output_path, dpi=150)
    print(f"Experiment 2 Plot saved to {output_path}")
    plt.close()

def experiment3():
    """
    Experiment 3: 
    
    Vary solar multiplier (gamma) from 0.4 -> 1.4 -> 0.4.
    Use previous result as initial condition for next step.
    Plot Global Average Temperature vs Gamma to show Hysteresis loop.
    """
    print("\n--- Running Experiment 3: Solar Forcing Hysteresis ---")
    
    if 'BEST_LAMBDA' in globals():
        D = BEST_LAMBDA
        E = BEST_EPSILON
    else:
        D = 30
        E = 0.70
    
    global DIFFUSIVITY, EMISSIVITY, S0
    DIFFUSIVITY = D
    EMISSIVITY = E
    
    ORIGINAL_S0 = 1370.0 
    
    model = SnowballModel(npoints=18)
    model._build_matrices() 
    
    # Define Gamma ranges
    # Use numpy for precise float steps
    gammas_up = np.arange(0.4, 1.41, 0.05)
    gammas_down = np.arange(1.4, 0.39, -0.05) 
    
    avg_temps_up = []
    avg_temps_down = []
    
    def get_weighted_avg(T_profile):
        return np.sum(T_profile * model.Axz) / np.sum(model.Axz)

    print("Running Hysteresis Loop: Ramp UP...")
    
    T_current = np.ones(model.npoints) * -60.0
    
    for g in gammas_up:
        # Update Global S0
        S0 = ORIGINAL_S0 * g
        
        # Run to steady state
        # Use previous T_current as initial condition (Path Dependence)
        T_current = model.solve(t_max_years=4000, 
                                initial_temp=T_current, 
                                dynamic_albedo=True)
        
        avg_temps_up.append(get_weighted_avg(T_current))


    print("Running Hysteresis Loop: Ramp DOWN...")
    
    
    for g in gammas_down:
        S0 = ORIGINAL_S0 * g
        
        T_current = model.solve(t_max_years=4000, 
                                initial_temp=T_current, 
                                dynamic_albedo=True)
        
        avg_temps_down.append(get_weighted_avg(T_current))

    # Restore S0 just in case
    S0 = ORIGINAL_S0


    plt.figure(figsize=(9, 7))
    
    plt.plot(gammas_up, avg_temps_up, 'b-o', label='Ramp UP (Start Cold)')
    
    plt.plot(gammas_down, avg_temps_down, 'r-o', label='Ramp DOWN (Start Warm)')
        
    plt.axvline(x=1.0, color='k', linestyle=':', alpha=0.3, label=r'Present Day ($\gamma=1.0$)')
    plt.axhline(y=-10, color='gray', linestyle=':', alpha=0.3)
    
    plt.title(rf"Snowball Earth Hysteresis Loop ($\lambda$={D}, $\epsilon$={E})")
    plt.xlabel(r"Solar Multiplier $\gamma$ ($S = \gamma S_0$)")
    plt.ylabel(r"Global Average Temperature ($^\circ$C)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = "results/exp3_hysteresis.png"
    plt.savefig(output_path, dpi=150)
    print(f"Experiment 3 Plot saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    validation()
    experiment1()
    experiment2()
    experiment3()