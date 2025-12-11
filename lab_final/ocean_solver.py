#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# Import the ground truth solution
import test_solution

def solve_heat_equation(z_min, z_max, t_min, t_max, dz, dt, Kz_profile,
                        initial_condition, boundary_conditions, source_term_func=None):
    """
    Solve the 1D heat equation.

    Parameters
    ----------
    z_min : float
        minimum depth in meters.
    z_max : float
        maximum depth in meters.
    t_min : float
        start time in seconds.
    t_max : float
        end time in seconds.
    dz : float
        spatial step size.
    dt : float
        time step size.
    kz_profile : array
        diffusivity values for each depth point.
    initial_condition : function or array
        temperature at time zero.
    boundary_conditions : dict
        dictionary with upper and lower conditions.
    source_term_func : function, optional
        function to add external heat source.

    Returns
    -------
    time : array
        array of time steps.
    z : array
        array of depth steps.
    temp : array
        2d array of temperature (depth x time).
    """
    # 1. Create Grids
    z = np.arange(z_min, z_max + dz, dz)
    time = np.arange(t_min, t_max + dt, dt)
    
    n_space = len(z)
    n_time = len(time)

    # 2. Input Validation
    if len(Kz_profile) != n_space:
        raise ValueError(f"Kz_profile length ({len(Kz_profile)}) must match z grid length ({n_space}).")

    # 3. Calculate Stability Vector 'r'
    r = Kz_profile * dt / (dz**2)
    if np.any(r > 0.5):
        print("  [Warning] Stability condition violated (r > 0.5) at some depths.")

    # 4. Initialize Solution Array (All Zeros)
    temp = np.zeros((n_space, n_time))

    if callable(initial_condition):
        temp[:, 0] = initial_condition(z)
    else:
        temp[:, 0] = initial_condition
    

    for j in range(n_time - 1):
        current_t = time[j]
        
        # get temperatures at current step
        T_i = temp[1:-1, j]
        T_ip1 = temp[2:, j]
        T_im1 = temp[:-2, j]
        
        # get K at half-steps (Arithmetic Mean)
        K_center = Kz_profile[1:-1]
        K_right = Kz_profile[2:]
        K_left = Kz_profile[:-2]
        
        K_plus = 0.5 * (K_center + K_right)
        K_minus = 0.5 * (K_center + K_left)
        
        # calculate Fluxes
        flux_right = K_plus * (T_ip1 - T_i) / dz
        flux_left = K_minus * (T_i - T_im1) / dz
        
        # update interior
        diffusion_term = T_i + (dt / dz) * (flux_right - flux_left)
        
        # apply source (if any)
        if source_term_func:
            source = source_term_func(z[1:-1], current_t)
            temp[1:-1, j+1] = diffusion_term + source * dt
        else:
            temp[1:-1, j+1] = diffusion_term

        # Upper Boundary (Surface)
        bc_upper = boundary_conditions['upper']
        if callable(bc_upper):
            # pass the time to next step
            temp[0, j+1] = bc_upper(time[j+1]) 
        else:
            temp[0, j+1] = bc_upper

        # Lower Boundary (Deep)
        bc_lower = boundary_conditions['lower']
        if callable(bc_lower):
            temp[-1, j+1] = bc_lower(time[j+1])
        else:
            temp[-1, j+1] = bc_lower
            
    return time, z, temp


def create_diffusivity_profile(z_grid, ml_depth=50.0):
    """
    Creates the step-function diffusivity profile.
    Top 50m: 1e-2 (Mixed Layer)
    Below 50m: 1e-4 (Thermocline)

    Parameters
    ----------
    z_grid : array
        spatial grid points.
    ml_depth : float
        depth of the mixed layer.

    Returns
    -------
    kz : array
        diffusivity profile step function.
    """
    K_surface = 1e-2
    K_deep = 1e-4
    Kz = np.ones_like(z_grid) * K_deep
    
    # Apply high mixing to Mixed Layer
    # Using logical indexing
    Kz[z_grid <= ml_depth] = K_surface
    return Kz

def make_sst_function(warming_rate_per_century=0.0):
    """
    factory function that returns a SST(t) function.

    Parameters
    ----------
    warming_rate_per_century : float
        warming trend in degrees celsius per 100 years.

    Returns
    -------
    sst : function
        function that returns surface temperature at a given time.
    """
    # Convert rate to degrees per second
    seconds_in_century = 100 * 365 * 24 * 3600
    rate_per_sec = warming_rate_per_century / seconds_in_century
    
    def sst(t):
        year_sec = 365 * 24 * 3600
        # Seasonal: Mean 15C, Amp 5C
        # Phase shift -pi/2 to align min with start of year (Winter) or adjust as needed
        seasonal = 5.0 * np.sin(2 * np.pi * t / year_sec - np.pi/2)
        trend = rate_per_sec * t
        return 15.0 + seasonal + trend
        
    return sst

def validate_solver():
    """
    Validate the solver using the manufactured solutions.
    Compares the numerical result against the ground truth
    in 'test_solution.py'.
    """
    
    # 1. setup parameters 
    z_min, z_max = 0.0, 1.0
    t_min, t_max = 0.0, 0.1
    dz = 0.1
    dt = 0.0001
    
    # 2. variable diffusivity 
    z_grid = np.arange(z_min, z_max + dz, dz)
    Kz_test = 1.0 + z_grid
    
    # 3. source term S(z,t)
    def source_term(z, t):
        pi = np.pi
        # dU/dt - d/dz(K dU/dz)
        dA = 100 * np.sin(pi * z)
        dB = 100 * pi * t * (np.cos(pi * z) - pi * (1 + z) * np.sin(pi * z))
        return dA - dB
    
    # 4. define initial conditino and boundary
    def initial_cond(z):
        return np.zeros_like(z)

    boundary_conds = {
        'upper': 0.0,
        'lower': 0.0
    }

    # 5. Run Solver
    time, z, res = solve_heat_equation(
        z_min, z_max, t_min, t_max, dz, dt, Kz_test,
        initial_cond, boundary_conds, source_term_func=source_term
    )

    # 6. retrieve Ground Truth
    expected = test_solution.mms_ground_truth

    # 7. compare Results
    final_output = res[:, -1]
    

    print(f"Results at t = {t_max:.1f}")
    print("-" * 65)
    print(f"{'Index':<6} | {'Depth (z)':<10} | {'Truth':<10} | {'Model':<10} | {'Error':<10}")
    print("-" * 65)
    
    for i in range(len(z)):
        err = abs(final_output[i] - expected[i])
        print(f"{i:<6} | {z[i]:<10.2f} | {expected[i]:<10.4f} | {final_output[i]:<10.4f} | {err:<10.4f}")
        
    max_error = np.max(np.abs(final_output - expected))
    print(max_error)

def experiment1():
    """
    Experiment 1: Simulating the Seasonal Ocean Structure (Control Run).
    Goal: Identify the Mixed Layer.
    """

    # 1. Domain
    z_min, z_max = 0.0, 100.0
    dz = 0.5
    z_grid = np.arange(z_min, z_max + dz, dz)

    # 2. parameters
    # Create step-function Kz
    Kz = create_diffusivity_profile(z_grid, ml_depth=50.0)
    
    # Time settings: Run for 5 years
    years = 5
    t_max = years * 365 * 24 * 3600

    dt = 10.0 
    
    # boundary conditions (Control: 0 warming)
    bc_funcs = {
        'upper': make_sst_function(warming_rate_per_century=0.0),
        'lower': 4.0 
    }
    
    # initial Condition: Uniform 4C
    def initial_cond(z):
        return np.ones_like(z) * 4.0

    time, z, heat = solve_heat_equation(
        z_min, z_max, 0, t_max, dz, dt, Kz,
        initial_cond, bc_funcs
    )
    
    # 1. Convert time to years
    time_years = time / (365 * 24 * 3600)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # plot 1:
    step = 1000  
    im = ax1.pcolormesh(time_years[::step], z, heat[:, ::step], 
                        cmap='RdBu_r', shading='auto', vmin=4, vmax=22)
    ax1.invert_yaxis()
    ax1.set_title('Time-Depth Temperature Map')
    ax1.set_xlabel('Time (Years)')
    ax1.set_ylabel('Depth (m)')
    plt.colorbar(im, ax=ax1, label='Temperature (°C)')

    # plot 2: Vertical Profiles (Last Year)
    steps_per_year = int(365 * 24 * 3600 / dt)
    last_year_data = heat[:, -steps_per_year:]
    
    summer_idx = np.argmax(last_year_data[0, :])
    winter_idx = np.argmin(last_year_data[0, :])
    
    summer_profile = last_year_data[:, summer_idx]
    winter_profile = last_year_data[:, winter_idx]

    ax2.plot(winter_profile, z, 'b-', label='Winter Minimum')
    ax2.plot(summer_profile, z, 'r-', label='Summer Maximum')
    
    # Visual aids
    ax2.invert_yaxis()
    ax2.axhline(50, color='k', linestyle='--', alpha=0.5, label='Mixed Layer Base (50m)')
    ax2.set_title('Seasonal Profiles (Year 5)')
    ax2.set_xlabel('Temperature (°C)')
    ax2.set_ylabel('Depth (m)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/experiment1_results.png', dpi=150)
    plt.show()


def experiment2():
    """
    Experiment 2:
    Run year-by-year loops to save memory without modifying the solver.
    Increase dz to 2.0m to allow larger dt for faster execution.
    """

    z_min, z_max = 0.0, 100.0
    dz = 2.0   
    z_grid = np.arange(z_min, z_max + dz, dz)
    Kz = create_diffusivity_profile(z_grid, ml_depth=50.0)
    
    dt = 150.0  
    
    years = 100
    scenarios = [0.0, 2.0, 4.0]
    
    final_profiles = {}
    
    plt.figure(figsize=(8, 10))
    
    for rate in scenarios:
        current_bc_funcs = {
            'upper': make_sst_function(warming_rate_per_century=rate),
            'lower': 4.0
        }
        
        T_current = np.ones_like(z_grid) * 4.0
        for year in range(years):
            if year % 10 == 0:
                print(f"  Simulating Year {year}/{years}...")
            
            t_start = year * 365 * 24 * 3600
            t_end = (year + 1) * 365 * 24 * 3600
            
            def ic_wrapper(z):
                return T_current
            
            _, _, heat_year = solve_heat_equation(
                z_min, z_max, t_start, t_end, dz, dt, Kz,
                ic_wrapper, current_bc_funcs
            )
            
            T_current = heat_year[:, -1]
            
            if year == years - 1:
                final_profiles[rate] = np.mean(heat_year, axis=1)

    control_profile = final_profiles[0.0]
    idx_80m = np.abs(z_grid - 80.0).argmin()
    
    for rate in scenarios:
        prof = final_profiles[rate]
        label = f"Control (0°C/100yr)" if rate == 0 else f"+{rate}°C/100yr"
        plt.plot(prof, z_grid, linewidth=2.5, label=label)
        
        if rate > 0:
            delta_surf = prof[0] - control_profile[0]
            delta_deep = prof[idx_80m] - control_profile[idx_80m]
            print(f"+{rate}°C/100yr    | +{delta_surf:.2f}°C    | +{delta_deep:.2f}°C             | {delta_surf - delta_deep:.2f}°C")

    plt.gca().invert_yaxis()
    plt.axhline(50, color='k', linestyle=':', label='Mixed Layer Base')
    plt.title(f'Deep Ocean Heat Penetration (Year {years})')
    plt.xlabel('Average Temperature (°C)')
    plt.ylabel('Depth (m)')
    plt.legend()
    plt.grid(True, alpha=0.4)
    
    plt.savefig('results/experiment2_warming.png', dpi=150)
    plt.show()
    print("Experiment 2 Complete.")


if __name__ == "__main__":
    validate_solver()
    experiment1()
    experiment2()