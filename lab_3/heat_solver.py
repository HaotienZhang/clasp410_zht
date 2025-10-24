import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import argparse 

plt.style.use('fivethirtyeight')

def solve_heat_equation(x_min, x_max, t_min, t_max, dx, dt, c_squared, 
                        initial_condition, boundary_conditions):
    """
    Solve the 1D heat equation using forward-difference method.
    
    Parameters:
    -----------
    x_min : float
        Minimum spatial coordinate (e.g., 0 for surface)
    x_max : float
        Maximum spatial coordinate (e.g., 100 for depth in meters)
    t_min : float
        Starting time
    t_max : float
        Ending time
    dx : float
        Spatial step size
    dt : float
        Time step size
    c_squared : float
        Thermal diffusivity (m^2/s or mm^2/s)
    initial_condition : function or array
        Function U(x) or array of initial temperatures
    boundary_conditions : dict
        Dictionary with 'lower' and 'upper' boundary conditions
        Each can be a constant or a function of time
        Example: {'lower': 5.0, 'upper': lambda t: 10*np.sin(t)}
    
    Returns:
    --------
    time : ndarray
        Array of time points
    x : ndarray
        Array of spatial grid points
    heat : ndarray
        2D array of temperatures (space x time)
    """
    
    # Calculate r parameter for stability
    r = c_squared * dt / (dx**2)
    
    # Check stability criterion: r <= 0.5
    if r > 0.5:
        raise ValueError(f"Solution is unstable! r = {r:.4f} > 0.5\n")
    
    # Create spatial and temporal grids
    x = np.arange(x_min, x_max + dx, dx)
    time = np.arange(t_min, t_max + dt, dt)
    
    n_space = len(x)
    n_time = len(time)
    
    # Initialize solution array
    heat = np.zeros((n_space, n_time))
    
    # Set initial condition
    if callable(initial_condition):
        heat[:, 0] = initial_condition(x)
    else:
        heat[:, 0] = initial_condition
    
    # Time stepping loop
    for j in range(n_time - 1):
        # Update interior points using forward-difference formula
        for i in range(1, n_space - 1):
            heat[i, j+1] = (1 - 2*r) * heat[i, j] + r * (heat[i-1, j] + heat[i+1, j])
        
        # Apply boundary conditions
        # Lower boundary, i = 0, surface
        bc_lower = boundary_conditions['lower']
        if callable(bc_lower):
            heat[0, j+1] = bc_lower(time[j+1])
        else:
            heat[0, j+1] = bc_lower
        
        # Upper boundary, i=n_space-1, bottom
        bc_upper = boundary_conditions['upper']
        if callable(bc_upper):
            heat[-1, j+1] = bc_upper(time[j+1])
        else:
            heat[-1, j+1] = bc_upper
    
    return time, x, heat


# Example: Validate with the test problem from the lab
def validate_solver():
    """
    Validate the solver with the example problem from the lab manual.
    """
    print("VALIDATING SOLVER WITH TEST PROBLEM\n")
    
    # Problem parameters
    x_min, x_max = 0.0, 1.0
    t_min, t_max = 0.0, 0.2
    dx = 0.2
    dt = 0.02
    c_squared = 1.0  # m^2/s
    
    # Initial condition: U(x,0) = 4x - 4x^2
    def initial_cond(x):
        return 4*x - 4*x**2
    
    # Boundary conditions: U(0,t) = U(1,t) = 0
    boundary_conds = {
        'lower': 0.0,
        'upper': 0.0
    }
    
    # Solve
    time, x, heat = solve_heat_equation(
        x_min, x_max, t_min, t_max, dx, dt, c_squared,
        initial_cond, boundary_conds
    )
    
    # Display results
    print("\nSolution:\n")
    
    # also check the solution with groundtruth value
    sol10p3 = [[0.000000, 0.640000, 0.960000, 0.960000, 0.640000, 0.000000],
               [0.000000, 0.480000, 0.800000, 0.800000, 0.480000, 0.000000],
               [0.000000, 0.400000, 0.640000, 0.640000, 0.400000, 0.000000],
               [0.000000, 0.320000, 0.520000, 0.520000, 0.320000, 0.000000],
               [0.000000, 0.260000, 0.420000, 0.420000, 0.260000, 0.000000],
               [0.000000, 0.210000, 0.340000, 0.340000, 0.210000, 0.000000],
               [0.000000, 0.170000, 0.275000, 0.275000, 0.170000, 0.000000],
               [0.000000, 0.137500, 0.222500, 0.222500, 0.137500, 0.000000],
               [0.000000, 0.111250, 0.180000, 0.180000, 0.111250, 0.000000],
               [0.000000, 0.090000, 0.145625, 0.145625, 0.090000, 0.000000],
               [0.000000, 0.072812, 0.117813, 0.117813, 0.072812, 0.000000]]
    
    # Convert to array and transpose to get correct ordering (space x time)
    sol10p3 = np.array(sol10p3).transpose()
    
    for i in range(len(x)):
        print(f"{i:3}", end="")
        for j in range(min(11, len(time))):
            print(f"{heat[i,j]:10.6f}", end="")
            if heat[i,j] != sol10p3[i,j]:
                assert np.isclose(heat[i,j], sol10p3[i,j]), \
                    f"Value mismatch at i={i}, j={j}: {heat[i,j]} != {sol10p3[i,j]}"
        print()
    

    return time, x, heat

t_kanger = np.array([-19.7, -21.0, -17., -8.4, 2.3, 8.4,
                     10.7, 8.5, 3.1, -6.0, -12.0, -16.9])

def temp_kanger(t):
    '''
    For an array of times in days, return timeseries of temperature for
    Kangerlussuaq, Greenland.
    
    Parameters:
    -----------
    t : float or array
        Time in days
    
    Returns:
    --------
    Temperature in °C at time t
    '''
    t_amp = (t_kanger - t_kanger.mean()).max()
    return t_amp * np.sin(np.pi/180 * t - np.pi/2) + t_kanger.mean()


def run_exp_1():
    """
    Experiment 1: Investigate permafrost in Kangerlussuaq, Greenland.
    Creates two plots as specified in the lab manual.
    """
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    print("EXPERIMENT 1: PERMAFROST IN KANGERLUSSUAQ, GREENLAND")
    
    # define model parameters
    x_min, x_max = 0.0, 100.0
    dx = 0.5
    c_2 = 0.25e-6  # thermal diffusivity (m²/s)
    
    # define temporal domain
    dt_max = dx**2 / (2 * c_2)
    dt = 0.5 * dt_max
    t_min = 0.0
    t_max = 365 * 100 * 24 * 3600  # simulation period (s)
    
    # initial condition: start at 0°C
    def initial_condition(x):
        return np.zeros_like(x)
    
    # define boundary conditions
    def upper_bc(t):
        t_days = t / (24 * 3600)
        return temp_kanger(t_days)
    
    boundary_conditions = {
        'lower': upper_bc,  # surface (index 0)
        'upper': 5.0        # deep boundary (fixed geothermal)
    }
    
    
    # solve the heat equation
    time, x, heat = solve_heat_equation(
        x_min, x_max, t_min, t_max, dx, dt, c_2,
        initial_condition, boundary_conditions
    )
    
    # Convert time to years for visualization
    time_years = time / (365 * 24 * 3600)
    
    
    # generate temperature plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # plot temperature field (time-depth heat map)
    im1 = ax1.pcolor(time_years, x, heat, cmap='seismic', vmin=-25, vmax=25, shading='auto')
    ax1.invert_yaxis()  # Invert y-axis so surface is at top
    ax1.set_xlabel('Time (Years)', fontsize=11)
    ax1.set_ylabel('Depth (m)', fontsize=11)
    ax1.set_title('Ground Temperature: Kangerlussuaq, Greenland', fontsize=12, fontweight='bold')
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, label='Temperature (°C)')
    cbar1.ax.tick_params(labelsize=10)
    
    # plot seasonal extremes (winter vs summer)
    last_year_start = int(-365 * 24 * 3600 / dt)  # Index for start of last year
    
    # Get winter (minimum) and summer (maximum) temperatures over the last year
    winter = heat[:, last_year_start:].min(axis=1)
    summer = heat[:, last_year_start:].max(axis=1)
    
    ax2.plot(winter, x, 'b-', linewidth=2.5, label='Winter')
    ax2.plot(summer, x, 'r--', linewidth=2.5, label='Summer')
    ax2.axvline(x=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.7)
    ax2.invert_yaxis()
    ax2.set_xlabel('Temperature ($^\circ$C)', fontsize=12)
    ax2.set_ylabel('Depth (m)', fontsize=12)
    ax2.set_title('Ground Temperature: Kangerlussuaq', fontsize=13)
    ax2.legend(loc='lower left', fontsize=11)
    ax2.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    ax2.set_xlim(-8, 6)
    
    plt.savefig('results/exp_1.png', dpi=300)
    plt.tight_layout()
    plt.show()

def run_exp_2():
    """
    Experiment 2: Investigate global warming effects on permafrost.
    Analyzes how temperature shifts of 0.5°C, 1°C, and 3°C affect
    the active and permafrost layer depths.
    """
    # define model parameters (same as experiment 1)
    x_min, x_max = 0.0, 100.0
    dx = 0.5
    c_2 = 0.25e-6
    
    # define temporal domain (long-term evolution)
    dt_max = dx**2 / (2 * c_2)
    dt = 0.5 * dt_max
    t_min = 0.0
    t_max = 365 * 250 * 24 * 3600  # simulate 250 years
    
    # define warming scenarios
    temp_shifts = [0.0, 0.5, 1.0, 3.0]

    # store the experiment results
    results = {}
    
    # Initial condition: start at 0°C throughout
    def initial_condition(x):
        return np.zeros_like(x)
    plt.figure(figsize=(10, 12))
    for shift in temp_shifts:
        # Modified boundary condition with temperature shift
        def upper_bc(t, temp_shift=shift):
            t_days = t / (24 * 3600)
            return temp_kanger(t_days) + temp_shift
        
        boundary_conditions = {
            'lower': upper_bc,  # surface boundary
            'upper': 5.0        # geothermal at depth
        }
        
        # solve the heat equation
        time, x, heat = solve_heat_equation(
            x_min, x_max, t_min, t_max, dx, dt, c_2,
            initial_condition, boundary_conditions
        )
        
        # extract last year for equilibrium analysis
        last_year_start = int(-365 * 24 * 3600 / dt)
        last_year_data = heat[:, last_year_start:]
        
        # compute seasonal temperature profiles
        winter_profile = last_year_data.min(axis=1)
        summer_profile = last_year_data.max(axis=1)
        
        # determine active layer depth (max depth where T > 0°C)
        active_layer_mask = summer_profile > 0
        active_layer_depth = x[active_layer_mask].max() if active_layer_mask.any() else 0.0
        
        
        # Find permafrost extent (where winter temps < 0°C)
        permafrost_mask = winter_profile < 0
        if permafrost_mask.any():
            permafrost_indices = np.where(permafrost_mask)[0]
            permafrost_top = x[permafrost_indices[0]]
            permafrost_bottom = x[permafrost_indices[-1]]
            permafrost_thickness = permafrost_bottom - permafrost_top
        else:
            permafrost_top = None
            permafrost_bottom = None
            permafrost_thickness = 0.0
        
        # Store results
        results[shift] = {
            'active_layer_depth': active_layer_depth,
            'permafrost_top': permafrost_top,
            'permafrost_bottom': permafrost_bottom,
            'permafrost_thickness': permafrost_thickness
        }
        plt.plot(summer_profile, x, label=f'+{shift}°C')

    
    plt.gca().invert_yaxis()
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Depth (m)')
    plt.title('Summer Temperature Profiles under Warming Scenarios')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/exp_2.png', dpi=300)
    plt.show()
    
    # print summary table
    print("Temperature Shift | Active Layer Depth | Permafrost Thickness")
    print("-" * 60)
    for shift in temp_shifts:
        r = results[shift]
        print(f"+{shift:4.1f}°C          | {r['active_layer_depth']:6.1f} m          | ", end="")
        if r['permafrost_thickness'] > 0:
            print(f"{r['permafrost_thickness']:6.1f} m")
        else:
            print("None")
    
    return results

# Run validation
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--validation", action='store_true')
    parser.add_argument("--exp1", action='store_true')
    parser.add_argument("--exp2", action='store_true')
    args = parser.parse_args()

    if args.validation:
        validate_solver()
    if args.exp1:
        run_exp_1()
    if args.exp2:
        run_exp_2()
