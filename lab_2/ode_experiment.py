import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import argparse 

# initialize the N1 and N2 with proper subscript for better visulization
N1_TEXT = 'N₁'
N2_TEXT = 'N₂'


def dNdt_comp(t, N, a=1, b=2, c=1, d=3):
    '''
    This function calculates the Lotka-Volterra competition equations for two species.
    
    Parameters
    ----------
    t : float
        The current time (not used here).
    N : two-element list
        The current value of N1 and N2 as a list (e.g., [N1, N2]).
    a, b, c, d : float, defaults=1, 2, 1, 3
        The value of the Lotka-Volterra coefficients.
    
    Returns
    -------
    dN1dt, dN2dt : floats
        The time derivatives of `N1` and `N2`.
    '''

    # Here, N is a two-element list such that N1=N[0] and N2=N[1]
    dN1dt = a * N[0] * (1 - N[0]) - b * N[0] * N[1]
    dN2dt = c * N[1] * (1 - N[1]) - d * N[0] * N[1]
    
    return dN1dt, dN2dt

def dNdt_predprey(t, N, a=1, b=2, c=1, d=3):
    '''
    This function calculates the Lotka-Volterra predator-prey equations.
    
    Parameters
    ----------
    t : float
        The current time (not used here).
    N : two-element list
        The current value of N1 (prey) and N2 (predator) as a list.
    a, b, c, d : float, defaults=1, 2, 1, 3
        The value of the Lotka-Volterra coefficients.
    
    Returns
    -------
    dN1dt, dN2dt : floats
        The time derivatives of `N1` and `N2`.
    '''
    #prey population
    dN1dt = a * N[0] - b * N[0] * N[1]    
    #predator population 
    dN2dt = -c * N[1] + d * N[0] * N[1]    
    
    return dN1dt, dN2dt

def euler_solver(func, N1_init=0.5, N2_init=0.5, dt=0.1, t_final=100.0, **kwargs):
    '''
    Solves a system of two ordinary differential equations using the forward Euler method.

    Parameters
    func : function
        A Python function that calculates the time derivatives. 
    N1_init, N2_init : float, optional
        The initial condition for population N1 and N2. The default for both N1 and N2 are 0.5.
    dt : float, optional
        The time step size. The default is 0.1.
    t_final : float, optional
        The final time to integrate to. The default is 100.0.
    **kwargs : dict
        A dictionary of keyword arguments (a, b, c, d)

    Returns
    -------
    time : numpy.ndarray
        The array of time points.
    N1 : numpy.ndarray
        The array of solutions for population N1.
    N2 : numpy.ndarray
        The array of solutions for population N2.
    '''
    # create the time array
    time = np.arange(0, t_final + dt, dt)
    
    # create the solution arrays
    N1 = np.zeros(time.size)
    N2 = np.zeros(time.size)
    
    # initial condition
    N1[0] = N1_init
    N2[0] = N2_init
    
    for i in range(1, time.size):
        # get the derivatives from the provided function
        dN1, dN2 = func(time[i-1], [N1[i-1], N2[i-1]], **kwargs)
        
        # calculate the next step using the Euler formula
        N1[i] = N1[i-1] + dt * dN1
        N2[i] = N2[i-1] + dt * dN2
        
    return time, N1, N2

def rk8_solver(func, N1_init=0.5, N2_init=0.5, max_step=10, t_final=100.0, a=1, b=2, c=1, d=3):
    '''
    This function will solve the Lotka-Volterra equations using Scipy's 8th order Runge-Kutta solver.

    Parameters
    ----------
    func : function
        A Python function that takes `time`, [`N1`, `N2] as inputs and returns the time derivatives of N1 and N2.
    N1_init, N2_init : float
        Initial conditions for N1 and N2, ranging from (0, 1].
    max_step : float, default=10
        Largest timestep allowed in years.
    t_final : float, default=100
        Integrate until this value is reached, in years.
    a, b, c, d : float, defaults=1, 2, 1, 3
        Lotka-Volterra coefficient values.

    Returns
    -------
    time : numpy.ndarray
        Time elapsed in years.
    N1, N2 : numpy.ndarray
        Normalized population density solutions.
    '''
    # the time span for the integration
    time_span = [0, t_final]
    
    # the initial conditions vector
    initial_conditions = [N1_init, N2_init]
    
    # the arguments (coefficients) for the derivative function
    args = (a, b, c, d)
    
    # call the solver
    result = solve_ivp(
        fun=func, 
        t_span=time_span, 
        y0=initial_conditions,
        args=args, 
        method='DOP853', 
        max_step=max_step
    )
    
    # extract the results
    time = result.t
    N1 = result.y[0, :]
    N2 = result.y[1, :]
    
    return time, N1, N2

def reproduce_figure_1():
    '''
    This function is to verify the implementation by reproducing Figure 1 from the lab manual
    '''
    # for better visualization
    plt.style.use('ggplot')

    # parameters from the lab manual for Figure 1
    a, b, c, d = 1, 2, 1, 3
    N1_init, N2_init = 0.3, 0.6
    t_final = 100.0
    

    # competition model
    dt_comp = 1.0
    time_comp_euler, N1_comp_euler, N2_comp_euler = euler_solver(
        dNdt_comp, N1_init, N2_init, dt=dt_comp, t_final=t_final, a=a, b=b, c=c, d=d
    )
    time_comp_rk8, N1_comp_rk8, N2_comp_rk8 = rk8_solver(
        dNdt_comp, N1_init, N2_init, t_final=t_final, a=a, b=b, c=c, d=d
    )

    # predator-prey model
    dt_pred = 0.05
    time_pred_euler, N1_pred_euler, N2_pred_euler = euler_solver(
        dNdt_predprey, N1_init, N2_init, dt=dt_pred, t_final=t_final, a=a, b=b, c=c, d=d
    )
    time_pred_rk8, N1_pred_rk8, N2_pred_rk8 = rk8_solver(
        dNdt_predprey, N1_init, N2_init, t_final=t_final, a=a, b=b, c=c, d=d
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # plot 1: competition
    ax1.plot(time_comp_euler, N1_comp_euler, 'C0-', label=f'{N1_TEXT} Euler')
    ax1.plot(time_comp_euler, N2_comp_euler, 'C1-', label=f'{N2_TEXT} Euler')
    ax1.plot(time_comp_rk8, N1_comp_rk8, 'C0:', label=f'{N1_TEXT} RK8')
    ax1.plot(time_comp_rk8, N2_comp_rk8, 'C1:', label=f'{N2_TEXT} RK8')
    ax1.set_title('Lotka-Volterra Competition Model')
    ax1.set_xlabel('Time (years)')
    ax1.set_ylabel('Population/Carrying Cap.')
    ax1.legend()

    # plot 2: predator-prey
    ax2.plot(time_pred_euler, N1_pred_euler, 'C0-', label=f'{N1_TEXT} (Prey) Euler')
    ax2.plot(time_pred_euler, N2_pred_euler, 'C1-', label=f'{N2_TEXT} (Predator) Euler')
    ax2.plot(time_pred_rk8, N1_pred_rk8, 'C0:', label=f'{N1_TEXT} (Prey) RK8')
    ax2.plot(time_pred_rk8, N2_pred_rk8, 'C1:', label=f'{N2_TEXT} (Predator) RK8')
    ax2.set_title('Lotka-Volterra Predator-Prey Model')
    ax2.set_xlabel('Time (years)')
    ax2.set_ylabel('Population/Carrying Cap.')
    ax2.legend()
    
    fig.text(0.5, 0.01, f'Coefficients: a={a}, b={b}, c={c}, d={d}', ha='center')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig('results/model_validation_plot.png', dpi=300)
    plt.show()

def run_experiment_1():
    '''
    Performs the experiment for SQ 1, 
    comparing the Euler solver's performance at different time steps against the RK8 solver.
    '''
    plt.style.use('ggplot')

    # shared parameters
    params = {'a': 1, 'b': 2, 'c': 1, 'd': 3}
    initial_conditions = {'N1_init': 0.3, 'N2_init': 0.6}
    t_final = 100.0

    # competition model in sq1
    dt_values_comp = [2.0, 1.0, 0.1]
    fig_comp, axes_comp = plt.subplots(len(dt_values_comp), 1, figsize=(8, 10), sharex=True)
    fig_comp.suptitle('Euler Solver Performance with different Time Steps (Competition Model)', fontsize=16)

    # get the baseline solution from RK8 for comparision
    time_rk8_comp, n1_rk8_comp, n2_rk8_comp = rk8_solver(dNdt_comp, **initial_conditions, **params, t_final=t_final)

    # using different subplots for better visualization
    for i, dt in enumerate(dt_values_comp):
        ax = axes_comp[i]
        time_e, n1_e, n2_e = euler_solver(dNdt_comp, **initial_conditions, dt=dt, t_final=t_final, **params)
        
        # plot the RK 8 solver
        ax.plot(time_rk8_comp, n1_rk8_comp, 'k:', label=f'{N1_TEXT} RK8 (Baseline)')
        ax.plot(time_rk8_comp, n2_rk8_comp, 'k--', label=f'{N2_TEXT} RK8 (Baseline)')
        
        # Plot Euler results
        ax.plot(time_e, n1_e, '-', label=f'{N1_TEXT} Euler')
        ax.plot(time_e, n2_e, '-', label=f'{N2_TEXT} Euler')
        
        ax.set_title(f'dt = {dt} years')
        ax.set_ylabel('Population/Cap.')
        ax.legend()

    axes_comp[-1].set_xlabel('Time (years)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('results/experiment1_competition_model.png', dpi=300)
    plt.show()

    # predator-prey model in sq1
    dt_values_pred = [0.08, 0.05, 0.01]
    fig_pred, axes_pred = plt.subplots(len(dt_values_pred), 1, figsize=(8, 10), sharex=True)
    fig_pred.suptitle('Euler Solver Performance with different Time Step (Predator-Prey)', fontsize=16)
    
    # get the baseline solution from RK8
    time_rk8_pred, n1_rk8_pred, n2_rk8_pred = rk8_solver(dNdt_predprey, **initial_conditions, **params, t_final=t_final)

    for i, dt in enumerate(dt_values_pred):
        ax = axes_pred[i]
        time_e, n1_e, n2_e = euler_solver(dNdt_predprey, **initial_conditions, dt=dt, t_final=t_final, **params)
        
        # plot rk8 baseline
        ax.plot(time_rk8_pred, n1_rk8_pred, 'k:', label=f'{N1_TEXT} RK8 (baseline)')
        ax.plot(time_rk8_pred, n2_rk8_pred, 'k--', label=f'{N2_TEXT} RK8 (baseline)')

        # plot euler method
        ax.plot(time_e, n1_e, '-', label=f'{N1_TEXT} Euler')
        ax.plot(time_e, n2_e, '-', label=f'{N2_TEXT} Euler')
        
        ax.set_title(f'Dt = {dt} years')
        ax.set_ylabel('Population/Cap.')
        ax.legend()

    axes_pred[-1].set_xlabel('Time (years)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('results/experiment1_predator_prey_model.png', dpi=300)
    plt.show()

def run_experiment_2():
    '''
    Performs the experiment for SQ 2, 
    investigating how coefficients and initial conditions affect the outcome of the competition model.
    '''
    plt.style.use('ggplot')

    # different Scenarios, with specific parameters setting
    scenarios = [
        {
            'title': 'Scenario A: Stable Coexistence',
            'description': 'Intraspecific comp. > Interspecific comp. (a>b, c>d)',
            'params': {'a': 1, 'b': 0.5, 'c': 1, 'd': 0.5},
            'initial_conditions': {'N1_init': 0.2, 'N2_init': 0.8} 
        },
        {
            'title': 'Scenario B: Competitive Exclusion (Species 1 Dominates)',
            'description': 'Species 1 is the superior competitor (a>b, d>c)',
            'params': {'a': 2, 'b': 1, 'c': 1, 'd': 2},
            'initial_conditions': {'N1_init': 0.5, 'N2_init': 0.5}
        },
        {
            'title': 'Scenario C: Competitive Exclusion (Species 2 Dominates)',
            'description': 'Species 2 is the superior competitor (c>d, b>a)',
            'params': {'a': 1, 'b': 2, 'c': 2, 'd': 1},
            'initial_conditions': {'N1_init': 0.5, 'N2_init': 0.5}
        },
        {
            'title': 'Scenario D: Unstable Equilibrium (Winner Depends on Start)',
            'description': 'Interspecific comp. > Intraspecific comp. (b>a, d>c)',
            'params': {'a': 1, 'b': 2, 'c': 1, 'd': 2},
            'initial_conditions': {'N1_init': 0.51, 'N2_init': 0.49} # N1 starts slightly higher
        }
    ]

    fig, axes = plt.subplots(len(scenarios), 1, figsize=(8, 12), sharex=True)
    fig.suptitle('Parameter Study For Competition Model', fontsize=16)

    # create plot for different scenerios
    for i, scenario in enumerate(scenarios):
        ax = axes[i]
        time, n1, n2 = rk8_solver(dNdt_comp, **scenario['initial_conditions'], **scenario['params'], t_final=100.0)
        
        ax.plot(time, n1, '-', label=f'{N1_TEXT}')
        ax.plot(time, n2, '-', label=f'{N2_TEXT}')
        
        param_str = f"a={scenario['params']['a']}, b={scenario['params']['b']}, c={scenario['params']['c']}, d={scenario['params']['d']}"
        init_str = f"N1_init={scenario['initial_conditions']['N1_init']}, N2_init={scenario['initial_conditions']['N2_init']}"
        ax.set_title(f"{scenario['title']}\n({param_str} | {init_str})", fontsize=10)
        
        ax.set_ylabel('Population/Cap.')
        ax.legend()
        ax.set_ylim(0, 1.1)

    axes[-1].set_xlabel('Time (years)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('results/experiment2_competition_study.png', dpi=300)
    plt.show()

def run_experiment_3():
    '''
    Performs the experiment for SQ 3, exploring the predator-prey model dynamics.
    '''
    plt.style.use('ggplot')

    scenarios = [
        {
            'title': 'Scenario A: Baseline Stable Cycle',
            'params': {'a': 1, 'b': 1, 'c': 1, 'd': 1},
            'initial_conditions': {'N1_init': 0.5, 'N2_init': 0.5}
        },
        {
            'title': 'Scenario B: Same Cycle, Different Start',
            'params': {'a': 1, 'b': 1, 'c': 1, 'd': 1},
            'initial_conditions': {'N1_init': 0.2, 'N2_init': 0.8}
        },
        {
            'title': 'Scenario C: High Predation, Large Swings',
            'params': {'a': 1, 'b': 2, 'c': 1, 'd': 3},
            'initial_conditions': {'N1_init': 0.5, 'N2_init': 0.5}
        },
        {
            'title': 'Scenario D: Shifting the Equilibrium Point',
            'params': {'a': 2, 'b': 1, 'c': 3, 'd': 1},
            'initial_conditions': {'N1_init': 0.9, 'N2_init': 0.1}
        }
    ]
    
    # using sub plots
    fig, axes = plt.subplots(len(scenarios), 2, figsize=(12, 14))
    fig.suptitle('Parameter Study for Predator-Prey Model', fontsize=16)

    for i, scenario in enumerate(scenarios):
        ax_ts = axes[i, 0]  # time series Axe
        ax_ph = axes[i, 1]  # phase diagram Axe
        
        time, n1, n2 = rk8_solver(dNdt_predprey, **scenario['initial_conditions'], **scenario['params'], t_final=100.0)
        
        # left: time series plot
        params_str = f"(a={scenario['params']['a']}, b={scenario['params']['b']}, c={scenario['params']['c']}, d={scenario['params']['d']})"
        ax_ts.set_title(f"{scenario['title']}\n{params_str}", fontsize=11)
        ax_ts.plot(time, n1, label=f'{N1_TEXT} (Prey)')
        ax_ts.plot(time, n2, label=f'{N2_TEXT} (Predator)')
        ax_ts.set_ylabel('Population')
        ax_ts.legend()
        
        # right: phase diagram
        ax_ph.set_title("Phase Diagram", fontsize=11)
        ax_ph.plot(n1, n2, '-')
        ax_ph.plot(n1[0], n2[0], 'go', markersize=8, label='Start')
        ax_ph.plot(n1[-1], n2[-1], 'rs', markersize=8, label='End')
        ax_ph.set_ylabel(f'{N2_TEXT} (Predator)')
        ax_ph.legend()

    # set bottom labels only for the last row, for a better visualization
    axes[-1, 0].set_xlabel('Time (years)')
    axes[-1, 1].set_xlabel(f'{N1_TEXT} (Prey)')
    
    # adjust layout to prevent title overlap
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('results/experiment3_predator_prey_study.png', dpi=300)
    plt.show()

def main():

    # add arguments, to decide which experiment to run
    parser = argparse.ArgumentParser()

    # will perform the experiments that are set true 
    # validation, verifying the implementation by reproducing figure 1
    # exp1,2,3 , run experiment that designed for corresponding science question
    parser.add_argument("--validation", action='store_true')
    parser.add_argument("--exp1", action='store_true')
    parser.add_argument("--exp2", action='store_true')
    parser.add_argument("--exp3", action='store_true')
    args = parser.parse_args()
    
    print(args)
    if args.validation:
        reproduce_figure_1()
    if args.exp1:
        run_experiment_1()
    if args.exp2:
        run_experiment_2()
    if args.exp3: 
        run_experiment_3()
    
if __name__ == "__main__":
    main()