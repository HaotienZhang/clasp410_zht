import numpy as np
import matplotlib.pyplot as plt

def heat_equation_neumann(dx=0.02, dt=0.0002, L=1.0, T_final=0.2, c2=1.0):
    """
    This function solves the 1D heat equation using forward difference method with Neumann boundary conditions.
    
    Parameters:
    -----------
    dx : float
        Spatial step size
    dt : float
        Time step size
    L : float
        Length of the wire (1 meter)
    T_final : float
        Final time for simulation
    c2 : float
        Thermal diffusivity coefficient (m²/s)
    
    Returns:
    --------
    U : numpy array
        Temperature distribution over time (shape: [Nt, Nx])
    x : numpy array
        Spatial coordinates
    t : numpy array
        Time coordinates
    """
    
    # Calculate grid parameters
    Nx = int(L / dx) + 1
    Nt = int(T_final / dt) + 1 
    
    # Calculate the stability parameter r
    r = c2 * dt / (dx**2)

    
    # Create spatial and temporal grids
    x = np.linspace(0, L, Nx)
    t = np.linspace(0, T_final, Nt)
    
    # Initialize temperature array
    U = np.zeros((Nt, Nx))
    
    # Set initial condition: U(x, 0) = 4x - 4^2
    U[0, :] = 4 * x - 4 * x**2
    
    # Time stepping using forward difference method
    for j in range(Nt - 1):
        # Update interior points using the forward difference formula
        for i in range(1, Nx - 1):
            U[j+1, i] = (1 - 2*r) * U[j, i] + r * (U[j, i+1] + U[j, i-1])
        
        # Apply Neumann boundary conditions (dU/dx = 0)
        # This means zero gradient at boundaries, so boundary cells equal their neighbors
        
        # Left boundary (i = 0): U[j+1, 0] = U[j+1, 1]
        U[j+1, 0] = U[j+1, 1]
        
        # Right boundary (i = Nx-1): U[j+1, Nx-1] = U[j+1, Nx-2]
        U[j+1, Nx-1] = U[j+1, Nx-2]
    
    return U, x, t


def heat_equation_dirichlet(dx=0.02, dt=0.0002, L=1.0, T_final=0.2, c2=1.0):
    """
    This function solves the 1D heat equation using forward difference method with Dirichlet boundary conditions.
    This function is for comparision. 
    
    Parameters:
    -----------
    dx : float
        Spatial step size
    dt : float
        Time step size
    L : float
        Length of the wire (1 meter)
    T_final : float
        Final time for simulation
    c2 : float
        Thermal diffusivity coefficient (m²/s)
    
    Returns:
    --------
    U : numpy array
        Temperature distribution over time (shape: [Nt, Nx])
    x : numpy array
        Spatial coordinates
    t : numpy array
        Time coordinates
    """
    
    # Calculate grid parameters
    Nx = int(L / dx) + 1
    Nt = int(T_final / dt) + 1
    
    # Calculate the stability parameter r
    r = c2 * dt / (dx**2)
    
    
    # Create spatial and temporal grids
    x = np.linspace(0, L, Nx)
    t = np.linspace(0, T_final, Nt)
    
    # Initialize temperature array
    U = np.zeros((Nt, Nx))
    
    # Set initial condition: U(x, 0) = 4x - 4x^2
    U[0, :] = 4 * x - 4 * x**2
    
    # Time stepping using forward difference method
    for j in range(Nt - 1):
        # Update interior points
        for i in range(1, Nx - 1):
            U[j+1, i] = (1 - 2*r) * U[j, i] + r * (U[j, i+1] + U[j, i-1])
        
        # Apply Dirichlet boundary conditions U = 0
        U[j+1, 0] = 0.0
        U[j+1, Nx-1] = 0.0
    
    return U, x, t


def plot_comparison(dx=0.02, dt=0.0002):
    """
    Create comparison plots between Neumann and Dirichlet boundary conditions.
    """
    
    # Solve with both boundary conditions
    U_neumann, x, t = heat_equation_neumann(dx, dt)
    
    U_dirichlet, _, _ = heat_equation_dirichlet(dx, dt)
    
    # Create figure with just the heat maps
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Heat map for Neumann BC
    im1 = axes[0].imshow(U_neumann.T, aspect='auto', origin='lower',
                          extent=[0, t[-1], 0, x[-1]], cmap='hot')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Position (m)')
    axes[0].set_title('Neumann Insulated Boundaries (dU/dx = 0)')
    plt.colorbar(im1, ax=axes[0], label='Temperature (°C)')
    
    # Plot 2: Heat map for Dirichlet BC
    im2 = axes[1].imshow(U_dirichlet.T, aspect='auto', origin='lower',
                          extent=[0, t[-1], 0, x[-1]], cmap='hot')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Position (m)')
    axes[1].set_title('Dirichlet Fixed Temperature at Boundaries (U = 0°C)')
    plt.colorbar(im2, ax=axes[1], label='Temperature (°C)')
    
    plt.suptitle('Comparison: Neumann vs Dirichlet Boundary Conditions\n1D Heat Equation dx=0.02, dt=0.0002', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("results/neumann_vs_dirichlet_heatmaps.png", dpi=300)
    plt.show()
    
    



if __name__ == "__main__":
    # Run the simulation with the specified parameters
    plot_comparison(dx=0.02, dt=0.0002)
