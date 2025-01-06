import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.stats import linregress

from initial_distr import init, exact
from time_evolution import *


def plot_final(f_ex, x_ex, f, x, params, save=False):
    # Plot the exact and numerical solutions at the final time of the simulations
    plt.plot(x_ex, f_ex,  lw=1, c='r')
    plt.plot(x, f, lw=1, c='k', ls='--')

    # Add 
    plt.xlim(params.xmin, params.xmax)
    plt.xlabel('x')
    plt.ylabel(r'$\overline{f}$', rotation=90)

    # Save the plot as an image file
    if save:
        plt.savefig(f'{params.stencil}_{params.evotype}_{params.func.name}_T{params.t:.2f}_C{params.C_max}_G{params.grid_size}_comp.png')
    plt.show()


def plot_interp(f_ex, x_ex, f_interp, x, params, num_steps, save=False):
    # Plot the exact solution and the numerical interpolation at the specified time
    plt.plot(x_ex, f_ex,  lw=1, c='r', label='exact volume averaged values')
    plt.scatter(x, f_interp, lw=0.5, c='k', marker='x', s=15, label='interpolated face averaged values')

    # Add 
    plt.xlim(params.xmin, params.xmax)
    plt.xlabel('x')
    plt.ylabel(r'$\left< f \right>$', rotation=90)
    plt.legend()

    # Print number of timesteps taken
    print(f'The number of timesteps since start of simulation: {num_steps}')
    # Save the plot as an image file
    if save:
        plt.savefig(f'{params.stencil}_{params.evotype}_{params.func.name}_T{num_steps:.2f}_C{params.C_max}_G{params.grid_size}_interp.png')
    plt.show()


def plot_multip_interp(f_ex, x_ex, f_list, x, params, num_steps, save=False):
    plt.figure(figsize=(8, 6)) # Width, Height

    # Plot the exact solution and the numerical interpolation at the specified time
    plt.plot(x_ex, f_ex,  lw=1, c='b')
    col = ['orange', 'green', 'magenta']
    form = ['o', '<', 'x']
    for i in range(params.mult_len):
        plt.scatter(x, f_list[:, i], lw=2, c=col[i], marker=form[i], s=30, label=f'{params.mult_stencil[i]}')

    # Add 
    plt.xlim(params.xmin, params.xmax)
    plt.ylim(-0.5, 1.4)
    plt.xlabel('x')
    plt.ylabel(r'$\left< f \right>$', rotation=90)
    plt.grid(True)
    plt.legend()

    # Save the plot as an image file
    if save:
        plt.savefig(f'{params.stencil}_{params.evotype}_{params.func.name}_T{num_steps:.2f}_C{params.C_max}_G{params.grid_size}_interp.png')
    plt.show()


def plot_norm_err(L1, L2, L_inf, res, params, save=False):
    #determine the linear regression for the norms
    log_L1 = np.log(L1)
    log_L2 = np.log(L2)
    log_L_inf = np.log(L_inf)
    log_x = np.log(res)
    slope_L2, intercept_L2, r_value, p_value, std_err = linregress(log_x, log_L2)
    slope_L1, intercept_L1, r_value, p_value, std_err = linregress(log_x, log_L1)
    slope_L_inf, intercept_L_inf, r_value, p_value, std_err = linregress(log_x, log_L_inf)

    #plot the resulting values of the norms w.r.t. the gridsize
    plt.loglog(res, L1, label='Numerical datapoints of L1')
    plt.loglog(res, np.exp(intercept_L1) * (res)** slope_L1, ls='--', c='k', label=f'Fit: y = {np.exp(intercept_L1):.2f} * x^{slope_L1:.2f}')
    plt.xlabel('number of gridpoints')
    plt.ylabel('relative L1-norm')
    plt.legend()
    plt.grid()

    # Save the plot as an image file
    if save:
        plt.savefig(f"{params.stencil}_{params.evotype}_{params.func.name}_T{params.t:.2f}_C{params.C_max}_L1.png", dpi=600, bbox_inches='tight')  

    plt.show()

    #plot the resulting values of L2 w.r.t. the gridsize
    plt.loglog(res, L2, label='Numerical datapoints of L2')
    plt.loglog(res, np.exp(intercept_L2) * (res)** slope_L2, ls='--', c='k', label=f'Fit: y = {np.exp(intercept_L2):.2f} * x^{slope_L2:.2f}')
    plt.xlabel('number of gridpoints')
    plt.ylabel('relative L2-norm')
    plt.legend()
    plt.grid()

    # Save the plot as an image file
    if save:
        plt.savefig(f"{params.stencil}_{params.evotype}_{params.func.name}_T{params.t:.2f}_C{params.C_max}_L2.png", dpi=600, bbox_inches='tight')

    plt.show()

    #plot the resulting values of L-infty w.r.t. the gridsize
    plt.loglog(res, L_inf, label='Numerical datapoints of L-$\\infty$')
    plt.loglog(res, np.exp(intercept_L_inf) * (res)** slope_L_inf, ls='--', c='k', label=f'Fit: y = {np.exp(intercept_L_inf):.2f} * x^{slope_L_inf:.2f}')
    plt.xlabel('number of gridpoints')
    plt.ylabel('relative L-$\\infty$-norm')
    plt.legend()
    plt.grid()

    # Save the plot as an image file
    if save:
        plt.savefig(f"{params.stencil}_{params.evotype}_{params.func.name}_T{params.t:.2f}_C{params.C_max}_L_inf.png", dpi=600, bbox_inches='tight')

    plt.show()  


def plot_ani_comp(params, exact_grid_size, num_steps, save=False):
    # Set up the plot for animation
    fig, ax = plt.subplots()  # Correctly creating the figure and axes

    # marking the x-axis and y-axis 
    ax.set_xlim(params.xmin, params.xmax)
    ax.set_ylim(-0.5, 1.5) 

    line_exact, = ax.plot([], [], label='Exact', color='red')
    line_numerical, = ax.plot([], [], label='Numerical', color='blue')
    ax.legend()

    # Add text object to show time
    time_text = ax.text(0.05, 0.95, f'Time: {params.t:.2f}', transform=ax.transAxes, fontsize=12, verticalalignment='top')

    # Initialization function: this will be called at the beginning
    def init_func():
        line_exact.set_data([], [])
        line_numerical.set_data([], [])
        time_text.set_text(f'Time: {params.t:.2f}')
        return line_exact, line_numerical, time_text

    # Animation function: this will be called for each frame
    def animate(i):
        # Initialize numerical and exact solutions
        f, x = init(params)  # Evolve the numerical solution
        params.duration = -params.t
        f_ex, x_ex = exact(params, grid_size=exact_grid_size)
        params.t += params.dt
        


        for _ in range(i):
            f = time_evo(f, params)  # Evolve the numerical solution
        

        # Update both the exact and numerical solutions in the plot
        line_exact.set_data(x_ex, f_ex)  # Exact solution
        line_numerical.set_data(x, f)    # Numerical solution

        # Update the time text on the plot
        time_text.set_text(f'Time: {params.t:.2f}')

        return line_exact, line_numerical, time_text
        
    # Create the animation
    anim = FuncAnimation(fig, animate, frames=num_steps + 1, init_func=init_func, blit=True, interval=25)

    # If 'save' is True, save the animation as a file
    if save:
        anim.save(f'{params.stencil}_{params.evotype}_{params.func.name}_T{params.duration:.2f}_C{params.C_max}_G{params.grid_size}_animated.mp4', writer='ffmpeg', fps=30)
    plt.show()