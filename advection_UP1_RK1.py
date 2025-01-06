import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.stats import linregress
    
def init(grid_size):
    #create the initial distribution and other required quantities
    dx = (xmax-xmin)/(grid_size)
    x = np.arange(grid_size) * dx + xmin + dx / 2 #required for plotting function later on
    f = np.zeros(grid_size) #represents the volume integrated values

    #block function
    f[grid_size //5:grid_size //5 * 2] = 1 

    #sine wave
    #f = np.sin(x * 2 * np.pi)

    t= 0
    return f, t, x, dx


def time_step(dx, C_max):
    #determines the maximal value of dt based on the CFL criterion
    return C_max * dx / v 


def advance_UP1_RK1(f, dx, dt):
    #determine the sum of the fluxes through the surfaces of the control volume
    f -= v * dt / dx * (f - np.roll(f, 1))
    return f


def run_UP1_RK1(grid_size):
    #determine initial conditions of system
    f_init, t, x, dx = init(grid_size)
    f = f_init.copy()

    #determine the timestep using CFL criterion
    dt = time_step(dx, C_max)

    #evolve the distribution in time while updating the time
    while t < duration:
        f = advance_UP1_RK1(f, dx, dt)
        t +=dt
    return f_init, f, x


def run_exact(grid_size):
    #determine the exact analytical solution
    dx = (xmax-xmin)/(grid_size)
    x = np.arange(grid_size) * dx + xmin + dx / 2

    #block function
    f = np.array([1 if ((x[i] + v * duration + 1) % 2 - 1 < -0.2) & ((x[i] + v * duration + 1) % 2 - 1 > -0.6) else 0 for i in range(grid_size)])
    
    #sine wave
    #f = np.sin((x- duration * v) * 2 * np.pi)
    
    return f, x


def calc_L1(f1, f2=None):
    #determine the L1-norm
    if f2 is None:
        L1 = norm(f1, 1)
    else:
        L1 = norm(f1-f2, 1)
    return L1


def calc_L2(f1, f2=None):
    #determine the L2-norm
    if f2 is None:
        L2 = norm(f1, 2)
    else:
        L2 = norm(f1-f2, 2)
    return L2


def calc_L_inf(f1, f2=None):
    #determine the L_infty-norm
    if f2 is None:
        L_inf = norm(f1, np.inf)
    else:
        L_inf = norm(f1-f2, np.inf)
    return L_inf


def calc_norm_evo(reps):
    #determine the relative norms of the system for different resolutions
    L1 = np.empty(reps)
    L2 = np.empty(reps)
    L_inf = np.empty(reps)

    res = grid_min * 2 ** np.arange(reps)
    for exp in range(reps):
        print(exp)
        #determine gridsize
        grid_size = grid_min * 2**exp

        #calculate the numerical and exact solutions
        _, f_num, _ = run_UP1_RK1(grid_size)
        f_ex, _ = run_exact(grid_size)

        #determine the relative value of L1
        L1_ex = calc_L1(f_ex)
        if L1_ex == 0:
            L1[exp] = 1
        else:
            L1[exp] = np.abs(calc_L1(f_num, f_ex)) / L1_ex

        #determine the relative value of L2
        L2_ex = calc_L2(f_ex)
        if L2_ex == 0:
            L2[exp] = 1
        else:
            L2[exp] = np.abs(calc_L2(f_num, f_ex)) / L2_ex

        #determine the relative value of L2
        L_inf_ex = calc_L_inf(f_ex)
        if L_inf_ex == 0:
            L_inf[exp] = 1
        else:
            L_inf[exp] = np.abs(calc_L_inf(f_num, f_ex)) / L_inf_ex

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
    plt.loglog(res, np.exp(intercept_L1) * (res)** slope_L1, 'r-', label=f'Fit: y = {np.exp(intercept_L1):.2f} * x^{slope_L1:.2f}')
    plt.title('Relative L1-norm using UP1 RK1 with respect to the gridsize')
    plt.xlabel('number of gridpoints')
    plt.ylabel('relative L1-norm')
    plt.legend()
    plt.grid()

    # Save the plot as an image file
    plt.savefig("fitted_L1_block_UP1_RK1.png", dpi=300, bbox_inches='tight')  # Saves as PNG with 300 DPI

    plt.show()

    #plot the resulting values of L2 w.r.t. the gridsize
    plt.loglog(res, L2, label='Numerical datapoints of L2')
    plt.loglog(res, np.exp(intercept_L2) * (res)** slope_L2, 'r-', label=f'Fit: y = {np.exp(intercept_L2):.2f} * x^{slope_L2:.2f}')
    plt.title('Relative L2-norm using UP1 RK1 with respect to the gridsize')
    plt.xlabel('number of gridpoints')
    plt.ylabel('relative L2-norm')
    plt.legend()
    plt.grid()

    # Save the plot as an image file
    plt.savefig("fitted_L2_block_UP1_RK1.png", dpi=300, bbox_inches='tight')  # Saves as PNG with 300 DPI

    plt.show()

    #plot the resulting values of L-infty w.r.t. the gridsize
    plt.loglog(res, L_inf, label='Numerical datapoints of L-$\\infty$')
    plt.loglog(res, np.exp(intercept_L_inf) * (res)** slope_L_inf, 'r-', label=f'Fit: y = {np.exp(intercept_L_inf):.2f} * x^{slope_L_inf:.2f}')
    plt.title('Relative L-$\\infty$-norm using UP1 RK1 with respect to the gridsize')
    plt.xlabel('number of gridpoints')
    plt.ylabel('relative L-$\\infty$-norm')
    plt.legend()
    plt.grid()

    # Save the plot as an image file
    plt.savefig("fitted_L_infty_block_UP1_RK1.png", dpi=300, bbox_inches='tight')  # Saves as PNG with 300 DPI

    plt.show()



duration = 10
grid_size = 1000
C_max = 0.9
v = 1
xmax, xmin = 1, -1

f_init, f, x = run_UP1_RK1(grid_size)
f_ex, x = run_exact(grid_size)

plt.plot(x, f_init)
plt.title('Distribution f(x) at t=0')
plt.xlabel('x')
plt.ylabel('<f(x)>')

# Save the plot as an image file
plt.savefig("initial_distr_block_UP1_RK1.png", dpi=300, bbox_inches='tight')  # Saves as PNG with 300 DPI

plt.show()

plt.plot(x, f_ex, label='exact', ls='--')
plt.plot(x, f, label='UP1 RK1')
plt.title(f'Distribution f(x) at t={duration}')
plt.xlabel('x')
plt.ylabel('<f(x)>')
plt.legend()

# Save the plot as an image file
plt.savefig("final_distr_block_UP1_RK1.png", dpi=300, bbox_inches='tight')  # Saves as PNG with 300 DPI

plt.show()


duration = 10
v = 1
C_max = 0.9
xmax, xmin = 1, -1
grid_min = 64  
reps = 9
calc_norm_evo(reps)