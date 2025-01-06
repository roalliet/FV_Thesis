import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.stats import linregress
    
def init():
    #determine some initial quantities
    t = 0
    min, max = -np.pi, np.pi
    dx = (max- min)/(grid_size-1) # dv = dx

    #create the grid for the volume integrated quantities at the volume centers
    x = np.arange(grid_size) * dx + min + dx / 2
    v = np.arange(grid_size) * dx + min + dx / 2
    X, V = np.meshgrid(x, v)
    
    F = np.zeros_like(X)
    for i in range(grid_size):
        for j in range(grid_size):
            # Calculate the values for conditions at each point
            X_val = X[i, j]
            V_val = V[i, j]
            
            d1 = np.sqrt(X_val**2 + (V_val - np.pi/2)**2)
            d2 = np.sqrt(X_val**2 + (V_val + np.pi/2)**2)
            d3 = np.sqrt((X_val + np.pi/2)**2 + V_val**2)
            
            # Apply the piecewise function conditions
            #if d1 <= 0.3 * np.pi and (abs(X_val) >= np.pi / 20 or V_val >= 0.3 * np.pi):
                #F[i, j] = 1
            if d2 <= 0.3 * np.pi:
                F[i, j] = 1 - d2 / (0.3 * np.pi)
            if d3 <= 0.3 * np.pi:
                F[i, j] = 0.25 * (1 + np.cos(np.pi * d3 / (0.3 * np.pi)))

    #determine the face centered values of the time indipendent part of the function g_x and g_v
    #for now we assume periodic boundaries so x_f[n+1/2] = x_f[1/2] with n = grid_size
    x_f = np.arange(grid_size) * dx + min + dx
    v_f = np.arange(grid_size) * dx + min + dx
    X_f, V_f = np.meshgrid(x_f, v_f)
    G0_xf = np.pi * np.cos(X_f/2) * np.cos(X_f/2) * np.sin(V) #centered on half x valued faces
    G0_vf = np.pi * np.cos(V_f/2) * np.cos(V_f/2) * np.sin(X) #centered on half v-valued faces

    return F, G0_xf, G0_vf, t, dx, dx


def time_step(dx, dv, Gx, Gv):
    #determines the maximal value of dt based on the CFL criterion
    C_max = 0.3
    dt = C_max * dx *dv / (abs(max(Gx.min(), Gx.max(), key=abs)) * dv + abs(max(Gv.min(), Gv.max(), key=abs)) * dx)
    return dt


def calc_F_f(F, h, ax):
    #calculate the face integrated values of f
    F_f = (np.roll(F, -1, axis=ax) + F) / (2 *h)
    return F_f


def calc_GFx(F, G0_xf, dx, dv, t):
    #determine the point values of g at the face centers from the analyticaly function
    g_f = G0_xf * np.cos(np.pi * t / 1.5) * np.pi

    #get the face integrated values
    F_f = calc_F_f(F, dx, ax=1)

    #determine the point values of f at the face centers from the face integrated values
    f_f = F_f / dv

    #calculate the face integrated fluxes on the x-faces
    GFx = dv * f_f * g_f
    return GFx
    
    
def calc_GFv(F, G0_vf, dx, dv, t):
    #determine the point values of g at the face centers from the analyticaly function
    g_f = G0_vf * np.cos(np.pi * t / 1.5) * np.pi

    #get the face integrated values
    F_f = calc_F_f(F, dv, ax=0)

    #determine the point values of f at the face centers from the face integrated values
    f_f = F_f / dx

    #calculate the face integrated fluxes on the x-faces
    GFv = dx * f_f * g_f
    return GFv


def advance_time(F, G0_xf, G0_vf, dv, dx, t):
    #determine the point values of g at the face centers from the analyticaly function
    g_fx = G0_xf * np.cos(np.pi * t / 1.5) * np.pi
    g_fv = G0_vf * np.cos(np.pi * t / 1.5) * np.pi

    #determine the time step dt
    dt = time_step(dx, dv, g_fx, g_fv)

    #ensures code always stops exactly at specified end time "duration"
    if t+dt > duration:
        dt = duration-t

    #determine the fluxes on the faces
    GFx = calc_GFx(F, G0_xf, dx, dv, t)
    GFv = calc_GFv(F, G0_vf, dx, dv, t)

    #determine the volume integrated values after a half timestep
    F_half = F + dt / 2 * (GFx - np.roll(GFx, 1, axis=1) - GFv + np.roll(GFv, 1, axis=0))

    #determine the point values of g at the face centers from the analyticaly function at t+dt/2
    
    #calculate the fluxes at t+dt/2
    GFx = calc_GFx(F_half, G0_xf, dx, dv, t+dt/2)
    GFv = calc_GFv(F_half, G0_vf, dx, dv, t+dt/2)

    F = F + dt * (GFx - np.roll(GFx, 1, axis=1) - GFv + np.roll(GFv, 1, axis=0))
    
    return F, dt


def plot_distr(F, t):
    im1 = plt.imshow(F, extent=[-np.pi,np.pi,np.pi,-np.pi])
    plt.colorbar(im1, label='Distribution F(x, t)')
    plt.xlabel('x')
    plt.ylabel('v')
    plt.title(f'Distribution F(x, t) after t = {t:.3f} sec')

    # Save the plot as an image file
    plt.savefig(f'swirling_deformation_figs_CD2_RK2/swirl_diff_res_{grid_size}_t_{t:.2f}.png', dpi=300, bbox_inches='tight')  # Saves as PNG with 300 DPI

    plt.show()


def run_CD2_RK2(plot=False):
    #create initial values 
    F_init, G0_xf, G0_vf, t, dx, dv = init()
    F = F_init.copy()
    if not plot:
        plot = duration 

    #plot the initial distribution
    plot_distr(F, t)
        
    #evolve the distribution in time while updating the time
    while t < duration:
        F, dt = advance_time(F, G0_xf, G0_vf, dv, dx, t)
        t += dt

        #plot the distribution every so often, if plot is not False
        if t % plot < dt:
            plot_distr(F, t)

    return F_init, F, t


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


def calc_norm_evo():
    #determine the relative norms of the system for different resolutions
    L1 = np.empty(reps)
    L2 = np.empty(reps)
    L_inf = np.empty(reps)

    res = grid_min * 2 ** np.arange(reps)
    for exp in range(reps):
        print(exp)
        #determine gridsize
        global grid_size
        grid_size = grid_min * 2**exp

        #calculate the numerical and exact solutions
        F_init, F, _ = run_CD2_RK2()

        #determine the relative value of L1
        L1_ex = calc_L1(F_init)
        if L1_ex == 0:
            L1[exp] = 1
        else:
            L1[exp] = np.abs(calc_L1(F, F_init)) / L1_ex

        #determine the relative value of L2
        L2_ex = calc_L2(F_init)
        if L2_ex == 0:
            L2[exp] = 1
        else:
            L2[exp] = np.abs(calc_L2(F, F_init)) / L2_ex

        #determine the relative value of L2
        L_inf_ex = calc_L_inf(F_init)
        if L_inf_ex == 0:
            L_inf[exp] = 1
        else:
            L_inf[exp] = np.abs(calc_L_inf(F, F_init)) / L_inf_ex
    
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
    plt.loglog(res, np.exp(intercept_L1) * (res)** slope_L1, 'r-', ls='--', label=f'Fit: y = {np.exp(intercept_L1):.2f} * x^{slope_L1:.2f}')
    plt.title('Relative L1-norm using CD2 RK2 with respect to the gridsize')
    plt.xlabel('number of gridpoints')
    plt.ylabel('relative L1-norm')
    plt.legend()
    plt.grid()

    # Save the plot as an image file
    plt.savefig("swirling_deformation_figs_CD2_RK2/swirl_diff_L1_CD2_RK2.png", dpi=300, bbox_inches='tight')  # Saves as PNG with 300 DPI

    plt.show()

    #plot the resulting values of L2 w.r.t. the gridsize
    plt.loglog(res, L2, label='Numerical datapoints of L2')
    plt.loglog(res, np.exp(intercept_L2) * (res)** slope_L2, 'r-', ls='--', label=f'Fit: y = {np.exp(intercept_L2):.2f} * x^{slope_L2:.2f}')
    plt.title('Relative L2-norm using CD2 RK2 with respect to the gridsize')
    plt.xlabel('number of gridpoints')
    plt.ylabel('relative L2-norm')
    plt.legend()
    plt.grid()

    # Save the plot as an image file
    plt.savefig("swirling_deformation_figs_CD2_RK2/swirl_diff_L2_CD2_RK2.png", dpi=300, bbox_inches='tight')  # Saves as PNG with 300 DPI

    plt.show()

    #plot the resulting values of L-infty w.r.t. the gridsize
    plt.loglog(res, L_inf, label='Numerical datapoints of L-$\\infty$')
    plt.loglog(res, np.exp(intercept_L_inf) * (res)** slope_L_inf, 'r-', ls='--', label=f'Fit: y = {np.exp(intercept_L_inf):.2f} * x^{slope_L_inf:.2f}')
    plt.title('Relative L-$\\infty$-norm using CD2 RK2 with respect to the gridsize')
    plt.xlabel('number of gridpoints')
    plt.ylabel('relative L-$\\infty$-norm')
    plt.legend()
    plt.grid()

    # Save the plot as an image file
    plt.savefig("swirling_deformation_figs_CD2_RK2/swirl_diff_L_infty_CD2_RK2.png", dpi=300, bbox_inches='tight')  # Saves as PNG with 300 DPI

    plt.show()



#define some system values
grid_size = 256
duration = 1.5

#run program and plot 6 times during runtime
F_init, F, _ = run_CD2_RK2(plot=0.25)

#run simulation at different resolutions to determine the relative norms for different gridsizes
duration = 1.5
grid_min = 16
reps = 6
#run program and plot norms
calc_norm_evo()
