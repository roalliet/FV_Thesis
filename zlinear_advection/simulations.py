import time

from time_evolution import *
from initial_distr import init, exact
from visualisation import *
from error_analysis import *
from interpolation import *


# Run simulation for given duration and parameters --> output: plot of final distribution
def run_comp(params, exact_grid_size, save):
    # Create the initial and final volume averaged distributions using the exact solution
    # We might want to use different gridsizes to have the exact solution as a proper background to the possibly lower resolution numerical solution
    f, x = init(params)
    f_ex, x_ex = exact(params, grid_size=exact_grid_size)

    # As dt is constant for most of the simulation we can use a for loop with num_steps, avoiding a while loop
    num_steps = int(params.duration // params.dt)
    final_step = params.duration % params.dt

    # Evolve the distribution in time while updating the time
    for _ in range(num_steps):
        f = time_evo(f, params)
        params.t += params.dt
    
    if final_step > 0:
        params.dt = final_step
        f = time_evo(f, params)
        params.t += params.dt

    plot_final(f_ex, x_ex, f, x, params, save)


def ani_comp(params, exact_grid_size, save):
    # Save timestamp
    start_time = time.time()

    # As dt is constant for most of the simulation we can use a for loop with num_steps, avoiding a while loop
    num_steps = int(params.duration // params.dt)
    
    plot_ani_comp(params, exact_grid_size, num_steps, save)

    # Save timestamp
    end_time = time.time()
    # Calculate duration
    duration_seconds = end_time - start_time
    minutes = int(duration_seconds // 60)
    seconds = duration_seconds % 60

    print(f"Function executed in: {minutes} minutes and {seconds:.2f} seconds")
    

def run_multip_interp(params, exact_grid_size, save):
    # Create arrays to store final solutions in
    f_list = np.empty((params.grid_size, params.mult_len))

    # Determine the exact solution at t = 0 (requires an awkward reset of params.duration)
    dur_temp = params.duration
    params.duration = 0
    f_init, x_init = exact(params, grid_size=exact_grid_size)
    params.duration = dur_temp

    # Create the initial and final volume averaged distributions using the exact solution
    f, x = init(params)
    f_ex, x_ex = exact(params, grid_size=exact_grid_size)

    for i in range(params.mult_len):
        # Set stencil to the correct interpolation annd resets t and dt
        params.new_stencil(params.mult_stencil[i])

        # Determine the interpolation of face averaged values at all relevant grid points
        f_list[:, i] = calc_flux(f, params.stencil)

    # Plot the face averaged values
    plot_multip_interp(f_init, x_init, f_list, x, params, 0, save)

    # As dt is constant for most of the simulation we can use a for loop with num_steps, avoiding a while loop
    num_steps = int(params.duration // params.dt)
    final_step = params.duration % params.dt

    for i in range(params.mult_len):
        # Set stencil to the correct interpolation annd resets t and dt
        params.new_stencil(params.mult_stencil[i])
        f, _ = init(params)

        # Evolve the distribution in time while updating the time
        for _ in range(num_steps):
            f = time_evo(f, params)
            params.t += params.dt
        
        if final_step > 0:
            params.dt = final_step
            f = time_evo(f, params)
            params.t += params.dt

        # Determine the interpolation of face averaged values at all relevant grid points
        f_list[:, i] = calc_flux(f, params.stencil)

    plot_multip_interp(f_ex, x_ex, f_list, x, params, num_steps, save)




def run_interp(params, exact_grid_size, save):
    # Determine the exact solution at t = 0 (requires an awkward reset of params.duration)
    dur_temp = params.duration
    params.duration = 0
    f_init, x_init = exact(params, grid_size=exact_grid_size)
    params.duration = dur_temp

    # Create the initial and final volume averaged distributions using the exact solution
    f, x = init(params)
    f_ex, x_ex = exact(params, grid_size=exact_grid_size)

    # Determine the interpolation of face averaged values at all relevant grid points
    f_interp = calc_flux(f, params.stencil)

    # Plot the face averaged values
    plot_interp(f_init, x_init, f_interp, x, params, 0, save)

    # As dt is constant for most of the simulation we can use a for loop with num_steps, avoiding a while loop
    num_steps = int(params.duration // params.dt)
    final_step = params.duration % params.dt

    # Evolve the distribution in time while updating the time
    for _ in range(num_steps):
        f = time_evo(f, params)
        params.t += params.dt
    
    if final_step > 0:
        params.dt = final_step
        f = time_evo(f, params)
        params.t += params.dt

    # Determine the interpolation of face averaged values at all relevant grid points
    f_interp = calc_flux(f, params.stencil)

    plot_interp(f_ex, x_ex, f_interp, x, params, num_steps, save)


def run_norm_err(params, reps, save):
    # Save timestamp
    start_time = time.time()

    # Initialize lists to store the relative error per norm
    L1 = np.empty(reps)
    L2 = np.empty(reps)
    L_inf = np.empty(reps)

    # Create a list to store the grid_size at each repetition
    res = params.grid_size * 2 ** np.arange(reps)

    for rep in range(reps):
        # Calculate the numerical and exact solutions
        f, _ = init(params)
        f_ex, _ = exact(params)

        # As dt is constant for most of the simulation we can use a for loop with num_steps, avoiding a while loop
        num_steps = int(params.duration // params.dt)
        final_step = params.duration % params.dt

        # Evolve the distribution in time while updating the time
        for _ in range(num_steps):
            f = time_evo(f, params)
            params.t += params.dt
        
        if final_step > 0:
            params.dt = final_step
            f = time_evo(f, params)
            params.t += params.dt

        #determine the relative value of error of the norms L1, L2, L\infty
        L1[rep] = calc_rel_L1(f, f_ex)
        L2[rep] = calc_rel_L2(f, f_ex)
        L_inf[rep] = calc_rel_L_inf(f, f_ex)

        # Print the grid_size that just finnished calculating
        print(f'Iteration {rep+1}/{reps} with gridsize {params.grid_size} is done computing.')

        # Double the gridsize for the next iteration and change the nesecary variables in params
        params.new_grid(params.grid_size * 2)

    plot_norm_err(L1, L2, L_inf, res, params, save)

    # Save timestamp
    end_time = time.time()
    # Calculate duration
    duration_seconds = end_time - start_time
    minutes = int(duration_seconds // 60)
    seconds = duration_seconds % 60

    print(f"Function executed in: {minutes} minutes and {seconds:.2f} seconds")