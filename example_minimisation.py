import numpy as np
from TfFunctions import spiral_weighing


# Here you need to specify some 2xN array of (Z,W) coordinates.
coords = ...
sw = spiral_weighing(coords=coords, mask_params=[[300., 800.], [20., 44.], [2.5e-1, 1e-1]], num_Gs=6, pixels_smear=2.)

# First step of the minimisation procedure, when fitting the bulk density distribution.
fixed_solar_params = [0., 0.]
print("Minimize bulk")
minusLogPosterior, p0 = sw.minimize_bulk_likelihood(p0=None, learning_rate=1e-1, number_of_iterations=1000, print_gap=200, fixed_solar_params=fixed_solar_params)
minusLogPosterior, p0 = sw.minimize_bulk_likelihood(p0=p0, learning_rate=1e-3, number_of_iterations=10000, print_gap=1000, fixed_solar_params=fixed_solar_params)


# Second step of the minimisation procedure, when fitting the bulk density distribution.
# The mask function is slighly smaller in this step, according to the paper.
sw = spiral_weighing(coords=coords, mask_params=[[300., 700.], [20., 40.], [2.5e-1, 1e-1]], num_Gs=6, pixels_smear=2., interp_resolution=1000, mask_z=mask_z)
# Minus-log-posterior-array.
mlp_vec = []
# Array of free parameters in "vectorform" (see main class description for more details).
vector_vec = []
# This is a quite low but safe learning rate.
lr = 5e-4
for sp_i in range(40):
    print('Minimize spiral, with learning rate:', lr, " -- iteration:", sp_i)
    minusLogPosterior, p0 = sw.minimize_spiral_likelihood(p0=p0, learning_rate=lr, number_of_iterations=2000)
    mlp_vec.append(minusLogPosterior)
    vector_vec.append(p0)
    # Here we save the different steps of the minimisation procedure.
    np.savez("./Results/minsearch_example", logposts=mlp_vec, vectors=vector_vec)