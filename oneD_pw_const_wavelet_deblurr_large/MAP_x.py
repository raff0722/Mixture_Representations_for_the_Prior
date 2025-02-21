import numpy as np
import cvxpy as cp
import sys

from Mixture_Representations_for_the_Prior import utils

def main(par):

    [x_sig, y_true, y, d_coeff, x_coeff, coeff_slices, coeff_shapes, delta, I_nz, A_mat] = utils.load(par['run_dir'] / 'problem')
    d = x_coeff.size
    lam = 1/par['noise_std']**2

    x = cp.Variable(d)
    objective = cp.Minimize( lam/2 *cp.sum_squares( y -A_mat @x ) + cp.norm( cp.multiply(delta, x), 1 ) )
    prob = cp.Problem(objective)
    result = prob.solve(max_iter=100_000, verbose=1)

    if prob.status != 'optimal': sys.exit('CVXPY OPTIMIZATION NOT OPTIMAL')

    return x.value


