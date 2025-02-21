from scipy.optimize import minimize
from scipy.stats import uniform
import numpy as np
from pathlib import Path

from My_modules import pickle_routines
from Mixture_Representations_for_the_Prior import log_pdf_mixing

def main(par, method, w0, n):
    """Computes the minimizer of -log pdf of W|Y=y in [0,\infty) (i.e., w_i=0 allowed!!). 
    Methods are 'L-BFGS-B', 'TNC' and 'trust-constr'. n is the number of different starting points.

    Args:
        par (_type_): _description_
        method (_type_): _description_
        w0 (_type_): _description_
        n (_type_): _description_
    """    

    # load problem
    _, _, y, d_coeff, _, _, _, delta, _, A_mat = pickle_routines.load( Path(par['run_dir'] / 'problem' ))
    lam = 1/par['noise_std']**2

    # starting point(s)
    w0_n = np.zeros((d_coeff, n))
    w0_n[:, 0] = w0
    if n>1: w0_n = uniform.rvs(loc=np.min(w0), scale=np.max(w0)-np.min(w0), size=(d_coeff,n))

    # for the computation of the log pdf of W|Y=y
    til_A = lam * A_mat.T @ A_mat
    til_y = lam * A_mat.T @ y

    # optimization parameters
    maxiter = 20_000
    bounds = [(0, None)]*d_coeff # [(1e-6, None)]*d
    gtol = 1e-8
    ftol = 1e-16

    if method == 'L-BFGS-B':
        # constrained BFGS
        fun = lambda w: tuple(map(lambda x: -x, log_pdf_mixing.log_pdf_w(w, delta, til_A, til_y, pdf=True, grad=True)[:2]))
        res=[]
        for ii in range(n):
            options = {'maxiter': maxiter, 'disp': True, 'gtol' : gtol, 'ftol' : ftol, 'maxcor' : 5} 
            res.append( minimize(fun, w0_n[:,ii], method='L-BFGS-B', jac=True, bounds=bounds, options=options) )

    if method == 'TNC':
        # truncated conjugate gradient
        fun = lambda w: tuple(map(lambda x: -x, log_pdf_mixing.log_pdf_w(w, delta, til_A, til_y, pdf=True, grad=True)[:2]))
        res=[]
        for ii in range(n):
            options = {'maxiter': maxiter, 'disp': True, 'gtol': gtol } # 'ftol' : ftol} 
            res.append( minimize(fun, w0[:,ii], method='TNC', jac=True, bounds=bounds, options=options) )

    if method == 'trust-constr': ## bounds are violated, I dont know why
        # trust region
        fun = lambda w: tuple(map(lambda x: -x, log_pdf_mixing.log_pdf_w(w, delta, til_A, til_y, pdf=True, grad=True)[:2]))
        Hess = lambda w: -log_pdf_mixing.log_pdf_w(w, delta, til_A, til_y, pdf=False, grad=False, Hess=True)[2]
        n = 5
        res=[]
        for ii in range(n):
            options = {'maxiter': maxiter, 'verbose': 2, 'gtol': gtol } # 'ftol' : ftol} 
            res.append( minimize(fun, w0[:,ii], method='trust-constr', jac=True, hess=Hess, bounds=bounds, options=options) )

    return res

# SLSQP
# constraints for SLSQP 
# A = np.eye(d)
# constraints = LinearConstraint(A, lb=0, ub=np.inf, keep_feasible=True) # LinearConstraint(A, lb=1e-6, ub=np.inf, keep_feasible=True) 
# options = {'maxiter': maxiter, 'ftol': 1e-8, 'disp': True}
# res_SLSQP = minimize(fun, w0, method='SLSQP', jac=grad_fun, constraints=constraints, options=options)
# pickle_routines.save( par['run_dir'] + '/w_MAP_SLSQP', res_SLSQP)
# Optimization terminated successfully    (Exit mode 0)
# Current function value: -315162.06046622526
# Iterations: 10
# Function evaluations: 36
# Gradient evaluations: 6
# `gtol` termination condition is satisfied.
