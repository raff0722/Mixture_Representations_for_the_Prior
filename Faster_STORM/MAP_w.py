from scipy.optimize import minimize
from scipy.stats import uniform
import numpy as np
from pathlib import Path

from Mixture_Representations_for_the_Prior import log_pdf_mixing, utils

def main(par, method, w0, n):
    """Computes the minimizer of -log pdf of W|Y=y in [0,\infty) (i.e., w_i=0 allowed!!). 
    Methods are 'L-BFGS-B' and 'TNC'. n is the number of different starting points.

    Args:
        par (_type_): _description_
        method (_type_): _description_
        w0 (_type_): _description_
        n (_type_): _description_
    """    

    # load problem
    A_mat, y, lam, _, _, _, d2, _ = utils.load( Path(par['run_dir'] / 'problem' ))
    delta = np.ones(d2) * par['delta']

    # starting point(s)
    w0_n = np.zeros((d2, n))
    w0_n[:, 0] = w0
    if n>1: w0_n = uniform.rvs(loc=np.min(w0), scale=np.max(w0)-np.min(w0), size=(d2,n))

    # for the computation of the log pdf of W|Y=y
    til_A = lam * A_mat.T @ A_mat
    til_y = lam * A_mat.T @ y

    # optimization parameters
    maxiter = 100_000
    bounds = [(0, None)]*d2 # [(1e-6, None)]*d
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

    return res