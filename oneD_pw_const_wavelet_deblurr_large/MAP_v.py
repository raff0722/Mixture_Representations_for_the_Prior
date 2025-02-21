from scipy.optimize import minimize, LinearConstraint, check_grad, differential_evolution
import numpy as np
from pathlib import Path
from scipy.stats import uniform

from Mixture_Representations_for_the_Prior import utils, log_pdf_mixing

def main(par, method, v0, n):

    # load problem
    [x_sig, y_true, y, d_coeff, x_coeff, coeff_slices, coeff_shapes, delta, I_nz, A_mat] = utils.load( Path(par['run_dir']/ 'problem') )
    
    # for the computation of the log pdf of W|Y=y
    lam = 1/par['noise_std']**2
    til_A = lam * A_mat.T @ A_mat
    til_y = lam * A_mat.T @ y

    # parameters
    maxiter = 20_000
    gtol = 1e-8
    ftol = 1e-16
    
    # starting point(s)
    v0_n = np.zeros((d_coeff, n))
    v0_n[:, 0] = v0
    if n>1: v0_n = uniform.rvs(loc=np.min(v0), scale=np.max(v0)-np.min(v0), size=(d_coeff,n))

    # optimization without Hessian
    if method == 'diff_ev':
        fun = lambda v: -log_pdf_mixing.log_pdf_v(v, delta, til_A, til_y, pdf=True, grad=False, Hess=False)[0]
        res = differential_evolution(func=fun, bounds=[(-100, 100)]*d_coeff, disp='True', polish='False', x0=v0)

    # optimization without Hessian
    if method == 'CG':
        options = {'maxiter': maxiter, 'gtol': gtol, 'norm' : np.inf, 'disp': True}
        fun = lambda v: tuple(map(lambda x: -x, log_pdf_mixing.log_pdf_v(v, delta, til_A, til_y, pdf=True, grad=True, Hess=False)[:2]))
        res = minimize(fun, v0, method='CG', jac=True, options=options )
        # OptimizeWarning: Desired error not necessarily achieved due to precision loss.
        #   res = _minimize_cg(fun, x0, args, jac, callback, **options)
        #          Current function value: -310505.807350
        #          Iterations: 4
        #          Function evaluations: 72
        #          Gradient evaluations: 60

    # optimization without Hessian
    if method == 'BFGS': # results in overflow (reformulation of log pdf required)
        res=[]
        for ii in range(n):
            options = {'maxiter': maxiter, 'gtol': gtol, 'norm' : np.inf, 'xrtol' : 1e-16, 'disp': True}
            fun = lambda v: tuple(map(lambda x: -x, log_pdf_mixing.log_pdf_v(v, delta, til_A, til_y, pdf=True, grad=True, Hess=False)[:2]))
            res.append( minimize(fun, v0_n[:,ii], method='BFGS', jac=True, options=options) )
        # OptimizeWarning: Desired error not necessarily achieved due to precision loss.
        #   res = _minimize_bfgs(fun, x0, args, jac, callback, **options)
        #          Current function value: -310380.130686
        #          Iterations: 13
        #          Function evaluations: 64
        #          Gradient evaluations: 53

    # optimization without Hessian
    if method == 'L-BFGS-B':
        bounds = [(-1e8, None)]*d_coeff # [(None, None)]*d_coeff # [(1e-6, None)]*d_coeff
        fun = lambda v: tuple(map(lambda x: -x, log_pdf_mixing.log_pdf_v(v, delta, til_A, til_y, pdf=True, grad=True, Hess=False)[:2]))
        res=[]
        for ii in range(n):
            options = {'maxiter': maxiter, 'disp': True, 'ftol' : ftol} 
            res.append( minimize(fun, v0_n[:,ii], method='L-BFGS-B', jac=True, bounds=bounds, options=options) )

    # # optimization with Hessian
    # if method == 'Newton-CG':
    #     xtol = 1e-12
    #     options = {'maxiter': maxiter, 'xtol': xtol, 'disp': True}
    #     fun = lambda v: tuple(map(lambda x: -x, log_pdf_mixing.log_pdf_v(v, delta, til_A, til_y, pdf=True, grad=True, Hess=False)[:2]))
    #     Hess = lambda v: -log_pdf_mixing.log_pdf_v(v, delta, til_A, til_y, pdf=False, grad=False, Hess=True)[2]
    #     res=[]
    #     for ii in range(n):
    #         res.append( minimize(fun, v0_n[:,ii], method='Newton-CG', jac=True, hess=Hess, options=options) )
    #     # Optimization terminated successfully.
    #     #  Current function value: -310517.033674
    #     #  Iterations: 21
    #     #  Function evaluations: 21
    #     #  Gradient evaluations: 21
    #     #  Hessian evaluations: 21

    # # optimization with Hessian
    # if method == 'dogleg': # fails since Hessian not always pd
    #     options = {'maxiter': maxiter, 'gtol': gtol, 'disp': True}
    #     fun = lambda v: tuple(map(lambda x: -x, log_pdf_mixing.log_pdf_v(v, delta, til_A, til_y, pdf=True, grad=True, Hess=False)[:2]))
    #     Hess = lambda v: -log_pdf_mixing.log_pdf_v(v, delta, til_A, til_y, pdf=False, grad=False, Hess=True)[2]
    #     res = minimize(fun, v0_n, method='dogleg', jac=True, hess=Hess, options=options)

    # # optimization with Hessian
    # if method == 'trust-ncg': 
    #     options = {'maxiter': maxiter, 'gtol': gtol, 'disp': True}
    #     fun = lambda v: tuple(map(lambda x: -x, log_pdf_mixing.log_pdf_v(v, delta, til_A, til_y, pdf=True, grad=True, Hess=False)[:2]))
    #     Hess = lambda v: -log_pdf_mixing.log_pdf_v(v, delta, til_A, til_y, pdf=False, grad=False, Hess=True)[2]
    #     res=[]
    #     for ii in range(n):
    #         res.append( minimize(fun, v0_n[:,ii], method='trust-ncg', jac=True, hess=Hess, options=options) )
    #     #  RuntimeWarning: A bad approximation caused failure to predict improvement.
    #     #   res = _minimize_trust_ncg(fun, x0, args, jac, hess, hessp,
    #     #          Current function value: -310507.236848
    #     #          Iterations: 26
    #     #          Function evaluations: 28
    #     #          Gradient evaluations: 6
    #     #          Hessian evaluations: 6

    # # optimization with Hessian
    # if method == 'trust-krylov': 
    #     options = {'maxiter': maxiter, 'gtol': gtol, 'disp': True}
    #     fun = lambda v: tuple(map(lambda x: -x, log_pdf_mixing.log_pdf_v(v, delta, til_A, til_y, pdf=True, grad=True, Hess=False)[:2]))
    #     Hess = lambda v: -log_pdf_mixing.log_pdf_v(v, delta, til_A, til_y, pdf=False, grad=False, Hess=True)[2]
    #     res = minimize(fun, v0, method='trust-krylov', jac=True, hess=Hess, options=options)
    #     #  RuntimeWarning: A bad approximation caused failure to predict improvement.
    #     #   res = _minimize_trust_krylov(fun, x0, args, jac, hess, hessp,
    #     #          Current function value: -310506.181495
    #     #          Iterations: 26
    #     #          Function evaluations: 28
    #     #          Gradient evaluations: 28
    #     #          Hessian evaluations: 7

    # # optimization with Hessian
    # if method == 'trust-exact': 
    #     options = {'maxiter': maxiter, 'gtol': gtol, 'disp': True}
    #     fun = lambda v: tuple(map(lambda x: -x, log_pdf_mixing.log_pdf_v(v, delta, til_A, til_y, pdf=True, grad=True, Hess=False)[:2]))
    #     Hess = lambda v: -log_pdf_mixing.log_pdf_v(v, delta, til_A, til_y, pdf=False, grad=False, Hess=True)[2]
    #     res = minimize(fun, v0, method='trust-exact', jac=True, hess=Hess, options=options)
    #     # RuntimeWarning: A bad approximation caused failure to predict improvement.
    #     #   res = _minimize_trustregion_exact(fun, x0, args, jac, hess,
    #     #          Current function value: -310500.104538
    #     #          Iterations: 26
    #     #          Function evaluations: 28
    #     #          Gradient evaluations: 8
    #     #          Hessian evaluations: 28
                
    return res