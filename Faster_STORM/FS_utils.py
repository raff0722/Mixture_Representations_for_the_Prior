from scipy.io import loadmat
from scipy.sparse import coo_array, dia_array, eye_array, lil_array, csr_array, dok_array
from scipy.sparse.linalg import inv, spsolve, cg
from scipy.linalg import cho_factor, cho_solve, cholesky_banded, cho_solve_banded, solve
import numpy as np
import cvxpy as cp
import sys
from pathlib import Path

from Mixture_Representations_for_the_Prior import utils

rav = lambda x: np.ravel(x, order='F')
unrav = lambda x, d: np.reshape(x, newshape=(d,d), order='F')

def blur_matrix(par):
    """Loads the MATLAB matrix from the original paper and adds boundary conditions.
    The resulting matrix is saved as sparse scipy array.

    Args:
        par (_type_): _description_

    Returns:
        _type_: _description_
    """    
   
    # load data from MATLAB file
    matfile = loadmat(par['A_mat_file'])
    A_noBC = coo_array( np.array( matfile['A'] ) )
    dim = int(matfile['dim'])
    R = int(matfile['div'])
    pad_width = int(matfile['pad_width'])

    x_sr_dim = dim*R # super resolution side length
    pad_width_sr = pad_width*R 

    # create blur operators respecting BCs
    if par['ext_mode'] == 'periodic':
        def blur_op(x): # x without background variable
            y = A_noBC @ rav( np.pad( unrav(x, x_sr_dim), pad_width=pad_width_sr, mode='wrap' ) )
            return y
    elif par['ext_mode'] == 'zero':
        def blur_op(x): # x without background variable
            y = A_noBC @ rav( np.pad( unrav(x, x_sr_dim), pad_width=pad_width_sr, mode='constant' ) )
            return y
    else: sys.exit('ext mode not known')
        
    # # check linearity
    # x0 = np.random.rand((dim*R)**2)
    # x1 = np.random.rand((dim*R)**2)
    # alpha = 3
    # beta = -3
    # print(f'Additivity: {np.max( np.abs( blur_op(x0+x1) - (blur_op(x0)+blur_op(x1)) ) )}')
    # print(f'Homogeneity 1: {np.max( np.abs( blur_op(alpha*x0) - alpha*blur_op(x0) ) )}')
    # print(f'Homogeneity 2: {np.max( np.abs( blur_op(beta*x0) - beta*blur_op(x0) ) )}')

    # compute sparse matrix
    A = dok_array((dim**2, (dim*R)**2))
    for ii in range((dim*R)**2):
        print(f'iteration {ii+1}/{(dim*R)**2}', end='\r')
        e = np.insert( np.zeros((dim*R)**2-1, dtype=np.double), ii, 1.0 )
        A_e = A(e)
        ind = np.nonzero(A_e)[0]
        A[ind, ii] = A_e[ind]
    A = csr_array(A)    
    A[np.abs(A)<1e-11] = 0

    # normalize, c can be used to precondition rate parameter, however is constant here anyways
    c = np.ravel( np.sum( A, axis=0 ) )
    PSF_integ = np.max(c) # integration of the PSF over space, used for normalization
    c = c/ PSF_integ # normalize to 1
    A = A/ PSF_integ
    
    # # add the extra optimization variable for the estimation of the background
    # c = np.append(c, 0)
    # A = hstack( ( A, np.ones( (A.shape[0], 1) ) ) )
    
    # sparse A
    A = csr_array(A)

    utils.save( par['A_file'], A )


def map_cvxpy(par):

    A, y, lam, _, _, _, _, _ = utils.load( Path(par['run_dir'] / 'problem') )

    # Construct the problem.
    x = cp.Variable(d2)    
    objective = cp.Minimize( lam/2* cp.sum_squares( y-A@x ) + par['delta'] * cp.norm(x, 1) )
    prob = cp.Problem(objective)

    result = prob.solve(max_iter=10_000, verbose=1)

    if prob.status != 'optimal':
        sys.exit('CVXPY OPTIMIZATION NOT OPTIMAL')
    
    return x.value
    


# def ab_array(M, d):
#     # computes lower form of banded symmetric hermitian matrix M
#     # (input for scipy function)
#     ab = M.diagonal(0)
#     not0 = True
#     k = 1
#     while not0:
#         # print(f'diagonal {k}', end='\r')
#         d_k = M.diagonal(k)
#         if d_k.any():
#             ab_k = np.zeros((1, d))
#             ab_k[0, :-k] = d_k
#             ab = np.vstack((ab, ab_k))
#             k += 1
#         else: not0 = False 

#     return ab

# def w_giv_y(par):

#     [A_mat, y, lam, y_truth, x_im_truth, ind_mol, d2, m2] = utils.load( Path(par['run_dir'] / 'problem' ))
    
#     til_A = lam* csr_array(A_mat.T@ A_mat)
#     til_A_diag = til_A.diagonal()
#     til_y = lam* A_mat.T@ y 
    
#     def log_pdf(w):

#         I = np.nonzero(w)[0]
#         Isize = I.size
#         if Isize == 0:
#             return 0
        
#         # Schur decomposition
#         # M = [A B] = [A 0]
#         #     [B D]   [0 I]
#         sqrt_w_I = np.sqrt(w[I])
#         cho = cho_factor( d_r(d_l(sqrt_w_I, til_A[np.ix_(I,I)].todense()), sqrt_w_I) + np.eye(Isize) )
#         w_y_I = sqrt_w_I*til_y[I]

#         t1 = - par['delta']**2/2 * np.sum(w)     
#         t2 = -1/2 * 2* np.sum( np.log( np.diag(cho[0]) ) )
#         t3 = 1/2 * np.inner( w_y_I, cho_solve(cho, w_y_I) )  

#         return t1 + t2 + t3
    
#     def grad_log_pdf(w):
    
#         # first term
#         t1 = -par['delta']**2/2

#         I = np.nonzero(w)[0] # indices not zero
#         Isize = I.size
#         print(f'{Isize} w_i not zero')
#         if Isize == 0:
#             t2 = -1/2 * til_A_diag
#             t3 = 1/2 * til_y**2
#             return t1 + t2 + t3

#         # Schur decomposition
#         # M = [A B] = [A 0]
#         #     [B D]   [0 I]
#         sqrt_w_I = np.sqrt(w[I])
#         cho = cho_factor( d_r(d_l(sqrt_w_I, til_A[np.ix_(I,I)].todense()), sqrt_w_I) + np.eye(Isize) )

#         # second term
#         TUA = d_l(sqrt_w_I, til_A[I, :].todense())
#         C1 = cho_solve(cho, TUA[:, :Isize])
#         C2 = cho_solve(cho, TUA[:, Isize:])
#         t2 = -1/2 *( til_A_diag - np.concatenate( (d_mp(TUA[:,:Isize].T, C1), d_mp(TUA[:,Isize:].T, C2)) ) ) 

#         # third term
#         t3 = 1/2 * (til_y - TUA.T@cho_solve(cho, sqrt_w_I*til_y[I]))**2

#         return t1 + t2 + t3
    
#     def Hess_log_pdf(w):

#         # via Woodbury 
#         sqrt_w = dia_array((np.sqrt(w),0), shape=(d2,d2))
#         chol_fac = cho_factor( (sqrt_w @ til_A @ sqrt_w + eye_array(d2, d2)).todense() )
#         M0 = eye_array(d2, d2) - til_A @ sqrt_w @ csr_array( cho_solve(chol_fac, sqrt_w.todense()) )
#         M1 = M0 @ til_A
#         Z = dia_array( (M0 @ til_y, 0), shape=(d2,d2) )
#         H = 1/2 * M1.power(2) - Z @ M1 @ Z
#         H = (H + H.T)/2

#         return H
    
#     return log_pdf, grad_log_pdf, Hess_log_pdf


# def rec_jumps(I_jumps, diagno):
    """Computes boolean vector indicating recovered jumps.
    
    Args:
        I_jumps (_type_): Indices of jumps
        diagno (_type_): Unsorted diagnostic

    Returns:
        _type_: Boolean vector indicating recovered jumps when h is sorted ASCENDINGLY
        _type_: Minimum number of coordinates to be selected to recover all jumps
    """    

    # boolean vector indicating jumps
    b_jumps = np.zeros(diagno.size, dtype=bool)
    b_jumps[I_jumps] = True

    # sort boolean vector according to sorted diagnnostic (ascending)
    I_diagno_sor = np.argsort(diagno)
    b_jumps_sor = b_jumps[I_diagno_sor]

    # find last entry to be True 
    # (number of coordinates to select (for set I) to recover all jumps)
    min_sel = diagno.size - np.argwhere(b_jumps_sor==True)[0]

    return b_jumps_sor, min_sel