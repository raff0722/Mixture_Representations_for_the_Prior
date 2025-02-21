import numpy as np
import sys
from pathlib import PurePath, Path
# import torch as xp
# import jax.numpy as jnp
from scipy.linalg import cho_factor, cho_solve
# import jax.scipy.linalg as jspl

from Mixture_Representations_for_the_Prior import utils

def d_r(A, d): # multiplication from the right
    # A @ np.diag( d ) without constructing the diagonal
    return np.einsum( 'ij, j -> ij', A, d)

def d_l(d, A): # multiplication from the left
    # np.diag( d ) @ A  without constructing the diagonal
    return np.einsum( 'ij, i -> ij', A, d)

def d_mp(A, B):
    # Diagonal of matrix product of two nxn square matrices (which do not have to be symmetric.)
    return np.einsum( 'ij, ji -> i', A, B) 
    
def log_pdf_w(w, delta, til_A, til_y, pdf=True, grad=False, Hess=False):
    """Efficiently computes the log pdf of W|Y=y and the gradient and the Hessian thereof (if flags set to True)
    by exploting zeros in w.

    Args:
        w (_type_): _description_
        delta (_type_): _description_
        til_A (_type_): _description_
        til_y (_type_): _description_
        pdf (bool, optional): _description_. Defaults to True.
        grad (bool, optional): _description_. Defaults to False.
        Hess (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """    

    log_pdf_eval, grad_log_pdf_eval, Hess_log_pdf_eval = None, None, None

    # check for zeros and do Schur decomposition (only for pdf and grad)
    I = np.nonzero(w)[0]
    Isize = I.size
    if Isize == 0:
        if pdf:
            log_pdf_eval = 0
        if grad:
            t1 = -delta**2/2
            t2 = -1/2 * np.diag(til_A)
            t3 = 1/2 * til_y**2
            grad_log_pdf_eval = t1 + t2 + t3
        if Hess:
            t1 = 1/2 * np.power(til_A, 2)
            t2 = - d_r(d_l(til_y, til_A), til_y)
    else:
        if pdf or grad:
            # Schur decomposition
            # M = [A B] = [A 0]
            #     [B D]   [0 I]
            sqrt_w_I = np.sqrt(w[I])
            cho = cho_factor( d_r(d_l(sqrt_w_I, til_A[np.ix_(I,I)]), sqrt_w_I) + np.eye(Isize) )
        
        # log pdf
        if pdf:
            w_y_I = sqrt_w_I*til_y[I]
            t1 = - np.sum(delta**2/2 * w)     
            t2 = -1/2 * 2* np.sum( np.log( np.diag(cho[0]) ) )
            t3 = 1/2 * np.inner( w_y_I, cho_solve(cho, w_y_I) )  
            log_pdf_eval = t1 + t2 + t3

            # # no tricks - check OK
            # sqrt_w = np.sqrt(w)
            # cho = cho_factor( d_r( d_l( sqrt_w, til_A), sqrt_w ) + np.eye(w.size) )
            # hat_y = np.multiply( sqrt_w, til_y )
            # t1c = -np.sum(delta**2/2 * w)         
            # t2c = -1/2 * 2* np.sum( np.log( np.diag(cho[0]) ) )
            # t3c = 1/2 * np.inner( hat_y, cho_solve(cho, hat_y) )  
            # print('pdf:')
            # print(f't1: {t1-t1c}')
            # print(f't2: {t2-t2c}')
            # print(f't3: {t3-t3c}')

        # gradient
        if grad:
            t1 = -delta**2
            
            # second term
            TUA = d_l(sqrt_w_I, til_A[I, :])
            C1 = cho_solve(cho, TUA[:, :Isize])
            C2 = cho_solve(cho, TUA[:, Isize:])
            t2 = -1/2 *( np.diag(til_A) - np.concatenate( (d_mp(TUA[:,:Isize].T, C1), d_mp(TUA[:,Isize:].T, C2)) ) ) 

            # third term
            t3 = 1/2 * (til_y - TUA.T@cho_solve(cho, sqrt_w_I*til_y[I]))**2
            
            grad_log_pdf_eval = t1 + t2 + t3
            
            # # no tricks - check OK
            # M_inv = np.linalg.inv( d_r(til_A,w) + np.eye(w.size) )
            # t1c = -delta**2/2
            # t2c = -1/2 * d_mp( M_inv, til_A)
            # t3c = 1/2 * ( M_inv @ til_y )**2
            # print('gradient:')
            # print(f't1: {np.max(np.abs(t1-t1c))}')
            # print(f't2: {np.max(np.abs(t2-t2c))}')
            # print(f't3: {np.max(np.abs(t3-t3c))}')

        # Hessian 
        if Hess:
            # # via Woodbury 
            # sqrt_w = np.diag(np.sqrt(w))
            # chol_fac = cho_factor( sqrt_w @ til_A @ sqrt_w + np.eye(w.size) )
            # M0 = np.eye(w.size) - til_A @ sqrt_w @ cho_solve(chol_fac, sqrt_w)
            # M1 = M0 @ til_A
            # Z = np.diag( M0 @ til_y )
            # t1 = 1/2 * np.power(M1, 2) 
            # t2 = - Z @ M1 @ Z
            # Hess_log_pdf_eval = t1 + t2
            # Hess_log_pdf_eval = (Hess_log_pdf_eval + Hess_log_pdf_eval.T)/2
            
            # brute force
            M_inv = np.linalg.inv( d_r(til_A, w) + np.eye(w.size) )
            M_inv_A = M_inv @ til_A
            M_inv_y = M_inv @ til_y
            t1 = 1/2 * M_inv_A**2
            t2 = - d_r( d_l( M_inv_y, M_inv_A), M_inv_y )
            Hess_log_pdf_eval = t1 + t2
            Hess_log_pdf_eval = (Hess_log_pdf_eval + Hess_log_pdf_eval.T)/2
            
            ## check
            # print('Hessian:')
            # print(f't1: {np.max(np.abs(t1-t1c))}')
            # print(f't2: {np.max(np.abs(t2-t2c))}')
        
    return log_pdf_eval, grad_log_pdf_eval, Hess_log_pdf_eval
    
def log_pdf_v(v, delta, til_A, til_y, pdf=True, grad=False, Hess=False):
    """Efficiently computes the log pdf of V|Y=y (after the transformation v=log(w))
    and the gradient and the Hessian thereof (if flags set to True).

    Args:
        v (_type_): _description_
        delta (_type_): _description_
        til_A (_type_): _description_
        til_y (_type_): _description_
        pdf (bool, optional): _description_. Defaults to True.
        grad (bool, optional): _description_. Defaults to False.
        Hess (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """    
    log_pdf_eval, grad_log_pdf_eval, Hess_log_pdf_eval = None, None, None

    cho = cho_factor(til_A + np.diag(np.exp(-v)))
    
    if pdf:    
        t1 = - np.sum( delta**2/2 * np.exp(v) )     
        t2 = -1/2 * 2 * np.sum( np.log( np.diag(cho[0]) ) )
        t3 = 1/2 * np.inner( til_y, cho_solve(cho, til_y) )
        t4 = 1/2 * np.sum( v )
        log_pdf_eval = t1 + t2 + t3 + t4
    
    if grad:
        t1 = - delta**2/2 * np.exp(v)
        t2 = - 1/2 * np.diag( cho_solve(cho, til_A) )
        t3 = 1/2 * ( np.exp(-v/2) * cho_solve(cho, til_y) )**2
        t4 = 1
        grad_log_pdf_eval = t1 + t2 + t3 + t4
    
    # if Hess: # not checked
    #     expv = np.exp(v)
    #     expv2 = np.exp(v/2)
    #     K = d_r(d_l(expv2, til_A), expv2)
    #     L = np.linalg.cholesky( K + np.eye(d) )
    #     L_inv = np.linalg.inv(L)
    #     M = L_inv.T @ L_inv
    #     N_inv =  d_r(M, expv2)
    #     N_inv_til_y = N_inv@til_y
    #     t1 = np.diag(-delta**2/2 * expv)
    #     t2 = ( - 1/2 * M @ K ) * M
    #     t3 = - 1/2 * d_r( d_l( N_inv_til_y, N_inv @ ( d_r( til_A, expv2 ) - np.diag( 1/expv2 ) ) ), N_inv_til_y ) 
    #     Hess_log_pdf_eval = t1 + t2 + t3
    
    return  log_pdf_eval, grad_log_pdf_eval, Hess_log_pdf_eval

def init_log_pdf_red_v(delta, til_A, til_y, I, offline_path=None):
    """Precomputes some matrices for the function log_pdf_red_v.
    This is possible by fixing some of the components in v to their prior mean.

    Args:
        delta (_type_): _description_
        til_A (_type_): _description_
        til_y (_type_): _description_
        I (_type_): _description_
        offline_path (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """    

    w_mean = 1 / ( delta**2/2 )
    v_mean = np.log( w_mean ) * np.ones(delta.size)

    Ic = np.setdiff1d(np.arange(delta.size), I)
    tilyI = til_y[I]
    tilyIc = til_y[Ic]
    til_y_per = np.concatenate( (tilyI, tilyIc) ) # permuted til y

    # offline computations
    if isinstance(offline_path, PurePath) and Path(offline_path).is_file():
        Dinv_BT, AII_m_B_Dinv_BT, B_Dinv_tilyIc, Dinv_tilyIc, AII_m_B_Dinv_AIcI, I_off = utils.load(offline_path)
        if I.size!=I_off.size: 
            sys.exit('I has different size as I from offline computations')
        if np.sum(np.abs(np.sort(I)-np.sort(I_off))) != 0: 
            sys.exit('I has different indices as I from offline computations')
    else:
        print('Pre-computing matrices for red dim w...')
        AII = til_A[np.ix_(I,I)]
        AIcI = til_A[np.ix_(Ic,I)]
        B = til_A[np.ix_(I,Ic)]
        D = til_A[np.ix_(Ic,Ic)] + np.diag( np.exp(-v_mean[Ic]) )
        #
        S, U = np.linalg.eigh(D)
        Dinv = U @ np.diag(1/S) @ U.T
        #
        Dinv_BT = Dinv @ B.T
        B_Dinv_BT = B @ Dinv_BT
        AII_m_B_Dinv_BT = AII - B_Dinv_BT
        B_Dinv_tilyIc = Dinv_BT.T @ tilyIc
        Dinv_tilyIc = Dinv @ tilyIc
        AII_m_B_Dinv_AIcI = AII - Dinv_BT.T @ AIcI
        #
        if isinstance(offline_path, PurePath):
            utils.save(offline_path, [Dinv_BT, AII_m_B_Dinv_BT, B_Dinv_tilyIc, Dinv_tilyIc, AII_m_B_Dinv_AIcI, I])

    return tilyI, til_y_per, Dinv_BT, AII_m_B_Dinv_BT, B_Dinv_tilyIc, Dinv_tilyIc, AII_m_B_Dinv_AIcI

def log_pdf_red_v(delta, til_A, til_y, I, offline_path=None, app_Grad=False):
    """Efficiently computes the reduced log pdf of V|Y=y (after the transformation v=log(w))
    and the approximate gradient (if flag set to True).

    Args:
        delta (_type_): _description_
        til_A (_type_): _description_
        til_y (_type_): _description_
        I (_type_): _description_
        offline_path (_type_, optional): _description_. Defaults to None.
        app_Grad (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """    

    tilyI, til_y_per, Dinv_BT, AII_m_B_Dinv_BT, B_Dinv_tilyIc, Dinv_tilyIc, AII_m_B_Dinv_AIcI = init_log_pdf_red_v(delta, til_A, til_y, I, offline_path)
    if app_Grad:
        norm_M_2 = np.linalg.norm(AII_m_B_Dinv_AIcI, ord=2, axis=0)**2

    def log_pdf_fun(v, pdf=True, grad=False, Hess=False):
        
        log_pdf_eval, grad_log_pdf_eval, Hess_log_pdf_eval = None, None, None

        S_chol = cho_factor( AII_m_B_Dinv_BT + np.diag( np.exp(-v) ) )
        # S_inv, logdet = fast_positive_definite_inverse( AII_m_B_Dinv_BT + np.diag( np.exp(-v) ), logdet=True )
        Sinv_yI = cho_solve(S_chol, tilyI)
        # Sinv_yI = S_inv @ tilyI
        Sinv_B_Dinv_tilyIc = cho_solve(S_chol, B_Dinv_tilyIc ) 
        # Sinv_B_Dinv_tilyIc = S_inv @ B_Dinv_tilyIc
        z1 = Sinv_yI - Sinv_B_Dinv_tilyIc

        if pdf:
            z2 = Dinv_BT @ ( -Sinv_yI + Sinv_B_Dinv_tilyIc ) + Dinv_tilyIc 
            t1 = - np.sum( delta[I]**2/2 * np.exp(v) )
            t2 = -1/2 * 2 * np.sum( np.log( np.diag(S_chol[0]) ) )
            # t2 = -1/2 * logdet
            t3 = 1/2 * np.inner(til_y_per, np.concatenate( (z1, z2) ))
            t4 = 1/2 * np.sum(v)
            log_pdf_eval = t1 + t2 + t3 + t4

        if grad:
            t1 = - delta[I]**2/2 * np.exp(v)
            if app_Grad:
                d = np.exp(-v)
                m = np.diag(AII_m_B_Dinv_AIcI)
                SPAI = ( m + d ) / ( norm_M_2+2*d*m+d**2 )
                t2 = -1/2 * SPAI*m
            else:
                t2 = -1/2 * np.diagonal(cho_solve(S_chol, AII_m_B_Dinv_AIcI))
                # t2 = -1/2 * np.sum( S_inv.T * AII_m_B_Dinv_AIcI, axis=0 )
            t3 = 1/2 * (np.exp(-v/2) * z1)**2
            t4 = 1

            grad_log_pdf_eval = t1 + t2 + t3 + t4

            # # check approximated gradient
            # t2c = -1/2 * np.diagonal(cho_solve(S_chol, AII_m_B_Dinv_AIcI))
            # print(np.linalg.norm(t2-t2c)/np.linalg.norm(t2c))
            # print(np.max(np.abs(t2-t2c)))
            # grad_log_pdf_eval_c = t1 + t2c + t3 + t4
            # print(np.linalg.norm(grad_log_pdf_eval-grad_log_pdf_eval_c)/np.linalg.norm(grad_log_pdf_eval_c))

        return log_pdf_eval, grad_log_pdf_eval, Hess_log_pdf_eval
    
    return log_pdf_fun

def log_pdf_red_v_pytorch(delta, til_A, til_y, I, offline_path=None):
    """Some function as log_pdf_red_v but for pyro.

    Args:
        delta (_type_): _description_
        til_A (_type_): _description_
        til_y (_type_): _description_
        I (_type_): _description_
        offline_path (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """    

    tilyI, til_y_per, Dinv_BT, AII_m_B_Dinv_BT, B_Dinv_tilyIc, Dinv_tilyIc, AII_m_B_Dinv_AIcI = init_log_pdf_red_v(delta, til_A, til_y, I, offline_path)

    # transform to pytorch
    d2_2 = xp.asarray( delta[I]**2/2, dtype=xp.float )
    tilyI = xp.asarray( tilyI, dtype=xp.float )[:,None]
    til_y_per = xp.asarray( til_y_per, dtype=xp.float )[:,None]
    Dinv_BT = xp.asarray( Dinv_BT, dtype=xp.float )
    AII_m_B_Dinv_BT = xp.asarray( AII_m_B_Dinv_BT, dtype=xp.float )
    B_Dinv_tilyIc = xp.asarray( B_Dinv_tilyIc, dtype=xp.float )[:, None]
    Dinv_tilyIc = xp.asarray( Dinv_tilyIc, dtype=xp.float )[:, None]
    AII_m_B_Dinv_AIcI = xp.asarray( AII_m_B_Dinv_AIcI, dtype=xp.float )

    def potential_fn(v):
        
        S_chol = xp.linalg.cholesky( AII_m_B_Dinv_BT + xp.diag( xp.exp(-v['x']) ) ) 
        Sinv_yI = xp.cholesky_solve(tilyI, S_chol)
        Sinv_B_Dinv_tilyIc = xp.cholesky_solve(B_Dinv_tilyIc, S_chol) 
        z1 = Sinv_yI - Sinv_B_Dinv_tilyIc
        z2 = Dinv_BT @ ( -Sinv_yI + Sinv_B_Dinv_tilyIc ) + Dinv_tilyIc 

        # log pdf
        t1 = - xp.sum( d2_2 * xp.exp(v['x']) )
        t2 = -1/2 * 2 * xp.sum( xp.log( xp.diag(S_chol) ) )
        t3 = 1/2 * xp.sum(til_y_per * xp.concatenate( (z1, z2) ))
        t4 = 1/2 * xp.sum(v['x'])
        pot = -( t1 + t2 + t3 + t4 )
              
        return pot
    
    return potential_fn


def log_pdf_red_v_jax(delta, til_A, til_y, I, offline_path=None):
    """Some function as log_pdf_red_v but for numpyro.

    Args:
        delta (_type_): _description_
        til_A (_type_): _description_
        til_y (_type_): _description_
        I (_type_): _description_
        offline_path (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """    

    tilyI, til_y_per, Dinv_BT, AII_m_B_Dinv_BT, B_Dinv_tilyIc, Dinv_tilyIc, AII_m_B_Dinv_AIcI = init_log_pdf_red_v(delta, til_A, til_y, I, offline_path)

    # transform to jax
    d2_2 = jnp.asarray( delta[I]**2/2, dtype=jnp.float32)
    tilyI = jnp.asarray( tilyI, dtype=jnp.float32 )
    til_y_per = jnp.asarray( til_y_per, dtype=jnp.float32 )
    Dinv_BT = jnp.asarray( Dinv_BT, dtype=jnp.float32 )
    AII_m_B_Dinv_BT = jnp.asarray( AII_m_B_Dinv_BT, dtype=jnp.float32 )
    B_Dinv_tilyIc = jnp.asarray( B_Dinv_tilyIc, dtype=jnp.float32 )
    Dinv_tilyIc = jnp.asarray( Dinv_tilyIc, dtype=jnp.float32 )
    AII_m_B_Dinv_AIcI = jnp.asarray( AII_m_B_Dinv_AIcI, dtype=jnp.float32 )

    def potential_fn(v):
        
        S_chol = jspl.cho_factor( AII_m_B_Dinv_BT + jnp.diag( jnp.exp(-v) ) ) 
        Sinv_yI = jspl.cho_solve(S_chol, tilyI)
        Sinv_B_Dinv_tilyIc = jspl.cho_solve(S_chol, B_Dinv_tilyIc) 
        z1 = Sinv_yI - Sinv_B_Dinv_tilyIc
        z2 = jnp.matmul(Dinv_BT, ( -Sinv_yI + Sinv_B_Dinv_tilyIc )) + Dinv_tilyIc 

        # log pdf
        t1 = - jnp.sum( jnp.multiply(d2_2, jnp.exp(v)) )
        t2 = -1/2 * 2 * jnp.sum( jnp.log( jnp.diag(S_chol[0]) ) )
        t3 = 1/2 * jnp.sum(til_y_per * jnp.concatenate( (z1, z2) ))
        t4 = 1/2 * jnp.sum(v)
        pot = -( t1 + t2 + t3 + t4 )
              
        return pot
    
    return potential_fn