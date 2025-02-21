from Mixture_Representations_for_the_Prior import utils

from pathlib import Path
import numpy as np
from time import time
import matplotlib.pyplot as plt

def prep_cus_step_adaption(N_b, dh_b, b=1, plot=False):
    """Creates upper bound function (with geometric decay) for step size adaption during burn-in according to 
    
    Marshall, Tristan, and Gareth Roberts. “An Adaptive Approach to Langevin MCMC.” 
    Statistics and Computing 22, no. 5 (2012): 1041–57. https://doi.org/10.1007/s11222-011-9276-6.

    Args:
        N_b (_type_): Length of burn-in
        dh_b (_type_): Last maximal step adaption
        b (int, optional): Coefficient in c(n). Defaults to 1.
        plot (bool, optional): Whether to plot c(n). Defaults to False.

    Returns:
        _type_: _description_
    """

    r = - np.log( dh_b/b ) / ( np.log(N_b) ) 
    c = lambda n: b * n**(-r)

    nn = np.arange(1, N_b+1, dtype=int) 
    c_nn = c(nn)
    
    if plot:
        
        plt.figure()
        plt.semilogy(nn, c_nn, label=f'c(i)')
        plt.show()

    print(f'last adaptation: {c_nn[-1]}')

    return c

def sample(log_tar_grad, h0, x0, N, N_save, save_dir, seed, th=1, N_b=0, adapt_step=False, c=None, acc_int_disp=1000, logpdf=False):

    d = x0.size
    Path.mkdir(Path(save_dir), parents=True, exist_ok=True)

    # sample array and storage
    n_parts = N//N_save
    sam_per_part = [N_save] * n_parts
    if N%N_save:
        sam_per_part.append(N%N_save)
        n_parts +=1 
    sam_per_part.append(0) # add dummy to avoid 'list index out of range' 

    # accepted samples
    acc = np.zeros(N * th + N_b + 1, dtype=bool)
    acc[0] = True # initial state is accepted

    # save log pdf evaluations
    lpdf = []

    # adaptive step size
    h = h0
    M = 10
    tar_acc = 0.574

    # initialize state 
    x = np.zeros((d, sam_per_part[0]))
    x_state = x0
    log_tar_state, log_tar_state_grad = log_tar_grad(x0)
    mean_state = x_state + h * log_tar_state_grad
    
    # set seed
    if isinstance(seed, dict):
        np.random.set_state(seed)
    else:
        rand_gen = np.random.default_rng(seed)

    kk = 0 # counter for saving to physical storage
    jj = 0 # counter for N samples
    t0 = time() # total time
    for ii in range(1, N * th + N_b + 1):

        if ii == N_b+1: 
            adapt_step = False
            t1 = time() # time without burn-in

        # monitoring of sampling
        if ii%acc_int_disp == 0:
            sum_acc_int = np.sum( acc[ ii-acc_int_disp: ii ])
            t_int = (time() - t0)/60
            t_proj = t_int/ii * (N * th + N_b)
            print('sample {}/{} -- acc {:>4}/{} -- step size {:.2e} -- time {:.2f}/{:.2f} minutes'.format(ii, N*th+N_b, sum_acc_int, acc_int_disp, h, t_int, t_proj), end='\r')

        # propose
        if adapt_step: mean_state = x_state + h * log_tar_state_grad
        x_prop = mean_state + np.sqrt(2*h) * rand_gen.normal(size=d)
        log_tar_prop, log_tar_prop_grad = log_tar_grad(x_prop)
        mean_prop = x_prop + h * log_tar_prop_grad
        q_forw = -1/(4*h) *np.linalg.norm( x_prop - mean_state )**2
        q_back = -1/(4*h) *np.linalg.norm( x_state - mean_prop )**2
    
        # acceptance prob
        log_alpha = min( 0, (log_tar_prop + q_back) - (log_tar_state + q_forw) )

        # accept/reject
        if log_alpha > np.log(np.random.random()): # accept
            x_state = x_prop
            log_tar_state = log_tar_prop
            log_tar_state_grad = log_tar_prop_grad
            mean_state = mean_prop
            acc[ii] = True
        else: # reject
            pass

        # adapt step size
        if adapt_step and ii <= N_b and ii >= M:
            h_star = min( 0.001*h, c(ii) )
            acc_rate_M = np.sum( acc[(ii - M + 1) : (ii + 1)] ) / M
            if acc_rate_M < tar_acc:
                h -= h_star
            else: # acc_rate_M >= tar_acc:
                h += h_star

        # save sample
        if ii > N_b and (ii-N_b)%th==0:
            x[:, jj] = x_state
            if logpdf: lpdf.append(log_tar_state)
            jj += 1

            # new samples array if needed
            if jj == sam_per_part[kk]:    
                np.save( Path( save_dir / ('p_' + str(kk)) ), x, allow_pickle=False )
                out = {'h':h, 'rand_state':np.random.get_state()}
                utils.save( Path( save_dir / ('out_'+ str(kk)) ), out)
                kk += 1
                x = np.zeros((d, sam_per_part[kk]))
                jj = 0
    
    out = {'acc': acc[N_b:], 'time':[t1-t0,time()-t1] , 'h':h, 'lpdf':np.array(lpdf)}
    utils.save( Path( save_dir / 'out' ), out)