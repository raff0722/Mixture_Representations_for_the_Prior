# sample x from Gaussian mixture terms via RTO (linear least squares solves) in parallel    

import time
import sys
from pathlib import Path, PurePath, PureWindowsPath
import numpy as np
from scipy.sparse.linalg import lsmr, lsqr
from scipy.sparse import vstack, dia_array, eye_array
from multiprocessing import Pool
import os
import pywt

if len(sys.argv) > 1 and sys.argv[1] == 'cluster':
    cluster = 1
    os.chdir(r'/zhome/00/d/170891/Python/Mixture_Representations_for_the_Prior')
    sys.path.append(r'/zhome/00/d/170891/Python/Mixture_Representations_for_the_Prior')
    conf = sys.argv[2]
    sam_dir = sys.argv[3] # sampling directory
    sam_nr = sys.argv[4] # number of sampling configuration
else: 
    cluster = 0
    conf = '0'
    sam_dir = 'sample_x_via_v_CCS_MALA' # sampling directory
    sam_nr = '0' # number of sampling configuration

from Mixture_Representations_for_the_Prior import utils, eval_samples

def initializer(in_gen, in_RTO):
    
    global d, I, Ic, delta
    global L1TA, L1Ty, dm
    global atol, btol

    d, I, Ic, delta = in_gen
    L1TA, L1Ty, dm = in_RTO
    atol, btol = 1e-8, 1e-8

def worker(input): 
    ii, w_ii = input
    np.random.seed(ii)

    # combine sample with draw from mixing prior
    w = np.zeros(d)
    w[I] = w_ii
    w[Ic] = utils.fast_exponential(delta[Ic]**2/2, 1).flatten()

    # call RTO
    print(f'sample {ii+1}', end='\r')
    x = sample_x_RTO(w)

    return ii, x

def sample_x_RTO(w):
    z = np.concatenate( (L1Ty, np.zeros(d)) ) + np.random.randn(dm) 
    
    # standard lsmr
    L2 = np.diag(1/np.sqrt(w))
    M = np.vstack( (L1TA, L2) ) #hstack( (ATL1, L2) ).T
    # t = time.time()
    res = lsmr(M, z, atol=atol, btol=btol, show=False) 
    x = res[0]
    # print(f'lsmr took {time.time()-t} seconds')
    
    # # preconditioned lsmr
    # L2_inv = dia_array( (np.sqrt(w),0), shape=(d,d) )
    # M_pre = vstack( (L1TA@L2_inv, eye_array(d,d)) )
    # t = time.time()
    # res = lsmr(M_pre, z, atol=atol, btol=btol, show=False)
    # x = L2_inv@res[0]
    # print(f'prec lsmr took {time.time()-t} seconds')

    # # lsqr
    # t = time.time()
    # res = lsqr(M, z, atol=atol, btol=btol, show=False)
    # print(f'lsqr took {time.time()-t} seconds')

    return x


if __name__ == '__main__':

    # load problem parameters
    par = utils.load(PureWindowsPath(r'oneD_pw_const_wavelet_deblurr_large\Runs', 'conf'+conf, 'par'))
    [x_sig, y_true, y, d_coeff, x_coeff, coeff_slices, coeff_shapes, delta, I_nz, A_mat] = utils.load(Path( par['run_dir'] / 'problem' ))
    m = y_true.size

    # load sampling parameters
    sam = utils.load(PureWindowsPath(r'oneD_pw_const_wavelet_deblurr_large\Runs', 'conf'+conf, sam_dir, 'sam'+sam_nr, 'par'))
    sam_w = utils.load(sam['sam_dir_w_samples'] / 'par')
    
    # general input
    in_gen = d_coeff, sam_w['I'], sam_w['Ic'], delta

    # for RTO
    lam = 1/par['noise_std']**2
    dm = d_coeff + m
    L1TA = np.sqrt(lam) * A_mat 
    L1Ty = np.sqrt(lam) * y
    in_RTO = L1TA, L1Ty, dm
    
    # initialize workers
    if sam['n_cores']>1: pool = Pool(processes=sam['n_cores'], initializer=initializer, initargs=(in_gen, in_RTO, ))
    else: initializer(in_gen, in_RTO)

    # go over all chains
    for ii in range(sam_w['n_ch']):

        print(f'Chain {ii+1}/{sam_w["n_ch"]}...')

        # location for samples (coefficients)
        save_dir_x = PurePath(sam['sam_dir'], 'samples_x', 'ch'+str(ii))
        Path(save_dir_x).mkdir(parents=True, exist_ok=True)

        # location for samples (after transformation back to signal space)
        save_dir_signal = PurePath(sam['sam_dir'], 'samples_s', 'ch'+str(ii))
        Path(save_dir_signal).mkdir(parents=True, exist_ok=True)
        
        # go over all packets of samples
        for jj in range(sam_w['N_po']//sam_w['N_save']):
             
            # load w samples (or v)
            w = np.load(Path(PurePath(sam_w['sam_dir'], 'ch'+str(ii), 'p_'+str(jj)+'.npy')))
            tasks = []
            for kk in range(w.shape[1]):
                if not sam_w['w']: tasks.append( (kk, np.exp(w[:, kk]) ) )
                else: tasks.append( (kk, w[:, kk]) )

            # get coefficient samples
            x_sam = np.zeros((d_coeff, sam_w['N_save']))
            if sam['n_cores']>1:
                results = pool.map_async(worker, tasks)
                results.wait()
                for ll, x_ll in results.get():
                    x_sam[:, ll] = x_ll
            else: 
                for task_ii in tasks: 
                    ll, x_ll = worker(task_ii)
                    x_sam[:, ll] = x_ll
            np.save(Path(PurePath(save_dir_x, 'p_'+str(jj)+'.npy')), x_sam)

            # transform samples back to signal space
            syn = lambda x: pywt.waverecn(pywt.unravel_coeffs(x, coeff_slices, coeff_shapes, 'wavedecn'), par['wavelet'], par['ext_mode_wave'])
            s = np.apply_along_axis(syn, axis=0, arr=x_sam)
            np.save(Path(PurePath(save_dir_signal, 'p_'+str(jj)+'.npy')), s)

    # close pool
    if sam['n_cores']>1:
        pool.close() 
        pool.join()

    # evaluate samples in coefficient space
    flags = {'mean': 1, 'HDI': 0, 'CI': 1, 'ESS': 1, 'rhat' : 1}
    eval_samples.main(sam['sam_dir'] / 'samples_x', sam_w['n_ch'], sam_w['N_po']//sam_w['N_save'], flags, options = {'CI': 0.95})

    # evaluate samples in signal space
    flags = {'mean': 1, 'HDI': 0, 'CI': 1, 'ESS': 0, 'rhat' : 0}
    eval_samples.main(sam['sam_dir'] / 'samples_s', sam_w['n_ch'], sam_w['N_po']//sam_w['N_save'], flags, options = {'CI': 0.60}, name='stats_60')
    flags = {'mean': 1, 'HDI': 0, 'CI': 1, 'ESS': 0, 'rhat' : 0}
    eval_samples.main(sam['sam_dir'] / 'samples_s', sam_w['n_ch'], sam_w['N_po']//sam_w['N_save'], flags, options = {'CI': 0.90}, name='stats_90')

   
    # # check
    # initializer(in_gen, in_RTO)
    # Nc = 1000
    # x_RTO = np.zeros((d_coeff, Nc))
    # for ii in range(Nc):
    #     print(f'sample {ii+1}/{Nc}', end='\r')
    #     x_RTO[:,ii] = sample_x_RTO(w[:, 0])
    # import matplotlib.pyplot as plt
    # plt.subplot(1,2,1)
    # plt.plot(np.mean(x_chol, axis=1))
    # plt.plot(np.mean(x_RTO, axis=1))
    # plt.subplot(1,2,2)
    # plt.plot(np.std(x_chol, axis=1))
    # plt.plot(np.std(x_RTO, axis=1))
    # plt.show()


    # CG with sparse diagonal preconditioner  
    # def worker(ii):
    #     L2 = dia_array((1/np.sqrt(w_p[:,ii]), 0), shape=(d2_p, d2_p))
    #     zeta = np.random.randn(m2_p) 
    #     gamma = np.random.randn(d2_p)
    #     u = L1_A_matT_p @ zeta + L2 @ gamma
    #     b = til_y_p + u
    #     x, info = cg(pos_prec(w_p[:,ii]), b, x0_p, maxiter=1000, M=SPAI_prec(w_p[:,ii]) )
    #     print(f'sample {ii}: info={info}', end='\r')
    #     return x
    #     pos_prec = lambda w: til_A_p + dia_array((1/w, 0), shape=(d2_p, d2_p))
    #     def SPAI_prec(w):
    #           w_inv = 1/w
    #           return dia_array( ( (m_ii_p+w_inv) / (norm_til_A_i_2_p + 2*m_ii_p*w_inv + w_inv**2), 0 ), shape=(d2_p, d2_p) )