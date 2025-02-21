# sample from Laplace approximation in x via RTO

from pathlib import PureWindowsPath, Path, PurePath
import numpy as np
import sys
import os
from multiprocessing import Pool
from scipy.sparse import dia_array, vstack
from scipy.sparse.linalg import lsmr

if len(sys.argv) > 1 and sys.argv[1] == 'cluster':
        cluster = 1
        os.chdir(r'/zhome/00/d/170891/Python/Mixture_Representations_for_the_Prior')
        sys.path.append(r'/zhome/00/d/170891/Python/Mixture_Representations_for_the_Prior')
        conf = sys.argv[2]
        sam_r = sys.argv[3]
else: 
    cluster = 0
    conf = '0'
    sam_r = '0'

from Mixture_Representations_for_the_Prior import utils, eval_samples

def initializer(input):
    global L1Ty, L2Tmu_0, L1TA, L2, L2inv
    L1Ty, L2Tmu_0, L1TA, L2, L2inv = input

def paral_RTO(ii):
    np.random.seed(ii)
    z = np.concatenate( (L1Ty, L2Tmu_0) ) + np.random.randn(L1Ty.size + L2Tmu_0.size) 
    M = vstack( (L1TA, L2) )
    return lsmr(M, z, atol=1e-9, btol=1e-9, show=False)[0]
       
if __name__ == '__main__':
    
    # problem parameters
    par = utils.load(PureWindowsPath(r'Faster_STORM\Runs\conf'+conf+'\par'))
    [A_mat, y, lam, y_truth, x_im_truth, ind_mol, d2, m2] = utils.load(Path( par['run_dir'] / 'problem' ))

    # sampling parameters
    sam = utils.load(PureWindowsPath(r'Faster_STORM\Runs\conf'+conf+'\sam'+sam_r+'\par'))

    # MAP
    x_MAP = utils.load(sam['map'])

    # smoothed inverse Hessian of prior of posterior approximation
    eps = 4/np.max(par['delta']**2)
    Sig_pr_inv = par['delta'] * eps * 1/np.sqrt(x_MAP**2 + eps)**3

    # prepare RTO
    L1Ty = np.zeros_like(y)
    L2Tmu_0 = np.zeros(d2)
    L1TA = np.sqrt(lam) * A_mat
    L2 = dia_array( (np.sqrt(Sig_pr_inv), 0), shape=(d2, d2)) 
    L2inv = dia_array( (1/np.sqrt(Sig_pr_inv), 0), shape=(d2, d2))
    input = L1Ty, L2Tmu_0, L1TA, L2, L2inv

    # save dir
    save_dir_chain = sam['sam_dir'] / ('sam'+sam_r) / 'ch0'
    Path(save_dir_chain).mkdir(parents=True, exist_ok=True)

    # prepare pool
    pool = Pool(processes=sam['n_proc'], initializer=initializer, initargs=(input, ))

    # start pool
    for n in range(sam['N_po']//sam['N_save']):
        print(f'Part {n+1}/{sam["N_po"]//sam["N_save"]}...')

        # # for debugging
        # t0 = time()
        # x_sam = np.zeros((d2, sam['N_save']))
        # ll = 0
        # for ii in range(n*sam['N_save'], (n+1)*sam['N_save']):
        #     print(f'sample {ii}...')
        #     x_sam[:, ll] = paral_RTO(ii) + x_MAP
        #     ll += 1
        # print(f'time: {time()-t0}') 
        # #33.28672695159912
        # #56.50369095802307 with prec

        # parallel sampling
        results = pool.map_async(paral_RTO, np.arange(n*sam['N_save'], (n+1)*sam['N_save']))
        results.wait()
    
        # Get results
        x_sam = np.zeros((d2, sam['N_save']))
        ll = 0
        for x_ll in results.get():
            x_sam[:, ll] = x_ll + x_MAP
            ll += 1
        np.save(Path(PurePath(save_dir_chain, 'p_'+str(n)+'.npy')), x_sam)
    
    pool.close()
    pool.join()