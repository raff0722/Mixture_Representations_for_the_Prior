# sample dimension-reduced v with MALA, 50 selected components

from pathlib import PureWindowsPath, Path, PurePath
import numpy as np
import sys
import os
from multiprocessing import Pool
from scipy.stats import expon

if len(sys.argv) > 1 and sys.argv[1] == 'cluster':
        cluster = 1
        os.chdir(r'/zhome/00/d/170891/Python/Mixture_Representations_for_the_Prior')
        sys.path.append(r'/zhome/00/d/170891/Python/Mixture_Representations_for_the_Prior')
else: cluster = 0

from Mixture_Representations_for_the_Prior import utils, eval_samples
from Mixture_Representations_for_the_Prior.oneD_pw_const_wavelet_deblurr_large import sample_v_CCS_MALA

def paral_work_MCMC(input):
    par, sam, ch_ii = input
    
    # sampling
    np.random.seed(ch_ii)
    sam['x0'] = utils.load(par['run_dir'] / 'v_MAP_L-BFGS-B')[0].x[sam['I']]
    sam['x0'] *= np.random.rand(sam['x0'].size) * 2
    sample_v_CCS_MALA.main(par, sam, ch_ii)

def main(): 
    
    sam_nr = '0'

    # parameters
    sam = {
        'n_ch' : 5,
        'sam_dir' : PureWindowsPath(r'oneD_pw_const_wavelet_deblurr_large\Runs\conf0\sample_v_CCS_MALA', 'sam'+sam_nr),
        'N_po' : 10,#1000,   
        'N_b' : 10,#100_000,
        'N_save' : 5,#200,
        'th' : 10,
        'h0' : 5e-2,
        'adapt_step' : True,
        'acc_int_disp' : 1000,
        'n_I' : 50,
        'diagno' : PureWindowsPath(r'oneD_pw_const_wavelet_deblurr_large\Runs\conf0\sample_v_MALA_ref\sam0\h_ref'),
        'w' : False # wether samples are in w-space or v-space (v requires back transformation)
    }
    Path.mkdir(Path(sam['sam_dir']), parents=True, exist_ok=True)
    
    # problem parameters
    par_dir = PureWindowsPath(r'oneD_pw_const_wavelet_deblurr_large\Runs\conf0\par')
    par = utils.load( Path( par_dir ) )
    _, _, _, d_coeff, _, _, _, delta, _, _ = utils.load(Path( par['run_dir'] / 'problem' ))

    # select coordinates
    diagno = utils.load(sam['diagno'])
    sam['I'] = np.argsort(diagno)[-sam['n_I']:]
    sam['Ic'] = np.setdiff1d(np.arange(d_coeff), sam['I'])
    utils.save(Path(sam['sam_dir']) / 'par', sam)

    # parallel sampling and evaluation of samples
    TASKS = [(par, sam, ch_ii) for ch_ii in range(sam['n_ch'])]
    with Pool(processes=sam['n_ch']) as pool:
        pool.map(paral_work_MCMC, TASKS)
    # paral_work_MCMC(TASKS[0]) # for debugging
    
    # evaluate samples
    flags = {'mean': 1, 'HDI': 0, 'CI': 1, 'ESS': 1, 'rhat' : 1}
    options = {'CI': 0.99}
    eval_samples.main(sam['sam_dir'], sam['n_ch'], sam['N_po']//sam['N_save'], flags, options)

    # add mean and CI from not selected components to dim red stats
    stats = utils.load(PurePath( sam['sam_dir'], 'stats'))
    mean = 1/(2/delta**2)
    mean[sam['I']] = stats['mean']
    stats['mean'] = mean
    CI = np.zeros((d_coeff, 2))
    for ii in sam['Ic']:
        CI[ii, 0] = expon.ppf(q=(1-options['CI'])/2, loc=0, scale=1/(2/delta[ii]**2))
        CI[ii, 1] = expon.ppf(q=(1+options['CI'])/2, loc=0, scale=1/(2/delta[ii]**2))
    CI[sam['I'], :] = stats['CI']
    stats['CI'] = CI
    utils.save(sam['sam_dir'] / 'stats', stats)

if __name__ == '__main__':
    main()