# sample CCS-reduced x, number of selected components = 200

from pathlib import PureWindowsPath, Path
import numpy as np
import sys
import os
from multiprocessing import Pool
from scipy.stats import laplace
import pywt

if len(sys.argv) > 1 and sys.argv[1] == 'cluster':
        cluster = 1
        os.chdir(r'/zhome/00/d/170891/Python/Mixture_Representations_for_the_Prior')
        sys.path.append(r'/zhome/00/d/170891/Python/Mixture_Representations_for_the_Prior')
else: cluster = 0

from Mixture_Representations_for_the_Prior import utils, eval_samples
from Mixture_Representations_for_the_Prior.oneD_pw_const_wavelet_deblurr_large import sample_x_CCS_MALA

def paral_work_MCMC(input):
    par, sam, ch_ii = input
    
    # sampling
    np.random.seed(ch_ii)
    sam['x0'] = utils.load(par['run_dir'] / 'x_MAP')[sam['I']]
    sam['x0'] *= np.random.rand(sam['x0'].size) * 2
    sample_x_CCS_MALA.main(par, sam, ch_ii)

    # draw not selected components from prior 
    x_path = sam['sam_dir'] / 'samples_x_full' / ('ch'+str(ch_ii))
    Path(x_path).mkdir(parents=True, exist_ok=True)
    _, _, _, d_coeff, _, coeff_slices, coeff_shapes, delta, _, _ = utils.load(Path( par['run_dir'] / 'problem' ))
    x_full = np.zeros((d_coeff, sam['N_po'] ))
    x_full[sam['I'], :] = eval_samples.load_chain(sam['sam_dir']/('ch'+str(ch_ii)), sam['N_po']//sam['N_save'])
    x_full[sam['Ic'], :] = utils.fast_Laplace(delta[sam['Ic']], sam['N_po'])
    np.save(Path(x_path/'p_0.npy'), x_full)

    # transform to signal space
    s_path = sam['sam_dir'] / 'samples_signal' / ('ch'+str(ch_ii))
    Path(s_path).mkdir(parents=True, exist_ok=True)
    syn = lambda x: pywt.waverecn(pywt.unravel_coeffs(x, coeff_slices, coeff_shapes, 'wavedecn'), par['wavelet'], par['ext_mode_wave'])
    s_samples = np.apply_along_axis(syn, axis=0, arr=x_full) 
    np.save(Path(s_path/'p_0.npy'), s_samples)
        
def main():
     
    sam_nr = '2'

    # parameters
    sam = {
        'n_ch' : 5,
        'sam_dir' : PureWindowsPath(r'oneD_pw_const_wavelet_deblurr_large\Runs\conf0\sample_x_CCS_MALA', 'sam'+sam_nr),
        'N_po' : 1_000,   
        'N_b' : 100_000,
        'N_save' : 1_000,
        'th' : 10,
        'h0' : 5e-2,
        'acc_int_disp' : 1000,
        'd_I' : 200,
        'diagno_file' : PureWindowsPath(r'oneD_pw_const_wavelet_deblurr_large\Runs\conf0\sample_x_MALA_ref\sam0\h_ref'),
        'adapt_step' : True,
        'eps' : 1e-8, # for smoothing of gradient of prior
    }
    Path.mkdir(Path(sam['sam_dir']), parents=True, exist_ok=True)
    utils.save( sam['sam_dir'] / 'par', sam)

    # problem parameters
    par_dir = PureWindowsPath(r'oneD_pw_const_wavelet_deblurr_large\Runs\conf0\par')
    par = utils.load( par_dir )
    _, _, _, d_coeff, _, _, _, delta, _, _ = utils.load(Path( par['run_dir'] / 'problem' ))
    
    # select indices
    h = utils.load( sam['diagno_file'] )
    I_sor = np.argsort(h)[::-1] # indices for descending ordering
    sam['I'] = np.sort( I_sor[:sam['d_I']] ) 
    sam['Ic'] = np.sort( I_sor[sam['d_I']:] )
    utils.save(sam['sam_dir'] / 'par', sam)

    # parallel sampling and evaluation of samples
    TASKS = [(par, sam, ch_ii) for ch_ii in range(sam['n_ch'])]
    with Pool(processes=sam['n_ch']) as pool:
        pool.map(paral_work_MCMC, TASKS)
    # paral_work_MCMC(TASKS[0]) # for debugging

    # evaluate samples in reduced coefficient space
    flags = {'mean': 1, 'HDI': 0, 'CI': 1, 'ESS': 1, 'rhat' : 1}
    options = {'CI': 0.99}
    eval_samples.main(sam['sam_dir'], sam['n_ch'], sam['N_po']//sam['N_save'], flags, options)

    # add mean and CI from not selected components
    stats = utils.load(sam['sam_dir'] / 'stats')
    mean = np.zeros(d_coeff)
    mean[sam['I']] = stats['mean']
    stats['mean'] = mean
    CI = np.zeros((d_coeff, 2))
    for ii in sam['Ic']:
        CI[ii, 0] = laplace.ppf(q=(1-options['CI'])/2, loc=0, scale=1/delta[ii])
        CI[ii, 1] = laplace.ppf(q=(1+options['CI'])/2, loc=0, scale=1/delta[ii])
    CI[sam['I'], :] = stats['CI']
    stats['CI'] = CI
    utils.save(sam['sam_dir'] / 'stats', stats)

    # evaluate full samples in coefficient space
    flags = {'mean': 1, 'HDI': 0, 'CI': 1, 'ESS': 1, 'rhat' : 1}
    options = {'CI': 0.99}
    eval_samples.main(sam['sam_dir'] / 'samples_x_full', sam['n_ch'], 1, flags, options, remove_samples=True)

    # evaluate samples in signal space
    flags = {'mean': 1, 'HDI': 0, 'CI': 1, 'ESS': 0, 'rhat' : 0}
    options = {'CI': 0.60}
    eval_samples.main(sam['sam_dir'] / 'samples_signal', sam['n_ch'], 1, flags, options, remove_samples=True)
    
if __name__ == '__main__':
    main()