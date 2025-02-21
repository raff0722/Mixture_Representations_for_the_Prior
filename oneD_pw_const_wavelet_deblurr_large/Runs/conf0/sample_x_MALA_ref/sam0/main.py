import sys
from pathlib import PureWindowsPath, Path
import numpy as np
import os
from multiprocessing import Pool
import pywt

if len(sys.argv) > 1 and sys.argv[1] == 'cluster':
        cluster = 1
        os.chdir(r'/zhome/00/d/170891/Python/Mixture_Representations_for_the_Prior')
else: cluster = 0

from Mixture_Representations_for_the_Prior import utils, eval_samples
from oneD_pw_const_wavelet_deblurr_large import sample_x_MALA_ref

def paral_work_MCMC(input):
    par, sam, ch_ii = input
    
    # sampling
    np.random.seed(ch_ii)
    sam['x0'] = utils.load(par['run_dir'] / 'x_MAP')
    sam['x0'] *= np.random.rand(sam['x0'].size) * 2
    sample_x_MALA_ref.main(par, sam, ch_ii)

    # estimate diagnostic
    _, _, y, _, _, _, _, delta, _, A_mat = utils.load( par['run_dir'] / 'problem' )
    lam = 1/par['noise_std']**2
    log_grad_like = lambda x: lam * A_mat.T@(y-A_mat@x)
    x = eval_samples.load_chain(sam['sam_dir']/('ch'+str(ch_ii)), sam['N_po']//sam['N_save'])
    h = 0
    for ii in range(x.shape[1]):
        print(f'ii={ii}', end='\r')
        h += log_grad_like(x[:,ii])**2
    utils.save(sam['sam_dir']/('ch'+str(ch_ii))/'h_ref', 1/x.shape[1]*1/delta**2*h)
    
def main():

    # parameters
    sam = {
        'n_ch' : 5,
        'sam_dir' : PureWindowsPath(r'oneD_pw_const_wavelet_deblurr_large\Runs\conf0\sample_x_MALA_ref\sam0'),
        'N_po' : 5_000,   
        'N_b' : 200_000,
        'h0' : 1e-6,
        'N_save' : 5_000,
        'th' : 100,
        'acc_int_disp' : 1000,
        'adapt_step' : True,
        'eps' : 1e-8
    }
    Path.mkdir(Path(sam['sam_dir']), parents=True, exist_ok=True)
    utils.save(sam['sam_dir'] / 'par', sam)
    
    # problem parameters
    par_dir = PureWindowsPath(r'oneD_pw_const_wavelet_deblurr_large\Runs\conf0\par')
    par = utils.load( Path( par_dir ))

    # parallel sampling and evaluation of samples
    TASKS = [(par, sam, ch_ii) for ch_ii in range(sam['n_ch'])]
    with Pool(processes=sam['n_ch']) as pool:
        pool.map(paral_work_MCMC, TASKS)
    # paral_work_MCMC(TASKS[0]) # for debugging

    # evaluate samples in coefficient space
    flags = {'mean': 1, 'HDI': 0, 'CI': 1, 'ESS': 1, 'rhat' : 1}
    options = {'CI': 0.99}
    eval_samples.main(sam['sam_dir'], sam['n_ch'], sam['N_po']//sam['N_save'], flags, options)
    stats = utils.load(Path( sam['sam_dir'] / 'stats') )
    print(f'mean ESS: {np.mean(stats["ESS"])}/{sam["N_po"]}') # 956.7182168558674/5000
    print(f'max rhat: {np.max(stats[-1]["rhat"])}') # 1.025254279250724

    # reference diagnostic over all chains
    h = utils.load(sam['sam_dir']/'ch0'/'h_ref')
    for ch_ii in range(1, sam['n_ch']):
        h += utils.load(sam['sam_dir']/('ch'+str(ch_ii))/'h_ref') 
    utils.save(sam['sam_dir']/'h_ref', 1/sam['n_ch']*h)

    # transform to signal space
    _, _, _, _, _, coeff_slices, coeff_shapes, _, _, _ = utils.load( par['run_dir'] / 'problem' )
    syn = lambda x: pywt.waverecn(pywt.unravel_coeffs(x, coeff_slices, coeff_shapes, 'wavedecn'), par['wavelet'], par['ext_mode_wave'])
    for ch_ii in range(sam['n_ch']):
        x = np.load(Path(sam['sam_dir'] / ('ch'+str(ch_ii)) / 'p_0.npy'))
        print('transform x to signal space...')
        s = np.apply_along_axis(syn, axis=0, arr=x)
        save_path = sam['sam_dir'] / 'samples_signal' / ('ch'+str(ch_ii))
        Path(save_path).mkdir(parents=True, exist_ok=True)
        np.save( Path(save_path / 'p_0.npy'), s)

    # evaluate samples in signal space
    flags = {'mean': 1, 'HDI': 0, 'CI': 1, 'ESS': 1, 'rhat' : 1}
    eval_samples.main(sam['sam_dir'] / 'samples_signal', sam['n_ch'], sam['N_po']//sam['N_save'], flags, options={'CI': 0.60}, name='stats_60')
    flags = {'mean': 1, 'HDI': 0, 'CI': 1, 'ESS': 0, 'rhat' : 0}
    eval_samples.main(sam['sam_dir'] / 'samples_signal', sam['n_ch'], sam['N_po']//sam['N_save'], flags, options={'CI': 0.90}, name='stats_90')

if __name__ == '__main__':
    main()