import sys
from pathlib import PureWindowsPath, Path
import numpy as np
import os
from multiprocessing import Pool

if len(sys.argv) > 1 and sys.argv[1] == 'cluster':
        cluster = 1
        os.chdir(r'/zhome/00/d/170891/Python/Mixture_Representations_for_the_Prior')
else: cluster = 0

from Mixture_Representations_for_the_Prior import utils, eval_samples, log_pdf_mixing
from Mixture_Representations_for_the_Prior.oneD_pw_const_wavelet_deblurr_large import sample_v_MALA_ref

# sample v in transformed space

def paral_work_MCMC(input):
    par, sam, ch_ii = input
    
    # sampling
    np.random.seed(ch_ii)
    sam['x0'] = utils.load(par['run_dir'] / 'v_MAP_L-BFGS-B')[0].x
    sam['x0'] *= np.random.rand(sam['x0'].size) * 2
    sample_v_MALA_ref.main(par, sam, ch_ii)

    # load chain and back-transform
    w = np.exp(eval_samples.load_chain(sam['sam_dir'] / ('ch'+str(ch_ii)), sam['N_po']//sam['N_save']))

    # estimate diagnostic
    [x_sig, y_true, y, d_coeff, x_coeff, coeff_slices, coeff_shapes, delta, I_nz, A_mat] = utils.load( par['run_dir'] / 'problem' )
    lam = 1/par['noise_std']**2
    til_A = lam* A_mat.T@ A_mat
    til_y = lam* A_mat.T@ y 
    grad = lambda v: log_pdf_mixing.log_pdf_v(v, delta, til_A, til_y, pdf=False, grad=True, Hess=False)[1]
    h = 0
    for ii in range(w.shape[1]):
        # print(f'nabla log pi(w|y) of sample {ii+1}/{sam["N_po"]}', end='\r')
        h += grad(w[:,ii])**2
    utils.save(Path(sam['sam_dir'] / ('ch'+str(ch_ii)) / 'h_ref'), 1/(delta**2/2)**2 * 1/w.shape[1] * h)

def main():

    # parameters
    sam = {
        'n_ch' : 5,
        'sam_dir' : PureWindowsPath(r'oneD_pw_const_wavelet_deblurr_large\Runs\conf0\sample_v_MALA_ref\sam0'),
        'N_po' : 5_000,   
        'N_b' : 200_000,
        'h0' : 5e-2,
        'N_save' : 5_000,
        'th' : 30,
        'adapt_step' : True,
        'acc_int_disp' : 1000,
    }
    Path.mkdir(Path(sam['sam_dir']), parents=True, exist_ok=True)
    utils.save(sam['sam_dir'] / 'par', sam)
    
    # problem parameters
    par_dir = PureWindowsPath(r'oneD_pw_const_wavelet_deblurr_large\Runs\conf0\par')
    par = utils.load( par_dir)
    
    # parallel sampling
    TASKS = [(par, sam, ch_ii) for ch_ii in range(sam['n_ch'])]
    with Pool(processes=sam['n_ch']) as pool:
        pool.map(paral_work_MCMC, TASKS)
    # paral_work_MCMC(TASKS[0]) # for debugging

    # evaluate samples
    flags = {'mean': 1, 'HDI': 0, 'CI': 0, 'ESS': 1, 'rhat' : 1}
    options = {'CI': 0.99}
    eval_samples.main(sam['sam_dir'], sam['n_ch'], sam['N_po']//sam['N_save'], flags, options)
    stats = utils.load(Path( sam['sam_dir'] / 'stats'))
    print(f'mean ESS: {np.mean(stats["ESS"])}/{sam["N_po"]}') # 996.3199835042014/5000
    print(f'max rhat: {np.max(stats["rhat"])}') # 1.0035367318446045

    # reference diagnostic over all chains
    h = utils.load(sam['sam_dir']/'ch0'/'h_ref')
    for ch_ii in range(1, sam['n_ch']):
        h += utils.load(sam['sam_dir']/('ch'+str(ch_ii))/'h_ref') 
    utils.save(sam['sam_dir']/'h_ref', 1/sam['n_ch']*h)

if __name__ == '__main__':
    main()