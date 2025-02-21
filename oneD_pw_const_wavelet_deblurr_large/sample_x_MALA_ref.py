import numpy as np

from Mixture_Representations_for_the_Prior import utils, MALA

# samples in full x space with MALA with smoothed prior

def main(par, sam, ch):

    [x_sig, y_true, y, d_coeff, x_coeff, coeff_slices, coeff_shapes, delta, I_nz, A_mat] = utils.load( par['run_dir'] / 'problem' )
    lam = 1/par['noise_std']**2

    def log_tar_grad(x):
        y_m_Ax = y-A_mat@x
        log_tar = -lam/2 * np.linalg.norm(y_m_Ax)**2  - np.linalg.norm(delta*x, ord=1)        
        log_grad = lam * A_mat.T@y_m_Ax - delta * x / np.sqrt( x**2 + sam['eps'] ) # smoothed prior
        return log_tar, log_grad
    
    if sam['adapt_step']: c = MALA.prep_cus_step_adaption(sam['N_b'], 0.001*sam['h0'], plot=False)
    else: c = None

    # sample
    MALA.sample(
        log_tar_grad=log_tar_grad,
        h0 = sam['h0'],
        x0 = sam['x0'],
        N = sam['N_po'],
        N_save = sam['N_save'],
        save_dir=sam['sam_dir'] / ('ch'+str(ch)),
        seed = ch,
        th = sam['th'],
        N_b = sam['N_b'],
        adapt_step = sam['adapt_step'],
        c = c,
        acc_int_disp = sam['acc_int_disp'],
        logpdf=False
    )

    print('sampling finished!')
