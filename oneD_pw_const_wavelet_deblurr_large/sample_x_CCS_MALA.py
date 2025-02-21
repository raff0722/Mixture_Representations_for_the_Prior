# sample from the reduced posterior in x space with MALA

import numpy as np

from Mixture_Representations_for_the_Prior import utils, MALA

def main(par, sam, ch):
    
    _, _, y, _, _, _, _, delta, _, A_mat = utils.load( par['run_dir'] / 'problem' )
    lam = 1/par['noise_std']**2

    # log target and gradient
    def log_tar_grad(x):
        ymAx = y - A_mat[:,sam['I']]@x
        log_pdf = -lam/2 * np.linalg.norm(ymAx)**2  - np.linalg.norm(delta[sam['I']]*x, ord=1)
        log_grad =  lam * A_mat[:,sam['I']].T @ ymAx - delta[sam['I']] * x / np.sqrt( x**2 + sam['eps'] )
        return log_pdf, log_grad
    
    # adaptive step
    if sam['adapt_step']: c = MALA.prep_cus_step_adaption(sam['N_b'], 0.001*sam['h0'], plot=False)
    else: c = None

    # sample
    MALA.sample(log_tar_grad=log_tar_grad,
                  h0=sam['h0'],
                  x0=sam['x0'],
                  N=sam['N_po'],
                  N_save=sam['N_save'],
                  save_dir=sam['sam_dir'] / ('ch'+str(ch)),
                  seed=ch,
                  th=sam['th'],
                  N_b=sam['N_b'],
                  adapt_step=sam['adapt_step'],
                  c=c,
                  acc_int_disp=sam['acc_int_disp'],
                  logpdf=False)

    print('sampling finished!')