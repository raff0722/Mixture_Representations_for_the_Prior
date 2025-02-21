from Mixture_Representations_for_the_Prior import utils, log_pdf_mixing, MALA

# samples in full v space with MALA

def main(par, sam, ch):

    [x_sig, y_true, y, d_coeff, x_coeff, coeff_slices, coeff_shapes, delta, I_nz, A_mat] = utils.load( par['run_dir'] / 'problem' )

    lam = 1/par['noise_std']**2
    til_A = lam* A_mat.T@ A_mat
    til_y = lam* A_mat.T@ y 

    log_pdf_grad = lambda v: log_pdf_mixing.log_pdf_v(v, delta, til_A, til_y, pdf=True, grad=True, Hess=False)[:2]
    
    if sam['adapt_step']: c = MALA.prep_cus_step_adaption(sam['N_b'], 0.001*sam['h0'], plot=False)
    else: c = None

    # sample
    MALA.sample(
        log_tar_grad=log_pdf_grad,
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