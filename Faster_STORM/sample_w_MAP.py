# sample w via exponentials at w_MAP,i = 0 and truncated normal for w_MAP,i \neq 0

from Mixture_Representations_for_the_Prior import utils, log_pdf_mixing
from Mixture_Representations_for_the_Prior.truncated_mvn_sampler.minimax_tilting_sampler import TruncatedMVN

from pathlib import Path
import numpy as np

def main(par, sam):

    # load optimization result
    w_MAP_res = utils.load(sam['w_MAP'])[0]

    # load problem
    # A, y, lam, y_truth, x_im_truth, ind_mol, d2, m2 = utils.load(Path( par['run_dir'] / 'problem' ))

    # # Hessian at w_MAP
    # lam = 1/par['noise_std']**2
    # til_A = lam * A_mat.T @ A_mat
    # til_y = lam * A_mat.T @ y
    # _, _, Hess_MAP = log_pdf_mixing.log_pdf_w(w_MAP, delta, til_A, til_y, pdf=False, grad=False, Hess=True )

    # # covariance of truncated multivariate Gaussian
    # S, V = np.linalg.eigh( Hess_MAP[np.ix_(sam['I'], sam['I'])] )
    # i_svd = ( np.abs(S) > 1e-8 )
    # cov = - V[:, i_svd] @ np.diag(1/S[i_svd]) @ V[:, i_svd].T

    # inverse BFGS-Hessian (guaranteed to be pd)
    cov = w_MAP_res.hess_inv

    # n_ch chains
    for ch_ii in range(sam['n_ch']):

        print(f'chain {ch_ii}...')
        Path.mkdir(Path(sam['sam_dir'] / ('ch'+str(ch_ii)) ), parents=True, exist_ok=True)
        
        # sample and save
        my_trMVN = TruncatedMVN(mu=w_MAP_res.x[sam['I']], cov=cov, lb=np.zeros(sam['I'].size), ub=np.ones(sam['I'].size)*np.inf)
        my_trMVN.random_state = np.random.RandomState(ch_ii) 
        for ii in range(sam['N_po']//sam['N_save']):
            w_I = my_trMVN.sample(sam['N_save'])
            np.save(Path(sam['sam_dir'] /  ('ch'+str(ch_ii)) / f'p_{ii}.npy'), w_I)