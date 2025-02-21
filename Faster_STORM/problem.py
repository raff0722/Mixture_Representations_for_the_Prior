import numpy as np
from scipy.stats import lognorm, norm
from pathlib import Path

from Mixture_Representations_for_the_Prior import utils
from Faster_STORM.FS_utils import rav, unrav

# debug runtime warnings vsc python
# import warnings
# warnings.filterwarnings("error")

def main(par):

    d2 = (par['d']*par['R'])**2 # total number of super res pixels
    m2 = par['d']**2 # total number of data pixels

    lognorm_mean = np.log(par['mode']) + par['lognorm_std']**2
    # print(f'lognormal variance: {lognorm.var(loc=0, s=par["lognorm_std"], scale=np.exp(lognorm_mean))}, should be {1700**2}')

    # true super resolution image
    x_im_truth = np.zeros((par['d_m']*par['R'], par['d_m']*par['R']))

    # simulate photon counts
    np.random.seed(par['ind_mol_seed'])
    ind_mol_crop = np.random.choice(np.arange(x_im_truth.size), size=par['N'], replace=False)
    np.random.seed(par['N_pho_seed'])
    N_pho = lognorm.rvs(loc=0, s=par['lognorm_std'], scale=np.exp(lognorm_mean), size=par['N'])

    # create true image
    x_im_truth[np.unravel_index(ind_mol_crop, shape=(x_im_truth.shape[0], x_im_truth.shape[1]), order='F')] = N_pho
    x_im_truth = np.pad(x_im_truth, (par['d']-par['d_m'])//2*par['R'], mode='constant', constant_values=0) # no molecules in this area
    ind_mol = np.nonzero( rav( (x_im_truth > np.zeros((par['d']*par['R'], par['d']*par['R']))) ) )[0]

    # add background
    x_im_truth += par['ph_back']

    # forward operator
    A = utils.load(par['A_file'])

    # data and noise
    y_truth = A @ rav(x_im_truth)
    np.random.seed(par['noise_seed'])
    N_pho_noise = norm.rvs(scale=par['noise_std'], size=par['d']**2) # just some Gaussian background noise
    y = y_truth + N_pho_noise
    lam = 1/par['noise_std']**2
    print(f"SNR={np.linalg.norm(y)/np.sqrt(y.size)/par['noise_std']}")

    utils.save( Path( par['run_dir'] / 'problem'), [A, y, lam, y_truth, x_im_truth, ind_mol, d2, m2])