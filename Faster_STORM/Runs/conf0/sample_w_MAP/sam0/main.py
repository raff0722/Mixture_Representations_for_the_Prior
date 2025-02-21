# sample w via exponentials at w_MAP,i = 0 and truncated normal for w_MAP,i \neq 0

from pathlib import PureWindowsPath, Path
import numpy as np

from Mixture_Representations_for_the_Prior import utils, log_pdf_mixing, eval_samples
from Mixture_Representations_for_the_Prior.Faster_STORM import sample_w_MAP

# parameters
sam = {
    'n_ch' : 1,
    'sam_dir' : PureWindowsPath(r'Faster_STORM\Runs\conf0\sample_w_MAP\sam0'),
    'N_po' : 10_000,
    'N_save' : 10_000,
    'w_MAP' : PureWindowsPath(r'Faster_STORM\Runs\conf0\w_MAP_BFGS_0'),
    'w' : True # wether samples are in w-space or v-space (v requires back transformation)
}

# load problem
par_dir = PureWindowsPath(r'Faster_STORM\Runs\conf0\par')
par = utils.load( Path( par_dir ))
A, y, lam, y_truth, x_im_truth, ind_mol, d2, m2 = utils.load(par['run_dir'] / 'problem')
delta = par['delta'] * np.ones(d2)

# select coordinates
w_MAP = utils.load( sam['w_MAP'] )[0].x
eps = 1e-16
sam['Ic'] = np.where(w_MAP<eps)[0]
sam['I'] = np.setdiff1d(np.arange(d2), sam['Ic'])

# save selected coordinate indices
Path.mkdir(Path(sam['sam_dir']), parents=True, exist_ok=True)
utils.save(sam['sam_dir'] / 'par', sam)

# sample
sample_w_MAP.main(par, sam)

# estimate diagnostic
til_A = lam * A.T@A
til_y = lam * A.T@y

for ch_ii in range(sam['n_ch']):
    print(f'chain {ch_ii}...')

    # estimate diagnostic
    w_I = eval_samples.load_chain(sam['sam_dir'] / ('ch'+str(ch_ii)), sam['N_po']//sam['N_save'])
    grad = lambda w: log_pdf_mixing.log_pdf_w(w, delta, til_A, til_y, pdf=False, grad=True, Hess=False)[1]
    h = 0
    for ii in range(sam['N_po']):
        print(f'nabla log pi(w|y) of sample {ii+1}/{sam["N_po"]}', end='\r')
        w_ii = np.zeros(d2)
        w_ii[sam['I']] = w_I[:, ii]
        w_ii[sam['Ic']] = utils.fast_exponential(delta[sam['Ic']]**2/2, 1).flatten()
        h += grad(w_ii)**2
    h *= 1/sam['N_po'] * 1/(delta**2/2)**2
    utils.save(Path(sam['sam_dir'] / ('ch'+str(ch_ii)) / 'h_w_MAP'), h)

# mean of diagnostics of chains
h = utils.load(sam['sam_dir'] / 'ch0' / 'h_w_MAP')
for ii in range(1, sam['n_ch']): h += utils.load(sam['sam_dir'] / ('ch'+str(ii)) / 'h_w_MAP')
h = 1/sam['n_ch'] * h
utils.save(Path(sam['sam_dir'] / 'h_w_MAP'), h)

# # estimate diagnostic based on prior samples
# w_pr = utils.fast_exponential(delta**2/2, sam['N_po'])
# h_pr = 0
# for ii in range(sam['Npo']):
#     print(f'nabla log pi(w|y) of sample {ii+1}/{sam["N_po"]}', end='\r')
#     h_pr += grad(w_pr[:,ii])**2
# h_pr *= 1/sam['Npo'] * 1/ (delta**2/2)**2
# utils.save(Path(sam['sam_dir'] / 'h_pr'), h_pr)