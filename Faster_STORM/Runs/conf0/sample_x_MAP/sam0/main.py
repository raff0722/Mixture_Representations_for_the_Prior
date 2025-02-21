# sample from Laplace approximation in x via RTO

from pathlib import PureWindowsPath, Path, PurePosixPath, PurePath
import numpy as np
import sys
import os
from multiprocessing import Pool
from scipy.sparse import dia_array
from time import time

from Mixture_Representations_for_the_Prior import utils, eval_samples

sam_nr = '0'

# parameters
sam = {
    'n_proc' : 10,
    'sam_dir' : PureWindowsPath(r'Faster_STORM\Runs\conf0\sample_x_MAP\sam'+sam_nr),
    'seed' : 0,
    'N_po' : 5_000,   
    'N_save' : 1_000,
    'map' : PureWindowsPath(r'Faster_STORM\Runs\conf0\map_x_cvxpy'),
}
utils.save(Path(sam['sam_dir']) / 'par', sam)

# sample x -> run "sample_x_MAP.py" for parallel RTO sampling

# evaluate samples
print('Evaluate samples...')
flags = {'mean': 1, 'HDI': 0, 'CI': 1, 'ESS': 0, 'rhat' : 0}
options = {'CI': 0.99}
eval_samples.main(sam['sam_dir']/('sam'+sam_nr), 1, sam['N_po']//sam['N_save'], flags, options)

# load problem data
par = utils.load(PureWindowsPath(r'Faster_STORM\Runs\conf0\sample_x_MAP\sam0\par'))
[A_mat, y, lam, y_truth, x_im_truth, ind_mol, d2, m2] = utils.load(Path( par['run_dir'] / 'problem' ))
delta = par['delta']*np.ones(d2)

# estimate diagnostic
grad = lambda x: lam * A_mat.T @ (y-A_mat@x) 
x_samples = eval_samples.load_chain(sam['sam_dir']/('sam'+sam_nr), 1)
h = 0
for ii in range(sam['N_po']): h += grad(x_samples[:,ii])**2
h *= 1/sam['N_po'] * 1/delta**2
utils.save(Path(sam['sam_dir']/('sam'+sam_nr)/'h_x_MAP'), h)