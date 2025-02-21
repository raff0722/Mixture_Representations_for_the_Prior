import os
from pathlib import PureWindowsPath, Path
import numpy as np

from Faster_STORM import FS_utils, problem, MAP_w
from Mixture_Representations_for_the_Prior import utils

conf_nr = '0'

# 4x32 = 128 pixels side length

par = {
    'N' : 50, # number of molecules
    'd' : 32, # pixels per side
    'd_m' : 24, # pixels per side of area containing molecules
    'R' : 4, # ratio super-res grid size / pixel grid size
    'mode' : 3000, # mode of lognormal distribution modelling photon count of molecules

    'ph_back' : 70, # background photon noise
    'noise_std' : 30, # Gaussian additive noise
    
    'pad_width' : 1, # of boundary conditions in pixels
    'ext_mode' : 'periodic', #'zero', # boundary condition for super res image
    'lognorm_std' : .417,

    'ind_mol_seed' : 0, # seed for random determination of molecule indices
    'N_pho_seed' : 1, # seed for number of photons
    'noise_seed' : 2, # seed for photon noise

    'delta' : 1.275, # Laplace prior rate parameter

    'paper_data_file' : PureWindowsPath(r'Faster_STORM/100_molecules.tif'), # data from original paper
    'A_mat_file' : PureWindowsPath(r'Faster_STORM/Runs', 'conf'+conf_nr, 'A_4x32_cent.mat'), # forward matrix, exported from MATLAB script from original paper
    'A_file' : PureWindowsPath(r'Faster_STORM/Runs', 'conf'+conf_nr, 'A_4x32_cent'), # saved numpy array of forward matrix

    'run_dir' : PureWindowsPath(r'Faster_STORM/Runs', 'conf'+conf_nr)
}
os.makedirs(par['run_dir'], exist_ok=True)
utils.save(Path(par['run_dir'], 'par'), par)

# create blur matrix
FS_utils.blur_matrix(par)

# create problem data
problem.main(par)

# MAP x
map_cvxpy = FS_utils.map_cvxpy(par)
utils.save( par['run_dir'] / 'map_x_cvxpy', map_cvxpy)

# MAP w
_, _, _, _, _, _, d2, _ = utils.load(par['run_dir']/'problem')
w0 = np.zeros(d2)
res = MAP_w.main(par, 'L-BFGS-B', w0, 1)
utils.save(par['run_dir']/'w_MAP_BFGS_0', res)