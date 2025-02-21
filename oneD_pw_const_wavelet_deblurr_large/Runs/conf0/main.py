from Mixture_Representations_for_the_Prior.oneD_pw_const_wavelet_deblurr_large import problem, MAP_w, MAP_x, MAP_v
from Mixture_Representations_for_the_Prior import utils

from pathlib import PureWindowsPath, Path
import numpy as np

## conf 0 ##

# parameters
par = {
    'run_dir' : PureWindowsPath( r'oneD_pw_const_wavelet_deblurr_large/Runs/conf0' ),    
    'noise_seed' : 0,
    'noise_std' : 3e-2,
    
    'wavelet' : 'haar',
    'levels' : 10, # max level for given signal (d=1024)
    'ext_mode_wave' : 'symmetric', # also known as 'half-sample symmetric': ... x_2 x_1 | x_1 x_2 x_3 ...

    'radius' : 12,
    'blur_std' : 6,
    'ext_mode_blur' : 'reflect', # same as ext_mode_wave

    'delta_fac' : 1
}

# save parameters
Path.mkdir(Path(par['run_dir']), parents=True, exist_ok=True)
utils.save(par['run_dir'] / 'par', par)

# create problem
problem.main(par)
[x_sig, y_true, y, d_coeff, x_coeff, coeff_slices, coeff_shapes, delta, I_nz, A_mat] = utils.load( par['run_dir'] / 'problem' )

# MAP of W|Y=y
w0 = np.zeros(d_coeff)
n = 1
res = MAP_w.main(par, 'L-BFGS-B', w0, n)
utils.save( par['run_dir'] / 'w_MAP_BFGS_0', res)

# MAP of X|Y=y
res = MAP_x.main(par)
utils.save(par['run_dir'] / 'x_MAP', res)

# MAP of V|Y=y
w_mean = 1/(2/delta**2)
res = MAP_v.main(par, 'L-BFGS-B', np.log(w_mean), n)
utils.save( par['run_dir'] / 'v_MAP_L-BFGS-B', res)