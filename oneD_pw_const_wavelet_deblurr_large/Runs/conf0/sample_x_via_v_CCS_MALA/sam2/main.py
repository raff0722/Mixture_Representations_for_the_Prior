from pathlib import PureWindowsPath, Path
import numpy as np

from My_modules import pickle_routines
from My_modules.Sampling import eval_samples
from . import utils
from oneD_pw_const_wavelet_deblurr_large.sample_x_via_Gauss_mix import sample

par_dir = PureWindowsPath(r'oneD_pw_const_wavelet_deblurr_large\Runs\conf0\par')
par = pickle_routines.load( Path( par_dir ))

sam_nr = '2'

# sample x via w
sam = {
    'sam_dir' : PureWindowsPath(r'oneD_pw_const_wavelet_deblurr_large\Runs\conf0\sample_x_via_dim_red_w', 'sam'+sam_nr),
    'seed' : 0,
    'w_samples' : PureWindowsPath(r'oneD_pw_const_wavelet_deblurr_large\Runs\conf0\sample_dim_red_w', 'sam'+sam_nr, r'samples_w_full\ch0\p_0.npy'),
}
sam['N_po'] = np.load(sam['w_samples']).shape[1]
Path.mkdir(Path(sam['sam_dir']), parents=True, exist_ok=True)
pickle_routines.save_par(Path(sam['sam_dir']), sam)

## sample one-one in parllel
# run sampling script (parallel sampling) separately # #

# # ## sample one-one sequentially
# print('sample one x for one w...')
# x = sample(par, sam)
# np.save(Path(sam['sam_dir'] / 'ch0' / 'p_0.npy'), x)

# # evaluate samples
# flags = {'mean': 1, 'HDI': 0, 'CI': 1, 'ESS': 1, 'rhat' : 0}
# options = {'CI': 0.99}
# eval_samples.main(sam['sam_dir'], 1, 1, flags, options)
# stats = pickle_routines.load(sam['sam_dir']/'stats')
# print(f'nESS, ch0: {np.mean(stats[0]["ESS"])/sam["N_po"]}') # 0.92

# # transform samples to signal space
# x = np.load(Path(sam['sam_dir'] / 'ch0' / 'p_0.npy'))
# print('transform x to signal space...')
# s = functions.trans2signal(par, x )
# P_s = PureWindowsPath(sam['sam_dir'], 'samples_signal', 'ch0')
# Path(P_s).mkdir(parents=True, exist_ok=True)
# np.save(Path(P_s / 'p_0.npy'), s)

# evaluate samples in signal space
flags = {'mean': 1, 'HDI': 0, 'CI': 1, 'ESS': 0, 'rhat' : 0}
options = {'CI': 0.60}
eval_samples.main(sam['sam_dir']/'samples_signal', 1, 1, flags, options)