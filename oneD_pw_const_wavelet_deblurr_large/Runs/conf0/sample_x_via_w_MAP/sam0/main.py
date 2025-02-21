# sample x via w, where w is sampled from a truncated Gaussian and exponentials at the MAP

from pathlib import PureWindowsPath, Path

from Mixture_Representations_for_the_Prior import utils

# parameters
sam = {
    'sam_dir' : PureWindowsPath(r'oneD_pw_const_wavelet_deblurr_large\Runs\conf0\sample_x_via_w_MAP\sam0'),
    'n_cores' : 2,
    'sam_dir_w_samples' : PureWindowsPath(r'oneD_pw_const_wavelet_deblurr_large\Runs\conf0\sample_w_MAP\sam0'),
}
Path.mkdir(Path(sam['sam_dir']), parents=True, exist_ok=True)
utils.save(sam['sam_dir'] / 'par', sam)

# sample coefficients and evaluate samples in parallel
# -> run sampling script "main_sample_x_given_w_RTO.py" (parallel sampling) separately

