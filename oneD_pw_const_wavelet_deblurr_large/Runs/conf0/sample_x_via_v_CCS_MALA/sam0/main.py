# sample x via w, where w is sampled from CCS-reduced mixing density
# in fact, we transform log(w)=v and use MALA to sample v

from pathlib import PureWindowsPath, Path

from Mixture_Representations_for_the_Prior import utils

sam_nr = '0'

# parameters
sam = {
    'sam_dir' : PureWindowsPath(r'oneD_pw_const_wavelet_deblurr_large\Runs\conf0\sample_x_via_v_CCS_MALA\sam'+sam_nr),
    'n_cores' : 2,
    'sam_dir_w_samples' : PureWindowsPath(r'oneD_pw_const_wavelet_deblurr_large\Runs\conf0\sample_v_CCS_MALA\sam'+sam_nr),
}
Path.mkdir(Path(sam['sam_dir']), parents=True, exist_ok=True)
utils.save(sam['sam_dir'] / 'par', sam)

# sample coefficients and evaluate samples in parallel
# -> run sampling script "main_sample_x_given_w_RTO.py" (parallel sampling) separately