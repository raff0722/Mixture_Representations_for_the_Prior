# Mixture_Representations_for_the_Prior
Code of the paper "Continuous Gaussian mixture solution for linear Bayesian inversion with application to Laplace priors"

## Instructions
The used example and sampling configurations of the paper are in the folder "conf0" which can be found in both "Faster_STORM/Runs" and "oneD_pw_const_wavelet_deblurr_large/Runs". To set up the problems, first run "main.py" in "conf0". This script also contains the specific paramters of the problems. To sample via one of the methods presented in the paper, go to one of the "sample..." folders in "conf0". The "sam" subfolders (e.g., "sam0") contain different sampling configurations. To sample, run the "main.py" script therein. The sampling data is not stored in this repo to save storage. The plots in the paper can be recreated with the "plots_paper.py" script in the "conf0" folders. (Before doing so, all the sampling tasks have to be done.)
