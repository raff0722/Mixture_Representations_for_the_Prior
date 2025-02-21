import numpy as np
from scipy.ndimage import gaussian_filter1d
import pywt

from Mixture_Representations_for_the_Prior.oneD_pw_const_wavelet_deblurr_large import wavelet_1D_class
from Mixture_Representations_for_the_Prior import utils

def main(par):

    # true signal
    x_sig = np.load(r'oneD_pw_const_wavelet_deblurr_large\oneD_pw_const_1024.npy')

    # Gaussian blur
    R = lambda x: gaussian_filter1d(input=x, sigma=par['blur_std'], mode=par['ext_mode_blur'], truncate=par['radius']/par['blur_std'])
    
    # true data
    y_true = R(x_sig)

    # data
    np.random.seed(par['noise_seed'])
    y = y_true + np.random.normal(loc=0, scale=par['noise_std'], size=x_sig.size)
    # noise_level = par['noise_std'] / np.linalg.norm(y_true) * np.sqrt( y_true.size )

    # non-zero wavelet coeffs of true signal
    coeffs = pywt.wavedecn(x_sig, par['wavelet'], par['ext_mode_wave'], par['levels'])
    x_coeff, coeff_slices, coeff_shapes = pywt.ravel_coeffs(coeffs)
    I_nz = np.argwhere(np.abs(x_coeff)>1e-16)[:,0]

    # compute rates of Laplace prior (Besov 1-1 prior)        
    d_coeff = x_coeff.size
    delta = np.zeros(d_coeff)
    delta_f = lambda l: 2**(1/2*l) # 2^(j* (s+d/2-d/p) ) where: j level (approx level scaled with 1),  d (dimension) = 1, p=s=1 
    delta[coeff_slices[0]] = delta_f(0)
    for i in range(1, len(coeff_slices)):
        delta[coeff_slices[i]['d']] = delta_f(i)
    delta *= par['delta_fac']

    # construct matrix from linear operators
    A = lambda x: R( pywt.waverecn(pywt.unravel_coeffs(x, coeff_slices, coeff_shapes, 'wavedecn'), par['wavelet'], par['ext_mode_wave']) )
    A_mat = np.zeros((x_sig.size, d_coeff))
    for ii in range(d_coeff):
        e = np.insert( np.zeros(d_coeff-1, dtype=np.double), ii, 1 )
        A_mat[:, ii] = A(e)

    # save
    utils.save(par['run_dir'] / 'problem', [x_sig, y_true, y, d_coeff, x_coeff, coeff_slices, coeff_shapes, delta, I_nz, A_mat])