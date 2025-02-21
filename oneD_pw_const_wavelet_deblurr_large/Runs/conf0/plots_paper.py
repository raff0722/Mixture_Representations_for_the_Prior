# all plots in 
# Flock, Rafael, Yiqiu Dong, Felipe Uribe, and Olivier Zahm. 
# “Continuous Gaussian Mixture Solution for Linear Bayesian Inversion with Application to Laplace Priors.” 
# arXiv, 2024. http://arxiv.org/abs/2408.16594.

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path, PureWindowsPath
from scipy.special import logsumexp

from Mixture_Representations_for_the_Prior import utils

run_dir = PureWindowsPath(r'oneD_pw_const_wavelet_deblurr_large/Runs/conf0')
plots_dir = PureWindowsPath(r'C:\Users\flock\Projects\Mixture_Representations_for_the_Prior\plots_test')

par = utils.load( Path(run_dir / 'par'))
x_sig, y_true, y, d_coeff, x_coeff, coeff_slices, coeff_shapes, delta, I_nz, A_mat = utils.load( Path( run_dir / 'problem'))
lam = 1/par['noise_std']**2

#%% Figure 1: problem

utils.inv_prob(fig_height=6)

fig, ax = plt.subplots()
# plt.plot(y_true, label='y true')
ax.plot(y, label=r'$y$')
ax.plot(x_sig, label=r'$s_\mathrm{true}$', ls='dashed', lw=1)
ax.legend(loc='upper left')
ax.set_axisbelow(True)
plt.title('True signal $s_\mathrm{true}$ and data $y$')
plt.tight_layout()
plt.savefig( Path(plots_dir, 'oneD_pwconst_debl_problem.pdf'), dpi=1000)

#%% Figure 2: true coefficients and rate parameters of Laplace prior

utils.inv_prob(fig_width=8, fig_height=6)

fig, ax = plt.subplots()

ax.plot(I_nz, x_coeff[I_nz], lw=0, marker='.', zorder=100)
ax.set_ylabel(r'•   $|x_\mathrm{true}|>0$')
ax.set_axisbelow(1)

ax2 = ax.twinx()
ax2.plot(delta, color='tab:orange', zorder=99)
ax2.set_ylabel(r'—   $\delta_i$')
ax2.set_axisbelow(1)
ax2.set_title('True coefficients $x_\mathrm{true}$')

plt.tight_layout()
plt.savefig(Path(plots_dir, 'oneD_pwconst_debl_delta.pdf'), dpi=1000)

#%% Figure 3: posterior dim-red (MAP/CCS) vs reference 

utils.inv_prob(fig_width=15, fig_height=14)
CI = ['60', '90' ]

fig, ax = plt.subplots(ncols=2, nrows=2)

solution_paths = [PureWindowsPath(r'oneD_pw_const_wavelet_deblurr_large\Runs\conf0\sample_x_via_w_MAP\sam0\samples_s'),
                  PureWindowsPath(r'oneD_pw_const_wavelet_deblurr_large\Runs\conf0\sample_x_via_v_CCS_MALA\sam0\samples_signal'),
                  PureWindowsPath(r'oneD_pw_const_wavelet_deblurr_large\Runs\conf0\sample_x_via_v_CCS_MALA\sam1\samples_signal'),
                  PureWindowsPath(r'oneD_pw_const_wavelet_deblurr_large\Runs\conf0\sample_x_via_v_CCS_MALA\sam2\samples_signal')]

titles = [r'(a) $\tilde\pi_r(w|y) = \tilde\pi_r^\mathrm{MAP}(w|y)$',
          r'(b) $\tilde\pi_r(w|y) = \tilde\pi_{r=50}^\mathrm{CCS}(w|y)$',
          r'(c) $\tilde\pi_r(w|y) = \tilde\pi_{r=100}^\mathrm{CCS}(w|y)$',
          r'(d) $\tilde\pi_r(w|y) = \tilde\pi_{r=200}^\mathrm{CCS}(w|y)$']

ax = ax.flatten()
for ii, sol_path_ii in enumerate(solution_paths):

    # ax[ii].plot(x_sig, lw=1, alpha=0.5, color='black')

    # reference
    for CI_ii in CI:
        stats = utils.load(PureWindowsPath(r'oneD_pw_const_wavelet_deblurr_large\Runs\conf0\sample_x_MALA_ref\sam0\samples_signal', ('stats_'+CI_ii)))
        ax[ii].plot(stats['CI'][:,0], color='tab:blue')
        ax[ii].plot(stats['CI'][:,1], color='tab:blue')
    ax[ii].plot(stats['mean'], color='tab:blue', label=r'reference $\pi(x|y)$', lw=1.5)

    # solution
    for CI_ii in CI:
        stats = utils.load(PureWindowsPath(sol_path_ii, ('stats_'+CI_ii)))
        ax[ii].plot(stats['CI'][:,0], color='tab:orange', ls='dotted')
        ax[ii].plot(stats['CI'][:,1], color='tab:orange', ls='dotted')
    ax[ii].plot(stats['mean'], color='tab:orange', label=r'approximate solution $\int \; \pi(x|w,y) \tilde  \pi_r(w|y) \mathrm{d}w$', ls='dotted', lw=1.5)

    ax[ii].set_title(titles[ii])
    ax[ii].set_xlim([600, 780])
    ax[ii].set_ylim([0.55, 2.2])
    ax[ii].set_axisbelow(True)
    
# ax[0].legend(loc='lower center', framealpha=1)
ax[0].set_xticks(np.arange(600, 800, 20), [])
ax[1].set_xticks(np.arange(600, 800, 20), [])
ax[1].set_yticks(np.arange(0.6,2.4,0.2), [])
ax[3].set_yticks(np.arange(0.6,2.4,0.2), [])

plt.subplots_adjust(bottom=0.15)
plt.legend(bbox_to_anchor=(0.5, 0), loc="lower center", bbox_transform=fig.transFigure, ncol=2, framealpha=1)

plt.savefig( Path(plots_dir, 'oneD_pwconst_debl_post_approx.pdf'), dpi=1000)

#%% Figure 4: upper bound on Hellinger

utils.inv_prob(fig_width=9, fig_height=7)
fig, ax = plt.subplots()

# y_lim = (1e-7, 1e7)
# y_int = ( np.log(y_lim[1]) - np.log(y_lim[0]) ) / 4

h_MAP = utils.load(r'oneD_pw_const_wavelet_deblurr_large\Runs\conf0\sample_v_MAP\sam0\ch0\h_w_MAP')
h_ref = utils.load(r'oneD_pw_const_wavelet_deblurr_large\Runs\conf0\sample_v_MALA_ref\sam0\h_ref')
h_ref_x = utils.load(r'oneD_pw_const_wavelet_deblurr_large\Runs\conf0\sample_MALA_x_ref\sam0\h_ref')

bounds = np.zeros((4, d_coeff))
i_MAP = np.argsort(h_MAP)
bounds[0, :] = 2*np.cumsum(h_MAP[i_MAP])
i_ref = np.argsort(h_ref)
bounds[2, :] = 2*np.cumsum(h_ref[i_ref])
i_ref_x = np.argsort(h_ref_x)
bounds[3, :] = 2*np.cumsum(h_ref_x[i_ref_x])

# ref w
p1 = ax.semilogy( np.arange(1,d_coeff+1), bounds[2, :], label=r'$a=\mathrm{ref,w}$', ls='solid' )

# ref x
p1 = ax.semilogy( np.arange(1,d_coeff+1), bounds[3, :], label=r'$a=\mathrm{ref,x}$', ls='dashdot' )

# MAP w
p2 = ax.semilogy( np.arange(1,d_coeff+1), bounds[0, :], label=r'$a=w^\mathrm{MAP}$', ls='dashed' )

ax.set_axisbelow(1)
# ax.set_ylim(y_lim[0], y_lim[1])

ax.legend()

ax.set_title(r'$2 \, \sum_{i \in \mathcal{J}} \, \tilde{h}_{i,a}$')
# ax.set_xticks([0, 200, 400, 600, 800, 1000], [0, 200, 400, 600, 800, 1000])
ax.set_xlabel(r'$|\mathcal{J}|$')

plt.tight_layout()
plt.savefig(Path(plots_dir, 'oneD_pwconst_debl_Hell.pdf'), dpi=1000)


plt.show()