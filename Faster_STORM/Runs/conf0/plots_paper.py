import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path, PureWindowsPath

from Mixture_Representations_for_the_Prior import utils
from Mixture_Representations_for_the_Prior.Faster_STORM import FS_utils

run_dir = PureWindowsPath(r'Faster_STORM\Runs\conf0')
plots_dir = PureWindowsPath(r'C:\Users\raff\Projects\Mixture_Representations_for_the_Prior\Manuscript_7')

par = utils.load(run_dir / 'par')
[A_mat, y, lam, y_truth, x_im_truth, ind_mol, d2, m2] = utils.load(par['run_dir'] / 'problem')
map = utils.load(PureWindowsPath(r'Faster_STORM\Runs\conf0\map_x_cvxpy'))

# row and column indices of true molecule positions
row_true, col_true = np.unravel_index(ind_mol, shape=(par['d']*par['R'], par['d']*par['R']), order='F')

#%% Figure 5: Truth and Data

utils.inv_prob(grid=False, fig_width=15, fig_height=7)

fig, ax = plt.subplots(ncols=2)

# plot_data = np.log10(y)
plot_data = y
im = ax[0].imshow( FS_utils.unrav( plot_data, par['d']), cmap='gray')
divider = make_axes_locatable(ax[0])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')
ax[0].set_title(r'Data $y$')

# plot_truth = np.log10(x_im_truth)
plot_truth = x_im_truth
im = ax[1].imshow( plot_truth, cmap='gray')
divider = make_axes_locatable(ax[1])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')
ax[1].set_title(r'True image $x_\mathrm{true}$')

plt.tight_layout()
fig.savefig(Path( plots_dir / 'Faster_STORM_data-truth.pdf' ), dpi=1000)

#%% Figure 6: MAP-estimate

utils.inv_prob(grid=False, fig_width=8, fig_height=7)

fig, ax = plt.subplots()

# plot_map = np.log10(map)
# plot_map[plot_map==-np.inf] = 0
plot_map = map
im = ax.imshow( FS_utils.unrav(plot_map, par['d']*par['R']), cmap='gray') # vmin=np.min(x_im_truth), vmax=np.max(x_im_truth), 
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')
ax.scatter(col_true, row_true, s=20, marker='s', facecolor='none', edgecolor='red', alpha=0.7, linewidth=0.3, label=r'$x_{\mathrm{true},i}\neq0$', zorder=100)
ax.set_title(r'MAP estimate $x^\mathrm{MAP}$') 

plt.tight_layout()
fig.savefig(Path( plots_dir / 'Faster_STORM_MAP.pdf' ), dpi=1000)

#%% Figure 7: estimated upper bounds on Hellinger

utils.inv_prob(grid=True, fig_width=6.5, fig_height=7)

diagnos = [
    r'Faster_STORM\Runs\conf0\sample_w_MAP\sam0\h_w_MAP',
    r'Faster_STORM\Runs\conf0\sample_x_MAP\sam0\h_x_MAP'
]

label = [
    '$a=w^\mathrm{MAP}$',
    '$a=x^\mathrm{MAP}$'
    ]

ls = ['solid',
      'dashdot']

y_lim = (1e1, 1e5)
y_int = ( np.log(y_lim[1]) - np.log(y_lim[0]) ) / len(label)

fig, ax = plt.subplots()

for ii, dia in enumerate(diagnos):
    h = utils.load(dia)
    p = ax.semilogy(2*np.cumsum(np.sort(h)), label=label[ii], ls=ls[ii])
    
ax.set_ylim(y_lim)

fig.subplots_adjust(bottom=0.25)
ax.legend(bbox_to_anchor=(0.5, 0), loc="lower center", bbox_transform=fig.transFigure, ncol=2, framealpha=1)

ax.set_title(r'$2 \, \sum_{i \in \mathcal{J}} \, \tilde{h}_{i,a}$')
ax.set_xlabel(r'$|\mathcal{J}|$')

plt.tight_layout()
fig.savefig(Path(plots_dir, 'Faster_STORM_diagnostics.pdf'), dpi=1000)

#%% Figure 8: posterior mean and CI, x vs w, with markers for truth

stats_paths = [
    r'Faster_STORM\Runs\conf0\sample_x_MAP\sam0\stats',
    r'Faster_STORM\Runs\conf0\sample_x_via_w_MAP\sam0\stats'
]

utils.inv_prob(grid=False, fig_width=15, fig_height=15)

fig, ax = plt.subplots(ncols=2, nrows=2)

mean_min = 0 #0.4
mean_max = 8500 #3.6
CI_min = 0.5
CI_max = 5

for ii, path in enumerate(stats_paths):

    stats = utils.load(PureWindowsPath(path))
        
    # # log mean, MASKED
    # plot_mean_val = np.ma.masked_where(stats['mean']<=0, np.log10(stats['mean'])) # mask values <=0 -> pixels to plot
    # plot_mean_nval = np.ma.masked_where(stats['mean']>0, np.zeros(d2)) # mask values >0 -> one color
    # im = ax[0,ii].imshow( FS_utils.unrav( plot_mean_val, par['d']*par['R']), cmap='gray') #, vmin=mean_min, vmax=mean_max)
    # divider = make_axes_locatable(ax[0,ii])
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # fig.colorbar(im, cax=cax, orientation='vertical')
    # im = ax[0,ii].imshow( FS_utils.unrav( plot_mean_nval, par['d']*par['R']), vmin=0, vmax=1, cmap='gray')
    # ax[0,ii].scatter(col_true, row_true, s=2, alpha=1, c='r', marker='*', lw=0, label=r'$x_{\mathrm{true},i}\neq0$')
        
    # mean
    plot_mean = stats['mean']
    im = ax[0,ii].imshow( FS_utils.unrav( plot_mean, par['d']*par['R']), cmap='hot', vmin=mean_min, vmax=mean_max)
    divider = make_axes_locatable(ax[0,ii])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    # ax[0,ii].scatter(col_true, row_true, s=2, alpha=1, c='r', marker='*', lw=0, label=r'$x_{\mathrm{true},i}\neq0$')
    ax[0,ii].scatter(col_true, row_true, s=20, marker='s', facecolor='none', edgecolor='white', alpha=0.7, linewidth=0.3, label=r'$x_{\mathrm{true},i}\neq0$', zorder=100)
    
    # CI diff
    plot_CI = stats['CI'][:,1] - stats['CI'][:,0]
    plot_CI = np.log10(plot_CI)
    im = ax[1,ii].imshow( FS_utils.unrav( plot_CI , par['d']*par['R']), cmap='hot', vmin=CI_min, vmax=CI_max)
    divider = make_axes_locatable(ax[1,ii])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    # ax[1,ii].scatter(col_true, row_true, s=2, alpha=1, c='r', marker='*', lw=0)
    ax[1,ii].scatter(col_true, row_true, s=20, marker='s', facecolor='none', edgecolor='white', alpha=0.7, linewidth=0.3, label=r'$x_{\mathrm{true},i}\neq0$', zorder=100)

ax[0,0].set_xticks([])
ax[0,1].set_xticks([])
ax[0,1].set_yticks([])
ax[1,1].set_yticks([])

ax[0,0].set_title(r'MAP(X)')
ax[0,1].set_title(r'MAP(W)')

ax[0,0].set_ylabel(r'Mean')
ax[1,0].set_ylabel(r'99% CI difference ($\mathrm{log}_{10}$-scale)')

plt.tight_layout()
fig.savefig(Path( plots_dir / 'Faster_STORM_mean_CI.pdf' ), dpi=1000)

#%% Figure 9: posterior mean and CI, x vs w, sections

stats_paths = [
    r'Faster_STORM\Runs\conf0\sample_x_MAP\sam0\stats',
    r'Faster_STORM\Runs\conf0\sample_x_via_w_MAP\sam0\stats'
]

map_x = utils.load(r'Faster_STORM\Runs\conf0\map_x_cvxpy')
map_x_im = FS_utils.unrav(map_x, par['d']*par['R'])

utils.inv_prob(grid=True, fig_width=12, fig_height=12)

fig, ax = plt.subplots(ncols=1, nrows=3)

slice_x = [
    18, 
    87,
    np.arange(128),  
    ]
slice_y = [
    np.arange(128),
    np.arange(128),
    29
]
ylim = [
    (-4500, 11_000),
    (-1000, 12_000),
    (-5000, 8_000),
]
xlim = [
    (68, 80),
    (77, 82),
    (75, 80),
]

mean_min = 0 #0.4
mean_max = 8500 #3.6
CI_min = 0.5
CI_max = 5

for jj in range(len(slice_x)):

    # truth
    # im = ax[jj,2].imshow( x_im_truth, cmap='gray' )
    # if isinstance(slice_x[jj], int): ax[jj,2].axvline(slice_x[jj], color='yellow')
    # if isinstance(slice_y[jj], int): ax[jj,2].axhline(slice_y[jj], color='yellow')

    for ii, path in enumerate(stats_paths):

        stats = utils.load(PureWindowsPath(path))
        im_mean = FS_utils.unrav( stats['mean'], par['d']*par['R'])
        im_CI_l = FS_utils.unrav( stats['CI'][:,0], par['d']*par['R'])
        im_CI_u = FS_utils.unrav( stats['CI'][:,1], par['d']*par['R'])

        if jj == 0: # put label
            if ii == 0:
                p = ax[jj].plot(im_mean[slice_y[jj], slice_x[jj]], ls=':', color='tab:blue', lw=1.2, zorder=100,
                                label='sample mean MAP(X)')
                ax[jj].plot( x_im_truth[slice_y[jj], slice_x[jj]], color='tab:red', label='$x_\mathrm{true}$')
            if ii == 1:
                p = ax[jj].plot(im_mean[slice_y[jj], slice_x[jj]], ls='--', color='tab:green', lw=1.2, zorder=100,
                                label='sample mean MAP(W)')
            
        else: # no label
            if ii == 0:
                p = ax[jj].plot(im_mean[slice_y[jj], slice_x[jj]], ls=':', color='tab:blue', lw=1.2, zorder=100)
                ax[jj].plot( x_im_truth[slice_y[jj], slice_x[jj]], color='tab:red')
            if ii == 1:
                p = ax[jj].plot(im_mean[slice_y[jj], slice_x[jj]], ls='--', color='tab:green', lw=1.2, zorder=100)
            
        # ax[jj].plot(map_x_im[slice_y[jj], slice_x[jj]], color='tab:red', ls='-.')
        ax[jj].fill_between(np.arange(128), im_CI_l[slice_y[jj], slice_x[jj]], im_CI_u[slice_y[jj], slice_x[jj]], color=p[-1].get_color(), alpha=0.3, lw=0)

    ax[jj].set_ylim(ylim[jj])
    ax[jj].set_xlim(xlim[jj])
    ax[jj].set_axisbelow(True)

ax[0].set_title(r'$x_\mathrm{image}=18$')
ax[1].set_title(r'$x_\mathrm{image}=87$')
ax[2].set_title(r'$y_\mathrm{image}=28$')

fig.legend(bbox_to_anchor=(0.5, 0), loc="lower center", bbox_transform=fig.transFigure, ncol=3, framealpha=1)
plt.tight_layout(rect=[0, 0.1, 1, 1])
fig.savefig(Path( plots_dir / 'Faster_STORM_mean_CI_sect.pdf' ), dpi=1000)








plt.show()
