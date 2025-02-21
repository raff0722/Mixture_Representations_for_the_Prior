import numpy as np
import arviz as az
from pathlib import Path, PurePath
import sys
from shutil import rmtree
import pickle

def load(file):
    file = Path(file)
    file = open(file, 'rb')
    variable = pickle.load(file)
    file.close()
    return variable

def save(file, variable):
    file = Path(file)
    file = open(file, 'wb')
    pickle.dump(variable, file)
    file.close()

def main(sampling_dir, n_ch, n_parts, flags, options, prefix='p', remove_samples=False, name=False):
    """ Evaluates mean, HDI, CI, ess, rhat. Does not load all samples at once into memory.

    Required folder structure:
    
    sampling_dir
      ch0
        <prefix>_0.npy
        <prefix>_1.npy
        ...
        out
      ch1
        <prefix>_0.npy
        <prefix>_1.npy
        ...
        out
      ...
    
    out is a dict containing
    sam_per_part (i.e. how many p_i.npy), h, acc, time (all are floats)

    give HDI and CI in decimals, e.g., 0.95

    Args:
        sampling_dir (_type_): sampling directory
        n_ch (_type_): number of chains
        n_parts (_type_): number of parts per chain
        flags (_type_): {'mean': 1, 'HDI': 1, 'CI': 1, 'ESS': 1, 'rhat' : 1, 'autocorr' : 1}
        options (_type_): {'HDI': 0.95, 'CI': 0.95, 'lag' : 20}
    """    

    if flags['rhat'] and (n_ch == 1): sys.exit('Cannot compute rhat with only one chain.')

    # define partition of parameter vector
    max_N = 150**2 * 1000 * 5 # 900.0 Megabytes

    N_po = 0
    for ii in range(n_parts):
        if ii == 0: d = np.load( Path( PurePath( sampling_dir, 'ch0', prefix+'_'+str(ii)+'.npy' ))).shape[0] # dimension
        N_po += np.load( Path( PurePath( sampling_dir, 'ch0', prefix+'_'+str(ii)+'.npy' ))).shape[1]
    n_p = (d*N_po*n_ch) // max_N + 1 # number of partitions
    d_p = d // n_p # dimension of partitions
    d_slices = [np.s_[ii*d_p:ii*d_p+d_p] for ii in range(n_p)] # slices for partitioning
    d_lengths = [d_p]*n_p # list of dimensions of partitions
    if d != n_p*d_p: 
        d_slices.append(np.s_[n_p*d_p:d])
        d_lengths.append(d-n_p*d_p)
        n_p += 1

    # summary statistics
    stats = {}
    
    for ii in range(n_p):

        # load samples
        # samples structure of arviz is
        # chain - draw - dim 0 - dim 1 - dim 2 ...
        # coordinates in data object: 'chain', 'draw', 'x_dim_0'

        samples_ii = np.zeros((0, N_po, d_lengths[ii]))
        for jj in range(n_ch):

            samples_kk = np.zeros((0, d_lengths[ii]))
            for kk in range(n_parts):
                path = Path( PurePath( sampling_dir, 'ch'+str(jj), prefix+'_'+str(kk)+'.npy' ))
                samples_kk = np.concatenate( (samples_kk, np.swapaxes(np.load(path)[d_slices[ii], :], 0, 1)), axis=0 )

            samples_ii = np.concatenate( (samples_ii, samples_kk[None, :, :]), axis=0)
        
        # convert to inference data object to call arviz methods
        samples_ii = az.convert_to_inference_data(samples_ii)

        if flags['mean']:
            if ii == 0: stats['mean'] = np.zeros(d)
            stats['mean'][d_slices[ii]] = samples_ii.posterior.mean(dim=['chain','draw'])['x'].to_numpy()
            
        if flags['HDI']: # can be multimodal
            if ii == 0: stats['HDI'] = []
            stats['HDI'].append( (d_slices[ii], az.hdi(samples_ii, hdi_prob=options['HDI'], multimodal=options['HDI_multi_mod'])['x'].to_numpy()) )

        if flags['CI']: 
            if ii == 0: stats['CI'] = np.zeros((d, 2))
            stats['CI'][d_slices[ii], 0] = samples_ii.posterior.quantile(q=(1-options['CI'])/2, dim=['chain','draw'])['x'].to_numpy()
            stats['CI'][d_slices[ii], 1] = samples_ii.posterior.quantile(q=(1+options['CI'])/2, dim=['chain','draw'])['x'].to_numpy()

        if flags['ESS']:
            if ii == 0: stats['ESS'] = np.zeros((d, n_ch))
            for ll in range(n_ch):
                stats['ESS'][d_slices[ii], ll] = az.ess(samples_ii.sel(chain=[ll]))['x'].to_numpy()

        if flags['rhat']: 
            if ii == 0: stats['rhat'] = np.zeros(d)
            stats['rhat'][d_slices[ii]] = az.rhat( samples_ii )['x'].to_numpy()
            
        # if flags['autocorr']:
        #     sam_autocorr = np.squeeze(samples_ii.posterior.x.values)
        #     autocorr_ii = np.zeros_like((sam_autocorr.shape[1:]))
        #     for jj in range( autocorr_ii.size ):    
        #     autocorr_ii = az.autocorr( np.squeeze(samples_ii.posterior.x.values), axis=0)
        #     stat_ii.append({'autocorr':autocorr_ii})
        # stats.append( az.summary(data=samples_ii, fmt='xarray', round_to='none', hdi_prob=hdi_prob) )

        print(f'Partition {ii+1} / {n_p} done!', end='\r')

    # load sampling data if available
    stats['out'] = []
    for mm in range(n_ch):
        try:
            stats['out'].append( load( PurePath( sampling_dir, 'ch'+str(mm), 'out' ) ) )
        except:
            pass
            
    # save stats
    if name: save( PurePath( sampling_dir, name ), stats)
    else: save( PurePath( sampling_dir, 'stats' ), stats)

    # remove samples
    if remove_samples:
        for ii in range(n_ch):
            rmtree( PurePath( sampling_dir, 'ch'+str(ii) ))

def load_chain(sampling_dir_ch, n_parts):

    samples = np.load( Path( sampling_dir_ch / 'p_0.npy' ) )
    for ii in range(1, n_parts):
        samples = np.concatenate((samples, np.load(Path(sampling_dir_ch / ('p_'+str(ii)+'.npy')))), axis=-1)

    return samples