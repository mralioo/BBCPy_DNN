import numpy as np
import scipy as sp
import scipy.interpolate
from matplotlib import pyplot as plt


def load_data(fname):
    '''
    Usage:
        X, fs, clab, mnt = load_data(fname)
        X, fs, clab, mnt, mrk_pos, mrk_class, mrk_className = load_data(fname)
    Parameters:
        fname: name of the data file
    Returns:
        X:    a 2D array of multi-channel timeseries (channels x samples), unit [uV] 
        fs:   sampling frequency [Hz]
        clab: a 1D array of channel names  (channels)
        mnt:  a 2D array of channel coordinates (channels x 2)   
              The electrode montage "mnt" holds the information of the 
              2D projected positions of the channels, i.e. electrodes, 
              on the scalp - seem from the top with nose up.
        mrk_pos:   a 1D array of marker positions (in samples)
        mrk_class: a 1D array that assigns markers to classes (0, 1)
        mrk_className: a list that assigns class names to classes
    '''
    data = np.load(fname, allow_pickle=True)
    X = data['X']
    fs = data['fs']
    clab = data['clab']
    clab = [x[0] for x in clab]
    mnt = data['mnt']
    if len(data.keys())>4:
        mrk_pos = data['mrk_pos']
        mrk_class = data['mrk_class']
        mrk_className = data['mrk_className']
        return X, fs, clab, mnt, mrk_pos, mrk_class, mrk_className
    else:
        return X, fs, clab, mnt


def scalpmap(mnt, v, clim='minmax', cb_label=''): 
    '''
    Usage:
        scalpmap(mnt, v, clim='minmax', cb_label='')
    Parameters:
        mnt: a 2D array of channel coordinates (channels x 2)
        v:   a 1D vector (channels)
        clim: limits of color code, either
          'minmax' to use the minimum and maximum of the data
          'sym' to make limits symmetrical around zero, or
          a two element vector giving specific values
        cb_label: label for the colorbar
    '''    
    # interpolate between channels
    xi, yi = np.linspace(-1, 1, 100), np.linspace(-1, 1, 100)
    xi, yi = np.meshgrid(xi, yi)
    rbf = sp.interpolate.Rbf(mnt[:,0], mnt[:,1], v, function='linear')
    zi = rbf(xi, yi)
        
    # mask area outside of the scalp  
    a, b, n, r = 50, 50, 100, 50
    mask_y, mask_x = np.ogrid[-a:n-a, -b:n-b]
    mask = mask_x*mask_x + mask_y*mask_y >= r*r    
    zi[mask] = np.nan

    if clim=='minmax':
        vmin = v.min()
        vmax = v.max()
    elif clim=='sym':
        vmin = -np.absolute(v).max()
        vmax = np.absolute(v).max()
    else:
        vmin = clim[0]
        vmax = clim[1]
    
    plt.imshow(zi, vmin=vmin, vmax=vmax, origin='lower', extent=[-1, 1, -1, 1], cmap='jet')
    plt.colorbar(shrink=.5, label=cb_label)
    plt.scatter(mnt[:,0], mnt[:,1], c='k', marker='+', vmin=vmin, vmax=vmax)
    plt.axis('off')


def makeepochs(X, fs, mrk_pos, ival):
    '''
    Usage:
        makeepochs(X, fs, mrk_pos, ival)
    Parameters:
        X: 2D array of multi-channel timeseries (channels x samples) 
        fs: sampling frequency [Hz]
        mrk_pos: marker positions [sa]
        ival: a two element vector giving the time interval relative to markers (in ms)
    Returns:
        epo: a 3D array of segmented signals (samples x channels x epochs)
        epo_t: a 1D array of time points of epochs relative to marker (in ms)
    '''
    time = np.array([range(np.int(np.floor(ival[0]*fs/1000)), 
                           np.int(np.ceil(ival[1]*fs/1000))+1)])
    T = time.shape[1]
    nEvents = len(mrk_pos)
    nChans = X.shape[0]
    idx = (time.T+np.array([mrk_pos])).reshape(1, T*nEvents)    
    epo = X[:,idx].T.reshape(T, nEvents, nChans)
    epo = np.transpose(epo, (0,2,1))
    epo_t = np.linspace(ival[0], ival[1], T)
    return epo, epo_t


def baseline(epo, epo_t, ref_ival):
    '''
    Usage:
        epo = baseline(epo, epo_t, ref_ival)
    Parameters:
        epo: a 3D array of segmented signals, see makeepochs
        epo_t: a 1D array of time points of epochs relative to marker (in ms)
        ref_ival: a two element vector specifying the time interval for which the baseline is calculated [ms]
    '''
    idxref = (ref_ival[0] <= epo_t) & (epo_t <= ref_ival[1])
    eporef = np.mean(epo[idxref, :, :], axis=0, keepdims=True)
    epo = epo - eporef
    return epo

def proc_spatialFilter(cnt, clab, chan, neighbors='*'):
    '''
    Usage:
    cnt_sf = proc_spatialFilter(cnt, clab, chan, neighbors='*')
    Parameters:
    cnt:
    a 2D array of multi-channel timeseries (size: channels x␣
    , → samples),
    clab:
    a 1D array of channel names (size: channels)
    chan:
    channel of center location
    neighbors: labels of channels that are to be subtracted
    Returns:
    cnt_sf:
    timeseries of spatially filtered channel (size: 1 x samples)
    Examples:
    cnt_c4_bip = proc_spatialFilter(cnt, clab, 'C4', 'CP4')
    cnt_c4_lap = proc_spatialFilter(cnt, clab, 'C4',␣
    , → ['C2','C6','FC4','CP4'])
    cnt_c4_car = proc_spatialFilter(cnt, clab, 'C4', '*')
    '''
    cidx= clab.index(chan)
    if isinstance(neighbors, list):
        nidx = [clab.index(cc) for cc in neighbors]
    elif neighbors == '*':
        # Common Average Reference (CAR)
        nidx = range(len(clab))
    else:
        nidx = [clab.index(neighbors)]
        
    cnt_sf = cnt[[cidx],:] - np.mean(cnt[nidx,:], axis=0)
    
    return cnt_sf