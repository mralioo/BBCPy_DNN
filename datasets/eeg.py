# copy from the bbci toolbox the following files:   (1) load/eeg.py
import warnings

import numpy as np
import bbcpy
import h5py
import os


def dataset(fname):
    dataset_tmp = np.load(fname, allow_pickle=True)
    dataset = np.empty_like(bbcpy.datatypes.eeg.Data, shape=dataset_tmp.flatten().shape)
    for i, data in enumerate(np.nditer(dataset_tmp, ['refs_ok'])):
        data = dataset_tmp[i]
        dataset[i] = bbcpy.datatypes.eeg.Data(data['X'], data['fs'],
                                              bbcpy.datatypes.eeg.Marker(data['mrk_pos'], data['mrk_class'],
                                                                         data['mrk_className']),
                                              bbcpy.datatypes.eeg.Chans(data['clab'], data['mnt']))
    return np.reshape(dataset, dataset_tmp.shape)


def data(fname):
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
    _, file_extension = os.path.splitext(fname)
    if file_extension == '.mat':
        return bbci_mat(fname)
    data = np.load(fname, allow_pickle=True)
    if 'X' in data.keys():
        X = data['X']
    else:
        X = []
        warnings.warn('File without EEG data loaded.')
    fs = data['fs']
    clab = data['clab']
    clab = [x[0] for x in clab]
    mnt = data['mnt']
    if len(data.keys()) > 4:
        mrk_pos = data['mrk_pos']
        mrk_class = data['mrk_class']
        mrk_className = data['mrk_className']
        return bbcpy.datatypes.eeg.Data(X, fs, bbcpy.datatypes.eeg.Marker(mrk_pos, mrk_class, mrk_className, fs),
                                        bbcpy.datatypes.eeg.Chans(clab, mnt))
    else:
        return bbcpy.datatypes.eeg.Data(X, fs, chans=bbcpy.datatypes.eeg.Chans(clab, mnt))


def bbci_mat(fname, load_data=True):
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
    from os.path import exists
    if not exists(fname):
        raise FileNotFoundError(
            'File: ' + fname + ' not found. Please check correct name, path and existence.') from OSError
    try:
        import scipy.io as sio
        data = sio.loadmat(fname, uint16_codec='ascii')
        chankeys = [k for k in data.keys() if k[:2] == 'ch']
        channums = np.array([int(k[2:]) for k in chankeys])
        X = np.array([data[k] for k in chankeys])[channums - 1, :, 0] * data['dat']['resolution'][0, 0].T
        fs = data['nfo']['fs'][0, 0][0][0]
        clab = [x[0] for x in data['nfo']['clab'][0, 0][0, :]]
        mnt = np.array([data['mnt']['x'][0, 0], data['mnt']['y'][0, 0]])[:, :, 0].T
        if 'mrk' in data.keys():
            if 'time' in data['mrk'].dtype.names:
                mrk_pos = data['mrk']['time'][0, 0][0, :]
                mrk_fs = 1000  # time in ms
            else:
                mrk_pos = data['mrk']['pos'][0, 0][0, :]
                mrk_fs = fs
            mrk_class = np.where(data['mrk']['y'][0, 0].T)[1]
            mrk_className = np.array(data['mrk']['className'][0, 0][0, :])
    except NotImplementedError:  # v7.3
        try:
            import h5py
            data = h5py.File(fname, 'r')
            chankeys = [k for k in data.keys() if k[:2] == 'ch']
            channums = np.array([int(k[2:]) for k in chankeys])
            X = np.array([data[k] for k in chankeys])[channums - 1, 0, :] * data['dat']['resolution'][0, 0].T
            fs = data['nfo']['fs'][0, 0]
            clab = [''.join([chr(int(x)) for x in y]) for y in [data[ref] for ref in data['nfo']['clab'][:, 0]]]
            mnt = np.array([data['mnt']['x'][0, :], data['mnt']['y'][0, :]]).T
            if 'mrk' in data.keys():
                if 'time' in data['mrk'].keys():
                    mrk_pos = data['mrk']['time'][:, 0]
                    mrk_fs = 1000
                else:
                    mrk_pos = data['mrk']['pos'][:, 0]
                    mrk_fs = fs
                # mrk_class = data['mrk']['y']
                mrk_class = np.where(data['mrk']['y'])[1]
                mrk_className = np.array(
                    [''.join([chr(int(x)) for x in y]) for y in [data[ref] for ref in data['nfo']['className'][:, 0]]])
            data.close()
        except:
            ValueError('Error reading ' + fname + '...')
            return
    if not load_data:
        X = []
    if 'mrk_pos' in locals():
        return bbcpy.datatypes.eeg.Data(X, fs, bbcpy.datatypes.eeg.Marker(mrk_pos, mrk_class, mrk_className, mrk_fs),
                                        bbcpy.datatypes.eeg.Chans(clab, mnt))
    else:
        return bbcpy.datatypes.eeg.Data(X, fs, chans=bbcpy.datatypes.eeg.Chans(clab, mnt))


def epo(fname):
    raise NotImplementedError('Epo loading is not yet implemented, sorry.')
    return bbcpy.datatypes.eeg.Epo(data(fname), data.__initargs___())
