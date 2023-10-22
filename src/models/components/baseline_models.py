import numpy as np
import scipy as sp
import scipy.interpolate
import scipy.signal
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
    if len(data.keys()) > 4:
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
    rbf = sp.interpolate.Rbf(mnt[:, 0], mnt[:, 1], v, function='linear')
    zi = rbf(xi, yi)

    # mask area outside of the scalp
    a, b, n, r = 50, 50, 100, 50
    mask_y, mask_x = np.ogrid[-a:n - a, -b:n - b]
    mask = mask_x * mask_x + mask_y * mask_y >= r * r
    zi[mask] = np.nan

    if clim == 'minmax':
        vmin = v.min()
        vmax = v.max()
    elif clim == 'sym':
        vmin = -np.absolute(v).max()
        vmax = np.absolute(v).max()
    else:
        vmin = clim[0]
        vmax = clim[1]

    plt.imshow(zi, vmin=vmin, vmax=vmax, origin='lower', extent=[-1, 1, -1, 1], cmap='jet')
    plt.colorbar(shrink=.5, label=cb_label)
    plt.scatter(mnt[:, 0], mnt[:, 1], c='k', marker='+', vmin=vmin, vmax=vmax)
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
    time = np.array([range(np.int(np.floor(ival[0] * fs / 1000)),
                           np.int(np.ceil(ival[1] * fs / 1000)) + 1)])
    T = time.shape[1]
    nEvents = len(mrk_pos)
    nChans = X.shape[0]
    idx = (time.T + np.array([mrk_pos])).reshape(1, T * nEvents)
    epo = X[:, idx].T.reshape(T, nEvents, nChans)
    epo = np.transpose(epo, (0, 2, 1))
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


def crossvalidation(classifier_fcn, X, y, folds=10, verbose=False, feature_extraction=None):
    '''
    Synopsis:
        loss_te, loss_tr= crossvalidation(classifier_fcn, X, y, folds=10, verbose=False)
    Arguments:
        classifier_fcn: handle to function that trains classifier as output w, b
        X:              data matrix (features X epochs) or epochs (samples X channels x epochs)
                                with feature_extraction producing a data matrix (features X epochs)
        y:              labels with values 0 and 1 (1 x epochs)
        folds:         number of folds
        verbose:        print validation results or not
        feature_extraction: a function producing a data matrix (features X epochs) out of epoched data
    Output:
        loss_te: value of loss function averaged across test data
        loss_tr: value of loss function averaged across training data
    '''
    if len(X.shape) == 1:
        nSamples = X.shape[0]
        nCh, nT = X[0].shape
    if len(X.shape) == 2:
        nDim, nSamples = X.shape
        if feature_extraction == None:
            feature_extraction = lambda X, Y, Z: X
    elif len(X.shape) == 3:
        nT, nCh, nSamples = X.shape
        if feature_extraction == None:
            raise TypeError('For epoched data X, a feature extraction function has to be defined!')

    inter = np.round(np.linspace(0, nSamples, num=folds + 1)).astype(int)
    perm = np.random.permutation(nSamples)
    errTr = np.zeros([folds, 1])
    errTe = np.zeros([folds, 1])

    for ff in range(folds):
        idxTe = perm[inter[ff]:inter[ff + 1] + 1]
        idxTr = np.setdiff1d(range(nSamples), idxTe)
        fv = feature_extraction(X, y, idxTr)
        w, b = classifier_fcn(fv[:, idxTr], y[idxTr])
        out = w.T.dot(fv) - b
        errTe[ff] = loss_weighted_error(out[idxTe], y[idxTe])
        errTr[ff] = loss_weighted_error(out[idxTr], y[idxTr])

    if verbose:
        print('{:5.1f} +/-{:4.1f}  (training:{:5.1f} +/-{:4.1f})  [using {}]'.format(errTe.mean(), errTe.std(),
                                                                                     errTr.mean(), errTr.std(),
                                                                                     classifier_fcn.__name__))
    return np.mean(errTe), np.mean(errTr)


def loss_weighted_error(out, y):
    '''
    Synopsis:
        loss= loss_weighted_error( out, y )
    Arguments:
        out:  output of the classifier
        y:    true class labels
    Output:
        loss: weighted error
    '''
    loss = 50 * (np.mean(out[y == 0] >= 0) + np.mean(out[y == 1] < 0))
    return loss


def cov_shrink(X):
    '''
    Estimate covariance of given data using shrinkage estimator.

    Synopsis:
        C= cov_shrink(X)
    Argument:
        X: data matrix (features x samples)
    Output:
        C: estimated covariance matrix
    '''
    Xc = X - np.mean(X, axis=1, keepdims=True)
    d, n = Xc.shape
    Cemp = Xc.dot(Xc.T) / (n - 1)

    sumVarCij = 0
    for ii in range(d):
        for jj in range(d):
            varCij = np.var(Xc[ii, :] * Xc[jj, :])
            sumVarCij += varCij

    nu = np.mean(np.diag(Cemp))
    gamma = n / (n - 1.) ** 2 * sumVarCij / sum(sum((Cemp - nu * np.eye(d, d)) ** 2))
    S = gamma * nu * np.eye(d, d) + (1 - gamma) * Cemp
    return S


def train_LDA(X, y, useshrink=1):
    '''
    Synopsis:
        w, b= train_LDA(X, y)
    Arguments:
        X: data matrix (features X samples)
        y: labels with values 0 and 1 (1 x samples)
        useshrink: use shrinkage (default=1)
    Output:
        w: LDA weight vector
        b: bias term
    '''
    mu1 = np.mean(X[:, y == 0], axis=1)
    mu2 = np.mean(X[:, y == 1], axis=1)
    # pool centered features to estimate covariance on samples of both classes at once
    Xpool = np.concatenate((X[:, y == 0] - mu1[:, np.newaxis], X[:, y == 1] - mu2[:, np.newaxis]), axis=1)
    if useshrink:
        C = cov_shrink(Xpool)
    else:
        C = np.cov(Xpool)

    w = np.linalg.pinv(C).dot(mu2 - mu1)
    b = w.T.dot((mu1 + mu2) / 2.)
    return w, b


def train_PCA(X):
    '''
    Synopsis:
        d, V= train_PCA(X)
    Arguments:
        X: data matrix (features X samples)
    Output:
        d: Eigenvalues (explained variance)
        V: Filter matrix
    '''
    C = np.cov(X)
    d, V = np.linalg.eigh(C)
    return d, V


def train_CSP(epo, mrk_class):
    '''
    Usage:
        W, d = trainCSP(epo, mrk_class)
    Parameters:
        epo:   a 3D array of segmented signals (samples x channels x epochs)
        mrk_class: a 1D array that assigns markers to classes (0, 1)
    Returns:
        W:     matrix of spatial filters
        d:     vector of generalized Eigenvalues
    '''
    C = epo.shape[1]
    X1 = np.reshape(np.transpose(epo[:, :, mrk_class == 0], (1, 0, 2)), (C, -1))
    S1 = np.cov(X1)
    X2 = np.reshape(np.transpose(epo[:, :, mrk_class == 1], (1, 0, 2)), (C, -1))
    S2 = np.cov(X2)
    d, W = sp.linalg.eigh(a=S1, b=S1 + S2)
    return W, d


def train_SSD(cnt, center_band, flank1_band, flank2_band, filterorder, fs):
    ''' Usage: W, A, D, Y = train_SSD(cnt, center_band, flank1_band, flank2_band, filterorder, fs) '''

    b_c, a_c = sp.signal.butter(filterorder, np.array(center_band) * 2 / fs, btype='bandpass')
    cnt_filt_c = sp.signal.lfilter(b_c, a_c, cnt, axis=1)
    S1 = np.cov(cnt_filt_c)
    del cnt_filt_c
    b_f1, a_f1 = sp.signal.butter(filterorder, np.array(flank1_band) * 2 / fs, btype='bandpass')
    cnt_filt_f1 = sp.signal.lfilter(b_f1, a_f1, cnt, axis=1)
    S_f1 = np.cov(cnt_filt_f1)
    del cnt_filt_f1
    b_f2, a_f2 = sp.signal.butter(filterorder, np.array(flank2_band) * 2 / fs, btype='bandpass')
    cnt_filt_f2 = sp.signal.lfilter(b_f2, a_f2, cnt, axis=1)
    S_f2 = np.cov(cnt_filt_f2)
    del cnt_filt_f2

    S2 = (S_f1 + S_f2) / 2

    D, W = sp.linalg.eigh(a=S1, b=S2)
    Y = np.dot(W.T, cnt)
    S_x = np.cov(cnt)
    S_y = np.cov(Y)
    A = np.dot(np.dot(S_x, W), np.linalg.inv(S_y))
    D = np.diag(D)

    return W, A, D, Y


def train_TDSEP(cnt, tau=[0, 1]):
    ''' Usage: W, A, D, Y= train_TDSEP(cnt) '''
    if tau[0] == 0:
        S_T1 = np.cov(cnt)
    else:
        S_T1 = (np.dot(cnt[:, :-tau[0]], cnt[:, tau[0]:].T) + np.dot(cnt[:, tau[0]:], cnt[:, :-tau[0]].T)) / (
                    cnt.shape[1] - 1 - tau[0]) / 2
    if tau[1] == 0:
        S_T2 = np.cov(cnt)
    else:
        S_T2 = (np.dot(cnt[:, :-tau[1]], cnt[:, tau[1]:].T) + np.dot(cnt[:, tau[1]:], cnt[:, :-tau[1]].T)) / (
                    cnt.shape[1] - 1 - tau[1]) / 2

    D, W = scipy.linalg.eigh(a=S_T1, b=S_T2)
    Y = np.dot(W.T, cnt)
    S_x = np.cov(cnt)
    S_y = np.cov(Y)
    # For tau[0]=0 or tau[1]=0, the sources are uncorrelated by definition, but this is not generally the case, hence:
    A = np.dot(np.dot(S_x, W), np.linalg.inv(S_y))
    D = np.diag(D)

    return W, A, D, Y


def proc_spatialFilter(cnt, clab, chan, neighbors='*'):
    '''
    Usage:
        cnt_sf = proc_spatialFilter(cnt, clab, chan, neighbors='*')
    Parameters:
        cnt:       a 2D array of multi-channel timeseries (size: channels x samples),
        clab:      a 1D array of channel names  (size: channels)
        chan:      channel of center location
        neighbors: labels of channels that are to be subtracted
    Returns:
        cnt_sf:    timeseries of spatially filtered channel (size: 1 x samples)
    Examples:
        cnt_c4_bip = proc_spatialFilter(cnt, clab, 'C4', 'CP4')
        cnt_c4_lap = proc_spatialFilter(cnt, clab, 'C4', ['C2','C6','FC4','CP4'])
        cnt_c4_car = proc_spatialFilter(cnt, clab, 'C4', '*')
    '''
    cidx = clab.index(chan)
    if isinstance(neighbors, list):
        nidx = [clab.index(cc) for cc in neighbors]
    elif neighbors == '*':
        nidx = range(len(clab))  # Common Average Reference (CAR)
    else:
        nidx = [clab.index(neighbors)]
    cnt_sf = cnt[[cidx], :] - np.mean(cnt[nidx, :], axis=0)
    return cnt_sf


def plot_PSD(dat, fs, mrk_class, iCh=0, mrk_pos=[], ival=[]):
    if (len(dat.shape) == 2):
        dat, _ = makeepochs(dat, fs, mrk_pos, ival)

    X1 = dat[:, iCh, mrk_class == 0]
    X2 = dat[:, iCh, mrk_class == 1]
    f1, X1psd = sp.signal.welch(X1.flatten('F'), fs=fs)
    f2, X2psd = sp.signal.welch(X2.flatten('F'), fs=fs)

    plt.semilogy(f1, X1psd)
    plt.semilogy(f2, X2psd)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [$uV^2$/Hz]')


def signed_r_square(epo, y):
    '''
    Synopsis:
        epo_r = signed_r_square(epo, y)
    Arguments:
        epo:    3D array of segmented signals (time x channels x epochs),
                see makeepochs
        y:      labels with values 0 and 1 (1 x epochs)
    Output:
        epo_r:  2D array of signed r^2 values (time x channels)
    '''
    epo0 = epo[:, :, y == 0]
    epo1 = epo[:, :, y == 1]
    N1 = epo0.shape[2]
    N2 = epo1.shape[2]
    const = np.float(N1 * N2) / ((N1 + N2) ** 2)
    mudiff = epo0.mean(axis=2) - epo1.mean(axis=2)
    var = epo.var(axis=2)
    return np.sign(mudiff) * const * (mudiff) ** 2 / var