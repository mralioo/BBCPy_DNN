import bbcpy
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.model_selection import KFold, ShuffleSplit, GridSearchCV

from bbcpy.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

if __name__ == "__main__":
    ival = [1000, 4500]
    band = [10.5, 13]
    imagVPaw = bbcpy.load.eeg.data('../data/bbcpydata/imagVPaw.npz')

    # examples for new advanced indexing:

    # select all but some (here exclude all frontal and fronto-central) channels:
    print(imagVPaw['~F*'].shape)
    # can also be combined - selecting all frontal channels except for 7 and 8s e.g.:
    print(imagVPaw[['F?', '~*7', '~*8']].shape)
    print(imagVPaw[['F?,~*7,~*8']].chans)
    # adding advanced slicing with units - e.g. 2s to 5s in steps of 100ms:
    print(imagVPaw['~F*', '2s:5s:100ms'].shape)
    # you may also select single time points:
    print(imagVPaw[:, '2s,5s,100ms'].shape)
    # the same can be separated in a list:
    print(imagVPaw[:, ['2s', '5s', '100ms']].shape)

    cnt_bp = imagVPaw.lfilter(band)
    epo_bp = cnt_bp.epochs(ival)

    # also works on epoched data, also with comma separation, each taken individually
    print(epo_bp[:, ['C?,~*3,~*4', 'FC*,~*3,~*4']].chans)
    print(epo_bp[:, ['C?,~*3,~*4', 'FC*,~*3,~*4'], '2s:4s:100ms'].shape)
    # and also selecting classes
    print(epo_bp[['left']].shape)
    print(epo_bp['left'][:, ['C?,~*3,~*4', 'FC*,~*3,~*4']].shape)
    # changing order of classes
    print(epo_bp[['right', 'left']][:, ['C?,~*3,~*4', 'FC*,~*3,~*4'], '2s:4s:100ms'].shape)

    hilbilly = bbcpy.functions.base.ImportFunc(signal.hilbert, outClass='same', axis=-1)
    ERD1 = np.abs(hilbilly(cnt_bp)).epochs(ival)
    plt.figure()
    plt.subplot(131)
    bbcpy.visual.scalp.map(imagVPaw, ERD1.classmean(0)[0, :, '3s'])
    plt.subplot(132)
    bbcpy.visual.scalp.map(imagVPaw, ERD1.classmean(1)[0, :, '3s'])
    plt.subplot(133)
    bbcpy.visual.scalp.map(imagVPaw, ERD1.classmean(1)[0, :, '3s'] - ERD1.classmean(0)[0, :, '3s'])
    plt.figure()
    plt.subplot(211)
    plt.plot(ERD1.t, np.squeeze(ERD1.classmean()[:, 'C3']).T)
    plt.subplot(212)
    plt.plot(ERD1.t, np.squeeze(ERD1.classmean()[:, 'C4']).T)
    plt.show()

    # cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    cv = KFold()

    vary = bbcpy.functions.base.ImportFunc(np.var, axis=2)
    clf = make_pipeline(vary, np.log, lda())
    print(cross_val_score(clf, imagVPaw.lfilter(band).epochs(ival), ERD1.y, cv=cv))

    flatfeature = bbcpy.functions.base.ImportFunc(lambda x: np.reshape(x, (x.shape[0], -1)))
    clf = make_pipeline(bbcpy.functions.statistics.cov, flatfeature, svm.SVC(C=1))
    print(cross_val_score(clf, epo_bp, epo_bp.y, cv=cv))

    flatfeature = bbcpy.functions.base.ImportFunc(lambda x: np.reshape(x, (x.shape[0], -1)))
    clf = make_pipeline(bbcpy.functions.statistics.cov, flatfeature, lda())
    print(cross_val_score(clf, epo_bp, epo_bp.y, cv=cv))

    # print(bbcpy.functions.spatial.CSP()(imagVPaw.lfilter(band).epochs(ival)))

    clf = make_pipeline(bbcpy.functions.spatial.CSP(), vary, np.log, lda())
    benchmark_res = cross_val_score(clf, epo_bp, epo_bp.y, cv=cv)
    print(benchmark_res)

    clf = make_pipeline(bbcpy.functions.spatial.CSP(), vary, np.log, svm.SVC(C=1))
    print(cross_val_score(clf, epo_bp, epo_bp.y, cv=cv))

    meany = bbcpy.functions.base.ImportFunc(np.mean, axis=-1)
    clf = make_pipeline(meany, svm.SVC(C=1))
    print(cross_val_score(clf, ERD1, ERD1.y, cv=cv))

    clf = make_pipeline(flatfeature, svm.SVC(C=1))
    print(cross_val_score(clf, ERD1, ERD1.y, cv=cv))

    # testing baselining (and that its not  a good idea, here)
    ival2 = [-100, 5000]
    selival = bbcpy.functions.base.ImportFunc(lambda x: x[:, :, (x.t >= ival[0]) & (x.t <= ival[1])])
    ERD2 = np.abs(hilbilly(cnt_bp)).epochs(ival2)
    clf = make_pipeline(bbcpy.functions.normalizations.baseline, meany, svm.SVC(C=1))
    print(cross_val_score(clf, ERD2, ERD2.y, cv=cv))
