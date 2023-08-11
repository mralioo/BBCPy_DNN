import numpy as np
import scipy as sp
import sklearn as sk

import bbcpy.functions.helpers as helpers
# from bbcpy.functions import ImportFunc
from bbcpy.datatypes.eeg import Data
from bbcpy.datatypes.srm_eeg import SRM_Data
from bbcpy.functions.artireject import averagevariance
from src.data.srm_datamodule import SMR_Data


class _GEVDsf(sk.base.BaseEstimator, sk.base.TransformerMixin):
    d = None
    W = None
    A = None

    def calcAB(self, x, y=None):
        return None, None

    def calcPatterns(self, x):  #
        """
        calculates spatial pattern for data x
        :param x: data
        :return: A spatial pattern
        """
        covX = x.cov(target='all', estimator=self.estimator)
        return covX @ self.W_allcmps @ np.linalg.pinv(self.W_allcmps.T @ covX @ self.W_allcmps)[:, self.selected_cmps]

    def scoring(self, d, Y):  # just use EVs
        return d

    def select(self, score, n_cmps):  # this is the general case, CSP is a special case.
        return np.flipud(np.argsort(score))[:n_cmps]

    def fit(self, x, y=None, n_cmps=None):
        '''Fit CSP'''
        if n_cmps is None:
            n_cmps = self.n_cmps
        if y is not None:
            x.y = y
        A, B = self.calcAB(x, y)
        d, W = sp.linalg.eigh(A, B)
        score = self.scoring(d, W.T @ x)
        if n_cmps == 'all':
            selected_cmps = np.arange(x.nCh)
        else:
            selected_cmps = self.select(score, n_cmps)
        self.d = d[selected_cmps]
        self.W = W[:, selected_cmps]
        self.selected_cmps = selected_cmps
        self.W_allcmps = W
        self.A = self.calcPatterns(x)
        return self

    def transform(self, x, y=None):
        """Apply CSP.

        Parameters
        ----------
        x : ndarray, shape (n_matrices, n_channels, n_times)
            Multi-channel time-series
        y : ndarray, shape (n_trials, 2)
            Marker positions and marker class.
        Returns
        -------
        out : ndarray, shape (n_matrices, n_csp, n_time)
            transformed
        """
        return self.W.T @ x


class PCA(_GEVDsf):
    def __init__(self, n_cmps='all', excllev=None, estimator='scm', scoring=helpers.evscoring_EV,
                 select=helpers.evselect_best):
        self.n_cmps = n_cmps
        self.excllev = excllev
        self.estimator = estimator
        self.scoring = scoring
        self.select = select

    def calcAB(self, x, y=None):
        if isinstance(x, da.Array):
            x = x.compute()
        if isinstance(x, Data):
            if self.excllev is not None:
                Sigma_trial = x.cov(
                    target='all')  # no estimator here because on single trials, this will fuck up excllev
                covtr = np.trace(np.linalg.pinv(x.cov(target='all', estimator=self.estimator)) @ Sigma_trial, axis1=1,
                                 axis2=2) / x.shape[1]
                sel_tr = covtr <= self.excllev
                covs = x[sel_tr].cov(target='all', estimator=self.estimator)
            else:
                covs = x.cov(target='all', estimator=self.estimator)
            return covs, None
        else:
            c1 = sk.covariance.OAS().fit(x).covariance_
            return c1, None


class CSP(_GEVDsf):

    def __init__(self, n_cmps=6, excllev=None, estimator='scm', scoring=helpers.evscoring_EV,
                 select=helpers.evselect_best_csp):
        self.n_cmps = n_cmps
        self.excllev = excllev
        self.estimator = estimator
        self.scoring = scoring
        self.select = select

    def calcAB(self, x, y=None):

        if isinstance(x, da.Array):
            x = x.compute()
        if isinstance(x, Data) or isinstance(x, SRM_Data):  # has method cov etc
            if self.excllev is not None:
                covs = averagevariance(x, self.excllev, self.estimator).cov(target='class', estimator=self.estimator)
            else:
                covs = x.cov(target='class', estimator=self.estimator)
            return covs[0], covs[0] + covs[1]
        else:  # I would delete the following but merged it for Gabriel. Do we want to write all functions also to work
            # with arbitrary arrays? I think we should consider only our own datatypes.
            classes = np.unique(y)
            c1 = sk.covariance.OAS().fit(x[y == classes[0]]).covariance_
            c2 = sk.covariance.OAS().fit(x[y == classes[1]]).covariance_
            return c1, c1 + c2
        # if self.excllev is not None:
        #     covs = averagevariance(x, self.excllev, self.estimator).cov(target='class', estimator=self.estimator)
        # else:
        #     covs = x.cov(target='class', estimator=self.estimator)
        # return covs[0], covs[0] + covs[1]


if __name__ == "__main__":
    srm_raw_path = "data/SMR/raw/"
    srm_data = SMR_Data(data_dir=srm_raw_path,
                        bands=[8, 13],
                        classes=["R", "L"],
                        chans=['C*', 'FC*'],
                        ival="2s:8s:10ms")
    obj = srm_data.load_data(subjects_dict={"S1": [1]})

    import dask.array as da

    # Convert your data to Dask arrays or dataframes
    x_train = obj
    y_train = obj.y

    X_train_dask = da.from_array(x_train, chunks='auto')
    y_train_dask = da.from_array(y_train, chunks='auto')
