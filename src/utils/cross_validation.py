import numpy as np
from sklearn.model_selection._split import _BaseKFold
from sklearn.utils.validation import _num_samples


class TrialWiseKFold(_BaseKFold):
    """Trial wise KFold cross-validator

        # TODO: add to bbcpy package
        # TODO : support multi classes

        Provides train/test indices to split data in train/test sets. Split
        dataset into k consecutive folds (without shuffling by default) to
        achieve an equal number of movement trials in each split.
        Each fold is then used once as a validation while the k - 1 remaining
        folds form the training set.
    """

    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle,
                         random_state=random_state)

    def _iter_test_indices(self, X, y=None, groups=None):
        n_samples = _num_samples(X)
        indices = np.arange(n_samples)

        labeldiff = np.argwhere(np.diff(X.y) != 0)[:, 0] + 1
        labeldiff = np.insert(labeldiff, 0, 0)
        labeldiff = np.append(labeldiff, n_samples)

        ld_low = labeldiff[::2]
        ld_high = labeldiff[1::2]

        idx_low = ld_low[
            np.floor(np.linspace(0, len(ld_low) - 1,
                                 self.n_splits + 1)).astype(int)]
        idx_up = ld_high[np.floor(
            np.linspace(0, len(ld_high) - 1, self.n_splits + 1)).astype(int)]

        cv_indices = (idx_low + idx_up) // 2
        cv_indices[0] = 0
        cv_indices[-1] = n_samples
        for i in range(self.n_splits):
            yield indices[cv_indices[i]:cv_indices[i + 1]]
