import logging

import numpy as np
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import StratifiedKFold

import bbcpy


def normalize(data, norm_type="std", axis=None, keepdims=True, eps=10 ** -5, norm_params=None):
    """Normalize data along a given axis.

    Parameters
    ----------
    data : numpy.ndarray
        Data to normalize.
    norm_type : str
        Type of normalization. Can be 'std' or 'minmax'.
    axis : int
        Axis along which to normalize.
    keepdims : bool
        Whether to keep the dimensions of the original data.
    eps : float
        Epsilon to avoid division by zero.
    norm_params : dict
        Dictionary containing normalization parameters. If None, they will be computed.

    Returns
    -------
    data_norm : numpy.ndarray
        Normalized data.
    norm_params : dict
        Dictionary containing normalization parameters.

    """

    if norm_params is not None:
        if norm_params["norm_type"] == "std":
            data_norm = (data - norm_params["mean"]) / norm_params["std"]
        elif norm_params["norm_type"] == "minmax":
            data_norm = (data - norm_params["min"]) / (norm_params["max"] - norm_params["min"])
        else:
            raise RuntimeError("norm type {:} does not exist".format(norm_params["norm_type"]))

    else:
        if norm_type == "std":
            data_std = data.std(axis=axis, keepdims=keepdims)
            data_std[data_std < eps] = eps
            data_mean = data.mean(axis=axis, keepdims=keepdims)
            data_norm = (data - data_mean) / data_std
            norm_params = dict(mean=data_mean, std=data_std, norm_type=norm_type, axis=axis, keepdims=keepdims)
        elif norm_type == "minmax":
            data_min = data.min(axis=axis, keepdims=keepdims)
            data_max = data.max(axis=axis, keepdims=keepdims)
            data_max[data_max == data_min] = data_max[data_max == data_min] + eps
            data_norm = (data - data_min) / (data_max - data_min)
            norm_params = dict(min=data_min, max=data_max, norm_type=norm_type, axis=axis, keepdims=keepdims)
        elif norm_type is None:
            data_norm, norm_params = data, None
        else:
            data_norm, norm_params = None, None
            ValueError("Only 'std' and 'minmax' are supported")

    return data_norm, norm_params


def unnormalize(data_norm, norm_params):
    if norm_params["norm_type"] == "std":
        data = data_norm * norm_params["std"] + norm_params["mean"]
    elif norm_params["norm_type"] == "minmax":
        data = data_norm * (norm_params["max"] - norm_params["min"]) + norm_params["min"]
    return data


def transform_electrodes_configurations(epo_data):
    # FIXME : not implemented yet

    """
       This function will generate the channel order for TSception
       Parameters
       ----------
       original_order: list of the channel names

       Returns
       -------
       TS: list of channel names which is for TSception
       """

    original_order = epo_data.chans
    chan_name, chan_num, chan_final = [], [], []
    for channel in original_order:
        chan_name_len = len(channel)
        k = 0
        for s in [*channel[:]]:
            if s.isdigit():
                k += 1
        if k != 0:
            chan_name.append(channel[:chan_name_len - k])
            chan_num.append(int(channel[chan_name_len - k:]))
            chan_final.append(channel)
    chan_pair = []
    for ch, id in enumerate(chan_num):
        if id % 2 == 0:
            chan_pair.append(chan_name[ch] + str(id - 1))
        else:
            chan_pair.append(chan_name[ch] + str(id + 1))
    chan_no_duplicate = []
    [chan_no_duplicate.extend([f, chan_pair[i]]) for i, f in enumerate(chan_final) if
     f not in chan_no_duplicate]

    new_order = chan_no_duplicate[0::2] + chan_no_duplicate[1::2]
    # FIXME : not sure if this is the correct way to do it
    return epo_data[:, new_order, :]


def remove_reference_channel(clab, mnt, reference_channel="REF."):
    """ Set the EEG channels object, and remove the reference channel if exists

    Parameters
    ----------
    clab: numpy array
        List of channels names
    mnt: numpy array
        List of channels positions
    reference_channel: str
        Name of the reference channel

    Returns
    -------
    chans: bbcpy.datatypes.eeg.Chans

    """

    if reference_channel in clab:

        ref_idx = np.where(clab == "REF.")[0][0]
        clab = np.delete(clab, ref_idx)
        mnt = np.delete(mnt, ref_idx, axis=0)
        chans = bbcpy.datatypes.eeg.Chans(clab, mnt)

    else:
        logging.warning(f"Reference channel {reference_channel} not found in the data")
        chans = bbcpy.datatypes.eeg.Chans(clab, mnt)

    return chans


def calculate_pvc_metrics(trial_info, taskname="LR"):
    """PVC is metric introduced in the dataset paper and it descibes the percentage of valid correct trials
    formula: PVC = hits / (hits + misses)
    """
    task_num_dict = {"LR": 1.0, "UD": 2.0, "2D": 3.0}
    task_filter_idx = np.where(np.array(trial_info["tasknumber"]) == task_num_dict[taskname])[0]

    # get hits and misses TODO forcedresult
    trials_results = np.array(trial_info["forcedresult"])[task_filter_idx]
    res_dict = {"hits": np.sum(trials_results == True), "misses": np.sum(trials_results == False)}

    # calculate pvc
    pvc = res_dict["hits"] / (res_dict["hits"] + res_dict["misses"])
    return pvc


def train_valid_split(data_runs_list, val_ratio=0.1, random_seed=42):
    """ Split the data into train and validation sets and return indices
    Split is done as follow take 10 % for each the 5 runs """

    # Set the random seed for reproducibility
    np.random.seed(random_seed)
    val_runs_list = []
    train_runs_list = []
    for run in data_runs_list:
        data_shape = run.shape

        # Shuffle the data
        indices = np.random.permutation(data_shape[0])

        # Compute the index where the validation set starts
        val_start_idx = int(len(indices) * (1 - val_ratio))

        # Get the indices for training and validation sets
        train_indices = indices[:val_start_idx]
        val_indices = indices[val_start_idx:]

        val_runs_list.append(run[val_indices])
        train_runs_list.append(run[train_indices])

    return train_runs_list, val_runs_list


def cross_validation(data, kfold_idx):
    """ Split the data into train and validation sets and return indices. Split is done using StratifiedKFold
    Parameters
    ----------
    data : numpy.ndarray
        Data to split.
    labels : numpy.ndarray
        Labels to split.
    kfold_idx : int
        Index of the fold to use for validation.

    Returns
    -------
    train_indexes : list
        List of indexes for training set.
    val_indexes : list
        List of indexes for validation set.
    """

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=5)

    # Split data using StratifiedKFold
    all_splits_skf = [(train_index, val_index) for train_index, val_index in skf.split(data, data.y)]

    # Retrieve train and validation indexes for the specified fold
    train_indexes, val_indexes = all_splits_skf[kfold_idx]

    return train_indexes.tolist(), val_indexes.tolist()