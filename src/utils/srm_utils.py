import logging

import numpy as np

import bbcpy


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
