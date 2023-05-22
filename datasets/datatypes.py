import bbcpy
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import scipy.io as sio
from datasets import utils as dutils
from data.SMR.eeg import *
from bbcpy.datatypes.eeg import Data, Chans, Marker, Epo

class SRM_Data(Data):

    def __new__(cls, data,fs,trial_info,subject_info,chans=None):
        obj = super().__new__(cls,data, fs, chans)
        obj.data = data
        obj.fs = fs
        obj.chans = list(chans)
        obj.trial_info = trial_info
        obj.subject_info = subject_info
        return obj

    def __getitem__(self, key):
        # FIXME select all trials
        if isinstance(key , (int, slice, list, np.ndarray)) or (len(key) == 1):
            key = self.chans.index(key)
            chans = self.chans.copy()[key]
        elif isinstance(key, tuple) and (isinstance(key[0], (int, str, slice, list, np.ndarray))) and (
                len(self.chans.shape) > 0):  # channels selected together with time domain
            key = list(key)
            key[0] = self.chans.index(key[0])
            key = tuple(key)
            chans = self.chans.copy()[key[0]]
        else:  # channels not changed
            chans = self.chans.copy()

        if isinstance(key, tuple) and len(key) > 1 and (isinstance(key[1], (str, list, np.ndarray))):
            # time selected and probably string indexing given
            key = list(key)
            key[1] = gettimeindices(key[1], self.fs)
            key = tuple(key)
        if isinstance(key, tuple) and len(key) > 1 and isinstance(key[1], slice) and key[1].step is not None:
            # time is sliced, sampling rate might be changed
            fs = float(self.fs) / key[1].step
        else:
            fs = self.fs

        x = np.asarray(self)[key]



    def epochs(self, ival, mrk=None, clsinds=None):
        """Extract epochs from continuous data."""
        if mrk is None:
            try:
                mrk = self.mrk
            except (NameError, AttributeError):
                raise AttributeError('Error: Cannot make epochs: mrk not provided.')
        if clsinds is not None:
            mrk = mrk.select_classes(clsinds)
        ival = gettimeindices(ival, 1000)
        epo = bbcpy.functions.structural.makeepochs(self, ival, mrk)

        pass

def transform_srm_to_eeg(data, timepoints, fs):
    """transform srm data to the standard eeg data format"""
    pass


def gettimeindices(orig_key, fs):
    if isinstance(orig_key, (list, np.ndarray)):
        key = orig_key
        for i in range(len(orig_key)):
            key[i] = gettimeindices(orig_key[i], fs)
            if isinstance(key[i], slice):
                if len(key) > 1:
                    raise ValueError('It is not possible to combine multiple indexings if any of them is a slice.')
                return key[i]
        newkey = []
        for sublist in key:
            if isinstance(sublist, (list, np.ndarray)):
                for item in sublist:
                    newkey.append(item)
            else:
                newkey.append(sublist)
        key = newkey

    elif isinstance(orig_key, str):  # str "100ms:450ms" or "100ms,230ms,..."
        slice_found = False
        key = orig_key.split(",")
        for i, k in enumerate(key):
            key[i] = k.split(":")
            if isinstance(key[i], list) and len(key) > 1 and len(key[i]) > 1:
                raise ValueError('It is not possible to combine multiple indexings if any of them is a slice.')

        key = [[k.strip() for k in k2] for k2 in key]
        for i, k in enumerate(key):
            for i2, k2 in enumerate(k):
                # inums = np.sum([s.isnumeric() for s in k2]) does not work for floats due to point
                inums = len(k2) - np.sum([s.isalpha() for s in k2])
                num = float(k2[:inums])
                if inums < len(k2):
                    unit = k2[inums:]
                    if unit == 'ms':
                        factor = float(fs) / 1000
                    elif unit in ('s', 'sec'):
                        factor = float(fs)
                    elif unit in ('m', 'min'):
                        factor = float(fs) * 60
                    elif unit == 'h':
                        factor = float(fs) * 3600
                    k[i2] = int(np.round(num * factor))
                else:
                    k[i2] = int(k2)
                    warnings.warn(
                        'No unit given for one or more elements in [%s], assuming samples for these.' % (orig_key))
            if len(k) > 1:
                key = slice(*key[i])
                slice_found = True
        if not slice_found:
            key = [item for sublist in key for item in sublist]
    else:
        key = orig_key
    return key


def makeepochs(x: bbcpy.datatypes.eeg.Data, ival: np.ndarray, mrk = None) -> bbcpy.datatypes.eeg.Epo:
    """
    Usage:
        makeepochs(X, ival)
    Parameters:
        x: 2D array of multi-channel timeseries (channels x samples) of type EEGdata
        ival: a two element vector giving the time interval relative to markers (in ms)
    Returns:
        epo: a 3D array of segmented signals (samples x channels x epochs) of type EEGepo
    """
    if isinstance(ival, slice):
        if ival.start is None:
            start = 0
        else:
            start = ival.start
        if ival.stop is None:
            stop = 0
        else:
            stop = ival.stop
        if not ival.step is None:
            x = x[::ival.step]
            start = start / ival.step
            stop = stop / ival.step
        ival = [start, stop]
    if mrk is None:
        mrk = x.mrk.copy()
    time = np.arange(int(np.floor(ival[0] * x.fs / 1000)),
                     int(np.ceil(ival[1] * x.fs / 1000)) + 1, dtype=int)[np.newaxis, :]
    T = time.shape[1]
    nEvents = len(mrk)
    nChans = x.shape[0]
    idx = (time.T + np.array([mrk.in_samples(x.fs)])).reshape(T * nEvents).astype(int)
    epo = np.array(x)[:, idx].reshape(nChans, T, nEvents)
    epo = np.transpose(epo, (2, 0, 1))
    epo_t = np.linspace(ival[0], ival[1], T)
    if isinstance(x, bbcpy.datatypes.eeg.AppendedChData):
        epo = bbcpy.datatypes.eeg.AppendedChEpo(epo, epo_t, x.fs, mrk, x.chans)
    else:
        epo = bbcpy.datatypes.eeg.Epo(epo, epo_t, x.fs, mrk, x.chans)
    return epo