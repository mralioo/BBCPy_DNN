import bbcpy
import numpy as np
import numpy.ma as ma
from matplotlib import pyplot as plt
from scipy import signal
import scipy.io as sio
from datasets import utils as dutils
from data.SMR.eeg import *
from bbcpy.datatypes.eeg import Data, Chans, Marker, Epo

task_map_dict = {"LR": 1, "UD": 2, "2D": 3}
target_map_dict = {"R": 1, "L": 2, "U": 3, "D": 4}
class_name = ["R", "L", "U", "D"]


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


def makeepochs_srm(srm_data, timepoints, ival, fs, mrk):
    """Reshape trials to have the same length using numpy.ma """

    # TODO support time interval selection

    trial_maxlen_id = np.argmax([len(t) for t in timepoints])
    trial_maxlen = timepoints[trial_maxlen_id].shape[-1]
    nChans = srm_data[0].shape[0]
    # Create a masked array with the same size as the largest sub ndarray
    new_srm_data = ma.zeros((srm_data.shape[0], nChans, trial_maxlen))
    # Set the mask for each row based on the sub ndarray size
    for i, sub_arr in enumerate(srm_data):
        new_srm_data[i, :, :sub_arr.shape[-1]] = sub_arr

    # FIXME
    if isinstance(ival, slice):
        if ival.start is None:
            start = 0
        else:
            start = ival.start
        if ival.stop is None:
            stop = 0
        else:
            stop = ival.stop
        # FIXME
        if not ival.step is None:
            x = srm_data[::ival.step]
            start = start / ival.step
            stop = stop / ival.step
        ival = [start, stop]

    time = np.arange(int(np.floor(ival[0] * fs / 1000)),
                     int(np.ceil(ival[1] * fs / 1000)) + 1, dtype=int)

    epo = np.array(new_srm_data)[:, :, time]
    epo_t = np.linspace(ival[0], ival[1], len(time))

    return epo, epo_t

class SRM_Marker(np.ndarray):
    global task_map_dict
    global target_map_dict

    def __new__(cls, mrk_class, mrk_class_name, trialresult, triallength, mrk_fs=1):

        obj = np.asarray(mrk_class).view(cls)
        obj.y = np.array(mrk_class)  # target was presented to the participants (1=right, 2=lef, 3=up, 4=down)
        obj.className = mrk_class_name  # {"R": 1, "L": 2, "U": 3, "D": 4}
        # obj.className = np.array(trial_info["tasknumber"])  # 1 = "LR"; 2= "UD", 3 = "2D"
        # obj.y_true = np.array(trial_info["targetnumber"])  # target was presented to the participants (1=right, 2=lef, 3=up, 4=down)
        # obj.y_hit = np.array(trial_info["targethitnumber"])  # target was selected during feedback (1=right, 2=lef, 3=up, 4=down, NaN when a trial is a timeout)
        obj.trialresult = np.array(
            trialresult)  # the outcome of the trial which takes values of 1 when the correct target was selected, 0 when an incorrect target is selected and NaN if the trial was labeled as a timeout
        obj.triallength = np.array(triallength)  # length of the feedback control period in seconds
        # obj.subject_info = subject_info
        obj.fs = mrk_fs
        return obj

    def __array_wrap__(self, out_arr, context=None):  # used for printing,squeezing etc
        # return self.__class__(self, *self.__initargs__())
        obj = super().__array_wrap__(out_arr, context)
        obj.y = np.copy(self.y)
        obj.triallength = np.copy(self.triallength)
        obj.trialresult = np.copy(self.trialresult)
        obj.className = np.copy(self.className)
        # obj.subject_info = np.copy(self.subject_info)
        obj.fs = np.copy(obj.fs)
        return obj

    def __array_finalize__(self, obj, *args, **kwargs):
        self.y = getattr(obj, 'y', None)
        self.triallength = getattr(obj, 'triallength', None)
        self.trialresult = getattr(obj, 'trialresult', None)
        self.className = getattr(obj, 'className', None)
        # self.subject_info = getattr(obj, 'subject_info', None)
        self.fs = getattr(obj, 'fs', None)

    def __getitem__(self, key):
        # TODO: add support for slicing
        newy = self.y[key].copy()
        # FIXME should have the shape of number of trials
        newtriallength = self.triallength[key].copy()
        newtrialresult = self.trialresult[key].copy()
        if isinstance(key, int) or ((not isinstance(key, slice)) and len(key) == 1):
            newy = [newy]
        leftclasses = np.unique(self.y[key])
        if len(leftclasses) < len(self.className):
            newind = 0
            for i in range(len(self.className)):
                if i in leftclasses:
                    newy[newy == i] = newind
                    newind += 1
        newclassName = np.copy([self.className[lc] for lc in leftclasses])
        obj = SRM_Marker(newy, newclassName, newtrialresult, newtriallength, np.copy(self.fs))
        return obj

    def get_valid_trials(self):
        """
        get trials that are not timeouts
        :return:
        :rtype:
        """
        valid_trials = np.where(~np.isnan(self.trialresult))[0]
        return valid_trials

    def get_class_inds(self, classes):
        """
        tasks in srm datasett are labeled "R": 1, "L": 2, "U": 3, "D": 4
        :param task:
        :type task:
        :return:
        :rtype:
        """
        classes_int = []
        valid_trials = self.get_valid_trials()
        for class_name in classes:
            class_index = target_map_dict[class_name]
            inds_t = np.where(self.y == class_index)[0]

            # remove all trials that are timeouts or not valid
            inds = np.intersect1d(inds_t, valid_trials)

            if len(inds) >= 0:
                classes_int.append(inds)
            else:
                warnings.warn('Class "%s" not found.' % class_name)

        return classes_int

    def __select_classes__(self, classes):
        if isinstance(classes, str):  # single class string selection needs to be list for get_class_inds
            classes = [classes]
        if isinstance(classes, list) and isinstance(classes[0], str):
            classes_ids = self.get_class_inds(classes)

        selected = np.zeros(self.shape, dtype=bool)
        newy = np.zeros(self.shape, dtype=int)
        for ii, ic in enumerate(classes_ids):
            selected[ic] = True
            newy[ic] = ii + 1
        newclassName = np.unique(newy)
        return selected, newy, newclassName

    def select_classes(self, classes):
        obj = self.copy()
        selected, obj.y, obj.className = self.__select_classes__(classes)

        # FIXME : should have 450 trials
        obj.triallength =obj.triallength[selected]
        obj.trialresult = obj.trialresult[selected]
        # FIXME index not working on obj with 450 trials
        # obj = obj[selected]
        return obj

    def in_samples(self, fs=None):
        if fs is None:
            return self
        else:
            return self / self.fs * fs

    def in_ms(self):
        return self / self.fs * 1000


class SRM_Data(np.ndarray):
    def __new__(cls, data, time, fs, mrk, chans=None):
        """

        :param data: srm data of the shape trial x channels x time
        :type data: nested numpy array
        :param fs: sampling frequency
        :type fs: float
        :param mrk: object containing trial info
        :type mrk: SRM_Marker object
        :param chans: Channel names
        :type chans: Chans object
        """

        # data need to be with even length
        obj = np.asarray(data).view(cls)
        obj.fs = fs
        obj.mrk = mrk
        obj.chans = chans
        if not isinstance(time, (list, np.ndarray)):
            time = np.array([time])
        obj.t = time
        return obj

    def __array_wrap__(self, out_arr, context=None, *args, **kwargs):  # used for printing,squeezing etc
        # return self.__class__(self, *self.__initargs__())
        obj = super().__array_wrap__(out_arr, context)
        obj.fs = self.fs.copy()
        obj.mrk = self.mrk.copy()
        obj.chans = self.chans.copy()
        return obj

    def __array_finalize__(self, obj, *args, **kwargs):
        if obj is None:
            return
        self.fs = getattr(obj, 'fs', None)
        self.mrk = getattr(obj, 'mrk', None)
        self.chans = getattr(obj, 'chans', None)

    def __initargs__(self):
        return self.fs, self.mrk, self.chans

    def __getitem__(self, key):

        if self.ndim == 3:

            # TODO: index mrk object (FIXME)
            if isinstance(key, type(np.newaxis)):  # markers stay the same if dimension added
                mrk = self.mrk.copy()
            elif isinstance(key, str) or \
                    isinstance(key, (tuple, list, np.ndarray)) and isinstance(key[0], str) \
                    or isinstance(key, tuple) and isinstance(key[0], (list, np.ndarray)) and isinstance(key[0][0], str):
                # class selection by strings
                mrk = self.mrk.copy()
                if isinstance(key, (str, list, np.ndarray)):
                    selected, mrk.y, mrk.className = self.mrk.__select_classes__(key)
                    # mrk = mrk[selected]  # FIXME indexing mrk object
                    key = selected
                else:  # is tuple
                    key = list(key)
                    selected, mrk.y, mrk.className = self.mrk.__select_classes__(key[0])
                    # mrk = mrk[selected]
                    key[0] = selected
                    key = tuple(key)
            elif isinstance(key, (int, slice, list, np.ndarray)) or (len(key) == 1):
                # selecting only markers directly
                mrk = self.mrk.__getitem__(key).copy()
            elif isinstance(key, tuple) and (isinstance(key[0], (int, slice, list, tuple, np.ndarray))):
                # selecting markers directly together with other dimensions
                mrk = self.mrk.__getitem__(key[0]).copy()
            else:
                # no markers selected (I actually lost track when this should happen)
                mrk = self.mrk.copy()

            # TODO : index channels (checked)
            if isinstance(key, tuple) and len(key) > 1 and \
                    (isinstance(key[1], (int, str, slice, list, tuple, np.ndarray))) and (len(self.chans.shape) > 0):
                # selecting channels
                key = list(key)
                if isinstance(key[1], (list, tuple, np.ndarray)) and len(key[1]) == 1:
                    key[1] = key[1][0]
                key[1] = self.chans.index(key[1])
                key = tuple(key)
                chans = self.chans.copy()[key[1]]
            else:  # No channels selected
                chans = self.chans.copy()

            # TODO: index time (checked)
            if isinstance(key, tuple) and len(key) > 2 and (isinstance(key[2], str)
                                                            or (isinstance(key[2], (list, np.ndarray)) and any(
                        [isinstance(k, str) for k in key[2]]))):
                # selecting time points
                key = list(key)
                key[2] = gettimeindices(key[2], 1000)
                # recalculate according to t
                if isinstance(key[2], slice):
                    if key[2].start is None:
                        start = None
                    else:
                        start = np.where(self.t >= key[2].start)[0][0]
                    if key[2].stop is None:
                        stop = None
                    else:
                        stop = np.where(self.t >= key[2].stop)[0][0]
                    if key[2].step is None:
                        step = None
                    else:
                        if np.round(key[2].step / 1000 * self.fs) < 1:
                            raise ValueError('Step size too small.')
                        step = int(np.round(key[2].step / 1000 * self.fs))
                    key[2] = slice(start, stop, step)
                else:  # No slices, selecting individual timepoints
                    for i, k in enumerate(key[2]):
                        if k < self.t[0] or k > self.t[-1]:
                            raise ValueError('indexed timepoint %f is out of range of data' % k)
                        key[2][i] = np.argmin(np.abs(k - self.t))
                    tempkey = np.unique(key[2])
                    if len(tempkey) < len(key[2]):
                        warnings.warn('Some indices where used multiple times, reducing to unique indices. '
                                      'This might be caused by sampling rate issues')
                    key[2] = tempkey
                key = tuple(key)
            x = np.asarray(self)[key]  # get data only

            # TODO : time domain is changed (checked)
            if isinstance(key, tuple) and len(key) > 2:  # time domain is changed
                # FIXME : not clear how it works
                t = self.t[key[2]]
                if isinstance(key[2], slice) and key[2].step is not None:
                    # time is sliced, sampling rate might be changed
                    fs = float(self.fs) / key[2].step
                else:
                    fs = self.fs
            else:  # time domain is not changed
                t = self.t
                fs = self.fs

            # TODO : epochs selected (not tested)
            if isinstance(key, int):  # if only single epoch selected, still keep dimensionality
                x = x[np.newaxis]
            if isinstance(key, tuple):  # if multiple dimensions selected still keep general structure
                if isinstance(key[0], (int, str)) or isinstance(key, (list, np.ndarray)) and len(
                        key[0]) == 1:
                    x = x[np.newaxis]
                if len(key) > 1 and isinstance(key[1], (int, str)) or isinstance(key, (list, np.ndarray)) and len(
                        key[1]) == 1:
                    x = x[:, np.newaxis]
                if len(key) > 2 and isinstance(key[2], (int, str)) or isinstance(key, (list, np.ndarray)) and len(
                        key[2]) == 1:
                    x = x[:, :, np.newaxis]

            # TODO : return selected data
            obj = SRM_Data(x, t, fs, mrk, chans)
        else:  # needed for certain np operations because structure is lost
            obj = np.asarray(self)[key]
        return obj
        #     if isinstance(key, type(np.newaxis)):
        #         chans = self.chans.copy()
        #     elif isinstance(key, (int, str, slice, list, np.ndarray)) or (len(key) == 1):
        #         key = self.chans.index(key)
        #         chans = self.chans.copy()[key]
        #     elif isinstance(key, tuple) and (isinstance(key[0], (int, str, slice, list, np.ndarray))) and (len(self.chans.shape) > 0):  # channels selected together with time domain
        #         key = list(key)
        #         key[0] = self.chans.index(key[0])
        #         key = tuple(key)
        #         chans = self.chans.copy()[key[0]]
        #     else:  # channels not changed
        #         chans = self.chans.copy()
        #
        #     # FIXME: time domain selection not working
        #     if isinstance(key, tuple) and len(key) > 1 and (isinstance(key[1], (str, list, np.ndarray))):
        #         # time selected and probably string indexing given
        #         key = list(key)
        #         key[1] = gettimeindices(key[1], self.fs)
        #         key = tuple(key)
        #     if isinstance(key, tuple) and len(key) > 1 and isinstance(key[1], slice) and key[1].step is not None:
        #         # time is sliced, sampling rate might be changed
        #         fs = float(self.fs) / key[1].step
        #     else:
        #         fs = self.fs
        #
        # # iterate over trials and select from key
        # x = np.asarray(self)
        #
        # sliced_data = [trial[key] for trial in x]
        # # Transform the sliced_data list to an ndarray object with dtype set to object
        # sliced_data_array = np.asarray(sliced_data, dtype=object)
        # sliced_data_array[:] = sliced_data
        #
        #
        # # create new object with sliced data
        # obj = SRM_Data(sliced_data_array, fs,self.mrk, chans)
        #
        # return obj

    def copy(self, order='K'):
        obj = self.__class__(super().copy(order=order), *self.__initargs__())
        obj.fs = self.fs.copy()
        obj.mrk = self.mrk.copy()
        obj.chans = self.chans.copy()
        return obj

    @property
    def nT(self):
        if np.isscalar(self):
            return 1
        else:
            return self.shape[2]

    @property
    def nCh(self):
        if np.isscalar(self):
            return 1
        else:
            return self.shape[1]

    @property
    def nEpo(self):
        if np.isscalar(self):
            return 1
        else:
            return self.shape[0]

    @property
    def nClass(self):
        if np.isscalar(self):
            return 1
        else:
            return len(self.mrk.className)
