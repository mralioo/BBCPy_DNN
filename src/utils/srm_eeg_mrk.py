import copy
import warnings

import numpy as np

task_map_dict = {"LR": 1, "UD": 2, "2D": 3}
 # I subtracted 1 from the original values
class_name = ["R", "L", "U", "D"]
target_map_dict = {0: 'R', 1: 'L', 2: 'U', 3: 'D'}

class SRM_Marker(np.ndarray):
    """ Class for marker data.
    """

    def __new__(cls,
                mrk_pos,
                mrk_class,
                mrk_class_name,
                mrk_fs=1,
                parent_fs=None):
        """ Create a new Marker object.
        mrk_pos: index of marker in samples,that separate trial
        mrk_class: class name of each trial (e.g. 'left', 'right')
        mrk_class_name: name classes
        mrk_fs: sampling rate of epoch
        parent_fs: sampling rate of data
        """

        if not isinstance(mrk_pos, (np.ndarray, list)):
            mrk_pos = [mrk_pos]
        obj = np.asarray(mrk_pos).view(cls)
        obj.y = np.array(mrk_class)
        if mrk_class_name is None:
            mrk_class_name = np.unique(mrk_class)
        obj.className = copy.deepcopy(mrk_class_name)
        obj.fs = mrk_fs
        obj.parent_fs = parent_fs
        return obj

    def __array_wrap__(self, out_arr, context=None):  #
        obj = super().__array_wrap__(out_arr, context)
        obj.y = self.y.copy()
        obj.className = self.className.copy()
        obj.fs = obj.fs
        obj.parent_fs = obj.parent_fs
        return obj

    def __array_finalize__(self, obj):  #
        self.y = getattr(obj, 'y', None)
        self.className = getattr(obj, 'className', None)
        self.fs = getattr(obj, 'fs', None)
        self.parent_fs = getattr(obj, 'parent_fs', None)

    def __init__(self, mrk_pos, mrk_class, mrk_class_name=None, mrk_fs=1, parent_fs=1):
        return  # super().init()

    def __getitem__(self, key):
        newy = self.y[key].copy()
        if isinstance(key, int) or ((not isinstance(key, slice)) and len(key) == 1):
            newy = [newy]
        leftclasses = np.unique(self.y[key])
        if len(leftclasses) < len(self.className):
            # if not isinstance(key, int) and len(key) > 1:
            #    warnings.warn('removing void classes')
            newind = 0
            for i in range(len(self.className)):
                if i in leftclasses:
                    newy[newy == i] = newind
                    newind += 1
        newclassName = np.copy([target_map_dict[lc] for lc in leftclasses])
        obj = SRM_Marker(super().__getitem__(key).copy(),
                         mrk_class=newy,
                         mrk_class_name=newclassName,
                         mrk_fs=np.copy(self.fs),
                         parent_fs=np.copy(self.parent_fs))
        # add channel selection!
        # if len(self.shape):
        #    obj = Marker(super().__getitem__(key).copy(), newy, newclassName, mrk_fs=np.copy(self.fs))
        # else:
        #    obj = Marker(self.copy(), newy, newclassName, mrk_fs=np.copy(self.fs))
        return obj

    def sort(self, axis=-1, kind=None, order=None):
        sortedinds = self.argsort(axis, kind, order)
        return self[sortedinds]

    def in_ms(self):
        return self / self.fs * 1000

    def in_samples(self, fs=None):
        if fs is None:
            if not self.parent_fs == None:
                return self / self.fs * self.parent_fs
            else:
                warnings.warn('No data sampling rate provided, assuming %.2fHz for Markers and data' % self.fs)
                return self
        else:
            return self / self.fs * fs

    def get_class_inds(self, classes):
        classes_int = []
        for ii in classes:
            inds = np.where(self.className == ii)[0]
            if inds >= 0:
                classes_int.append(inds)
            else:
                warnings.warn('Class "%s" not found.' % ii)
        return np.array(classes_int)[:, 0]

    def __select_classes__(self, classes):
        if isinstance(classes, str):  # single class string selection needs to be list for get_class_inds
            classes = [classes]
        if isinstance(classes, list) and isinstance(classes[0], str):
            classes = self.get_class_inds(classes)

        if type(classes) == int:
            selected = self.y == classes
            newy = self.y
            newclassName = self.className
        else:
            selected = np.zeros(self.shape, dtype=bool)
            newy = self.y.copy()
            for ii, ic in enumerate(classes):
                selected |= self.y == ic  # select relevant markers
                newy[self.y == ic] = ii  # reorder indices
            newclassName = self.className[classes]
        return selected, newy, newclassName

    def select_classes(self, classes):
        obj = self.copy()
        selected, obj.y, obj.className = self.__select_classes__(classes)
        obj = obj[selected]
        return obj

# class SRM_Marker(np.ndarray):
#
#     def __new__(cls, mrk_class, mrk_class_name, trialresult, mrk_fs=1):
#         obj = np.asarray(mrk_class).view(cls)
#         obj.y = np.array(mrk_class)
#         obj.className = mrk_class_name
#         obj.trialresult = np.array(trialresult)
#         obj.fs = mrk_fs
#         return obj
#
#     def __init__(self, mrk_class, mrk_class_name, trialresult, mrk_fs=1):
#         return  # super().init()
#
#     def __array_wrap__(self, out_arr, context=None):  # used for printing,squeezing etc
#         # return self.__class__(self, *self.__initargs__())
#         obj = super().__array_wrap__(out_arr, context)
#         obj.y = np.copy(self.y)
#         obj.trialresult = np.copy(self.trialresult)
#         obj.className = np.copy(self.className)
#         obj.fs = np.copy(obj.fs)
#         return obj
#
#     def __array_finalize__(self, obj, *args, **kwargs):
#         self.y = getattr(obj, 'y', None)
#         self.trialresult = getattr(obj, 'trialresult', None)
#         self.className = getattr(obj, 'className', None)
#         self.fs = getattr(obj, 'fs', None)
#
#     def __getitem__(self, key):
#         # TODO: add support for slicing
#         newy = self.y[key].copy()
#         if isinstance(key, int) or ((not isinstance(key, slice)) and len(key) == 1):
#             newy = [newy]
#         leftclasses = np.unique(self.y[key])
#         if len(leftclasses) < len(self.className):
#             newind = 0
#             for i in range(len(self.className)):
#                 if i in leftclasses:
#                     newy[newy == i] = newind
#                     newind += 1
#
#         # FIXME: not sure if this is correct: walkaround
#         newclassName = [self.className[lc] for lc in leftclasses.astype(int)]
#         obj = SRM_Marker(newy, newclassName, self.trialresult, self.fs)
#         return obj
#
#     def get_valid_trials(self):
#         """
#         get trials that are not timeouts
#         :return:
#         :rtype:
#         """
#         valid_trials = np.where(~np.isnan(self.trialresult))[0]
#         return valid_trials
#
#     def get_class_inds(self, classes):
#         """
#         tasks in srm datasett are labeled "R": 1, "L": 2, "U": 3, "D": 4
#         :param task:
#         :type task:
#         :return:
#         :rtype:
#         """
#         classes_int = []
#         valid_trials = self.get_valid_trials()
#         for class_name in classes:
#             class_index = target_map_dict[class_name]
#             inds_t = np.where(self.y == class_index)[0]
#
#             # remove all trials that are timeouts or not valid
#             inds = np.intersect1d(inds_t, valid_trials)
#
#             if len(inds) >= 0:
#                 classes_int.append(inds)
#             else:
#                 warnings.warn('Class "%s" not found.' % class_name)
#
#         return classes_int
#
#     def __select_classes__(self, classes):
#         if isinstance(classes, str):  # single class string selection needs to be list for get_class_inds
#             classes = [classes]
#         if isinstance(classes, list) and isinstance(classes[0], str):
#             classes = self.get_class_inds(classes)
#
#         if type(classes) == int:
#             selected = self.y == classes
#             newy = self.y
#             newclassName = self.className
#         else:
#             dim = sum([len(c) for c in classes])
#             selected = np.zeros(self.shape, dtype=bool)
#             # newy = self.y.copy()
#             # newy = np.zeros(self.shape, dtype=int)
#             newy = np.full(self.shape, np.nan)
#             for ii, ic in enumerate(classes):
#                 selected[ic] = True
#                 # selected |= self.y == ic  # select relevant markers
#                 # newy[self.y == ic] = ii  # reorder indices
#                 newy[ic] = ii
#             # remove all trials that has zero label
#             # Create a mask of NaN values
#             mask = np.isnan(newy)
#             # Filter the array using the mask
#             newy = newy[~mask]
#             newclassName = [self.className[c] for c in np.unique(newy).astype(int)]
#         return selected, newy, newclassName
#
#     def select_classes(self, classes):
#         obj = self.copy()
#         selected, obj.y, obj.className = self.__select_classes__(classes)
#         # obj = obj[selected]
#         return obj
#
#     def in_samples(self, fs=None):
#         if fs is None:
#             return self
#         else:
#             return self / self.fs * fs
#
#     def in_ms(self):
#         return self / self.fs * 1000
