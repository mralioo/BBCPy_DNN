import os
import numpy as np

def pjoin(*args, **kwargs):
    '''Join path components intelligently for the target platform. This is needed to avoid problems if the script runs under windows and the target platform is linux.'''
    return os.path.join(*args, **kwargs).replace(os.path.sep, '/')

def default(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type '{type(obj).__name__}' is not JSON serializable")
