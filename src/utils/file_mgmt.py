import os
import numpy as np

def pjoin(*args, **kwargs):
    '''Join path components intelligently for the target platform. This is needed to avoid problems if the script runs under windows and the target platform is linux.'''
    return os.path.join(*args, **kwargs).replace(os.path.sep, '/')

def default(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
        return {'real': obj.real, 'imag': obj.imag}
    raise TypeError(f"Object of type '{type(obj).__name__}' is not JSON serializable")

