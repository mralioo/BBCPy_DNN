import os

def pjoin(*args, **kwargs):
    '''Join path components intelligently for the target platform. This is needed to avoid problems if the script runs under windows and the target platform is linux.'''
    return os.path.join(*args, **kwargs).replace(os.path.sep, '/')