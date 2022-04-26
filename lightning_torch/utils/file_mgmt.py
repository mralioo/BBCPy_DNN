import os
import sys
from typing import Any, Mapping, Optional, Sequence, Tuple, Union


def get_dir_by_indicator(path=sys.path[0], indicator=".git") -> str :
    """Returns the path of the folder that contains the given indicator

    Args:
        path [str]: Path from where to search the directory tree upwards.
        indicator [str]:  Name of the file that indicates the searched directory.

    Returns:
        path [str]: Relative path where the indicator file located.
    """

    is_root = os.path.exists(os.path.join(path, indicator))
    while not is_root:
        new_path = os.path.dirname(path)
        if new_path == path:
            raise FileNotFoundError(
                "Could not find folder containing indicator {:} in any path or any toplevel directory.".format(
                    indicator))
        path = new_path
        is_root = os.path.exists(os.path.join(path, indicator))

    return path

if __name__ == "__main__":
    r = get_dir_by_indicator(indicator="ROOT")