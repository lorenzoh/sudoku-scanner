# !/bin/python
"""
Utilities for files
"""
import os
from typing import List, Union


def get_filenames(
        folder: str,
        fullpath: bool = True,
        sort: bool = True,
        extensions: Union[None, List[str]] = None) -> List[str]:
    """Gets all files in folder and filters/transforms them"""
    filenames = os.listdir(folder)
    if extensions:
        filenames = [filename for filename in filenames if has_extensions(filename, extensions)]
    if sort:
        filenames = sort_filenames(filenames)
    if fullpath:
        filepaths = [os.path.join(folder, filename)
                     for filename in filenames]
        return filepaths
    return filenames


def has_extensions(filename: str, extensions: List[str]) -> bool:
    """Checks if a file has one of several extensions"""
    return filename.split(sep='.')[-1] in extensions


def sort_filenames(filenames: List[str]) -> List[str]:
    """Sorts a list of file names based on a number in the file name"""
    return sorted(filenames, key=lambda x: int(''.join([c for c in x if c in '0123456789'])))

def parse_dat(dat_path: str) -> List[int]:
    """Parses ground truth .dat file from original dataset"""
    with open(dat_path) as file:
        lines = file.readlines()
    digits = [int(x) for x in ''.join(lines[2:]) if x in '0123456789']
    assert len(digits) == 81
    return digits
