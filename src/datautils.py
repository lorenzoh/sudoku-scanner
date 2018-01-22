import os


def get_filenames(folder, fullpath=True, sort=True, extensions=None):
    filenames = os.listdir(folder)
    if extensions:
        filenames = [filename for filename in filenames if has_extensions(filename, extensions)]
    if sort:
        filenames = sort_filenames(filenames)
    if fullpath:
        filepaths = [os.path.join(folder, filename)
                     for filename in filenames]
        return filepaths
    else:
        return filenames


def has_extensions(filename, extensions):
    """Checks if a file has one of several extensions"""
    return filename.split(sep='.')[-1] in extensions


def sort_filenames(filenames):
    """Sorts a list of file names based on a number in the file name"""
    return sorted(filenames, key=lambda x: int(''.join([c for c in x if c in '0123456789'])))
