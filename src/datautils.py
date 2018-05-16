"""
Utilities for handling the data from
https://github.com/wichtounet
"""


def parse_dat(dat_path):
    """Parses ground truth .dat file from original dataset"""
    with open(dat_path) as file:
        lines = file.readlines()
    digits = [int(x) for x in ''.join(lines[2:]) if x in '0123456789']
    assert len(digits) == 81
    return digits
