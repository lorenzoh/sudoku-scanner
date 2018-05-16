"""Tests for the data utilities"""
import sys
import os

# Insert project code into path for importability
module_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, module_path)

import datautils

test_data_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 'testing_data'))


def test_parse_dat():
    assert datautils.parse_dat(os.path.join(test_data_path, 'image1000.dat')) == [
        0, 0, 6, 0, 7, 0, 0, 0, 0,
        0, 4, 0, 0, 0, 0, 0, 0, 7,
        0, 7, 0, 5, 0, 0, 2, 1, 0,
        0, 0, 8, 0, 5, 0, 0, 0, 1,
        0, 2, 1, 0, 0, 0, 4, 3, 0,
        6, 0, 0, 0, 3, 0, 7, 0, 0,
        0, 6, 3, 0, 0, 8, 0, 4, 0,
        1, 0, 0, 0, 0, 0, 0, 9, 0,
        0, 0, 0, 1, 0, 0, 6, 0, 0]
