import sys
import os

# Insert project code into path for importability
module_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, module_path)

import datautils

test_data_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 'testing_data'))


def test_has_extensions():
    assert datautils.has_extensions('image.jpg', ['jpg'])
    assert datautils.has_extensions('image.jpg.png.gif', ['gif'])
    assert not datautils.has_extensions('test.txt', ['gif'])


def test_sort_filenames():
    assert datautils.sort_filenames(['2.txt', '1.txt']) == ['1.txt', '2.txt']
    assert datautils.sort_filenames(
        ['a/b/c/2.d', 'a/b/c/1.d']) == ['a/b/c/1.d', 'a/b/c/2.d']


def test_parse_dat():
    assert datautils.parse_dat(os.path.join(test_data_path, 'image1.dat')) == [0, 0, 0, 7, 0, 0, 0, 8, 0,
                                                                               0, 9, 0, 0, 0, 3, 1, 0, 0,
                                                                               0, 0, 6, 8, 0, 5, 0, 7, 0,
                                                                               0, 2, 0, 6, 0, 0, 0, 4, 9,
                                                                               0, 0, 0, 2, 0, 0, 0, 5, 0,
                                                                               0, 0, 8, 0, 4, 0, 0, 0, 7,
                                                                               0, 0, 0, 9, 0, 0, 0, 3, 0,
                                                                               3, 7, 0, 0, 0, 0, 0, 0, 6,
                                                                               1, 0, 5, 0, 0, 4, 0, 0, 0]
