import sys
import os

# insert project code into path from importability
module_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../src'))
print(module_path)
sys.path.insert(0, module_path)

import sudoku
import data_utils


def test_is_image_file_works():
    assert data_utils.is_image_file('test.jpg')
    assert not data_utils.is_image_file('testjpg')
    assert data_utils.is_image_file('test.jpg.jpg')
