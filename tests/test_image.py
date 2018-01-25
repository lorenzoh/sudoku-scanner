import sys
import os
import cv2
import numpy as np

# Insert project code into path for importability
module_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, module_path)

import sudoku
import imageprocessing


test_data_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 'testing_data'))

def load_test_img():
    return cv2.imread(os.path.join(test_data_path, 'image1.jpg'))


def test_to_grayscale():
    img = get_test_img()
    assert img.ndim == 3
    assert imageprocessing.to_grayscale(img).ndim == 2


def test_get_mask_like():
    assert imageprocessing.get_mask_like(np.ones((8, 8))) == np.zeros((10, 10))
