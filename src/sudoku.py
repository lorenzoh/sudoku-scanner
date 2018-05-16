"""
Provides the Sudoku class
"""
import os
from pathlib import Path

import numpy as np
import cv2
from keras.models import load_model

import imageprocessing as imgprcs

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
MODEL_FILE = os.path.abspath(os.path.join(__file__, '../../models/modelv2.h5'))


class Sudoku:
    """Holds information and data on one sudoku image"""

    def __init__(self, img_path, save_steps=False):
        self.img_path = Path(img_path).absolute()

        if not self.img_path.name.split(sep='.')[-1] == 'jpg':
            raise ValueError(f'{self.img_path.name} is not a .jpg image')
        if not self.img_path.exists():
            raise OSError(f'Image at {img_path} not found')

        raw_img = cv2.imread(img_path)
        self.resized = imgprcs.resize(raw_img)
        self.corners = None
        self.digits = None
        self.predictions = None
        self.steps = [] if save_steps else None

    def process(self):
        """
        Applies several steps of processing to sudoku image to separate
        out the digits.
        """
        img = self.resized.copy()

        # Step 1: Process the image
        grayscale = imgprcs.to_grayscale(img)

        img = imgprcs.threshold(grayscale.copy())
        if self.steps is not None:
            self.steps.append(img.copy())

        img = imgprcs.expose_grid(img)
        if self.steps is not None:
            self.steps.append(img.copy())

        self.corners = imgprcs.find_corners(img)

        # Step 2: Transform

        img = imgprcs.transform(grayscale.copy(), self.corners)
        if self.steps is not None:
            self.steps.append(img.copy())

        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 19, 5)
        if self.steps is not None:
            self.steps.append(img.copy())

        # Step 3: Get and clean the digits

        self.digits = imgprcs.find_digits(img)

    def predict(self, model_file=MODEL_FILE):
        """
        Loads a Keras model with input (None, 1024) and saves digit
        predictions to attribute digits.
        """
        model = load_model(model_file, compile=False)

        self.predictions = np.array(
            [0 if digit is None else -1 for digit in self.digits])

        digits_arr = np.array(
            [digit for digit in self.digits if digit is not None])
        digits_arr = digits_arr.reshape((digits_arr.shape[0], -1))
        predictions = model.predict(digits_arr)
        self.predictions[self.predictions != 0] = np.argmax(
            predictions, axis=1) + 1

    def get_predictions(self):
        """ Returns: String of 81 digits detected in image"""
        return ''.join(map(str, self.predictions))


    def __str__(self):
        pred_array = np.array(self.predictions).reshape((9, 9))
        string = ''
        for y in range(9):
            for x in range(9):
                string += '{} '.format(pred_array[y, x])
                if x in (2, 5):
                    string += '|'
            if y in (2, 5):
                string += '\n' + '—' * 19
            string += '\n'
        string.replace('0', ' ')
        return string
