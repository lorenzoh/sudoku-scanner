import os

import numpy as np
import cv2
from keras.models import load_model

import vis
import datautils
import imageprocessing as imgprcs


class Sudoku:

    def __init__(self, img_path, save_steps=False):
        self.img_path = os.path.abspath(img_path)
        assert datautils.has_extensions(self.img_path, extensions=['jpg'])

        self.raw_img = cv2.imread(img_path)
        self.resized = imgprcs.resize(self.raw_img)

        self.steps = [] if save_steps else None

        dat_path = ''.join(self.img_path.split(sep='.')[:-1]) + '.dat'
        if os.path.isfile(dat_path):
            self.true_digits = datautils.parse_dat(dat_path)


    def process(self, processing_options={}):
        img = self.resized.copy()

        # Step 1: Process the image
        grayscale = imgprcs.to_grayscale(img)

        img = imgprcs.threshold(grayscale.copy())
        if self.steps is not None: self.steps.append(img.copy())

        img = imgprcs.expose_grid(img)
        if self.steps is not None: self.steps.append(img.copy())

        self.corners = imgprcs.find_corners(img)

        # Step 2: Transform

        img = imgprcs.transform(grayscale.copy(), self.corners)
        if self.steps is not None: self.steps.append(img.copy())

        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 19, 5)
        if self.steps is not None: self.steps.append(img.copy())

        # Step 3: Get and clean the digits

        self.digits = imgprcs.find_digits(img)


    def predict(self, model_file=os.path.abspath('../models/modelv2.h5')):
        model = load_model(model_file, compile=False)

        self.predictions = np.array([0 if digit is None else -1 for digit in self.digits])

        digits_arr = np.array([digit for digit in self.digits if digit is not None])
        digits_arr = digits_arr.reshape((digits_arr.shape[0], -1))
        predictions = model.predict(digits_arr)
        self.predictions[self.predictions != 0] = np.argmax(predictions, axis=1) + 1
