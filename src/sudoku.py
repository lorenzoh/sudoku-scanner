import os

import numpy as np

import cv2
import vis
import datautils


class Sudoku:
    """
    Central class that holds image data and is responsible for transforming it.
    Useful for inspecting the processing pipeline at every stage
    """

    def __init__(self, img_path, preprocessing_options={}):
        self.preprocessing_options = preprocessing_options
        self.img_path = os.path.abspath(img_path)
        assert datautils.has_extensions(self.img_path, extensions=['jpg'])

        self.raw_img = cv2.imread(img_path)
        self.resized = self.resize(self.raw_img)

        self.preprocessed = self.resized.copy()
        self.grid = None
        self.processed_grid = None
        self.corners = None

        self.is_preprocessed = False
        self.is_transformed = False
        self.is_grid_processed = False

        dat_path = ''.join(self.img_path.split(sep='.')[:-1]) + '.dat'
        if os.path.isfile(dat_path):
            self.true_digits = datautils.parse_dat(dat_path)

    def resize(self, img, max_side=640):
        factor = max_side / max(img.shape)
        img = cv2.resize(img.copy(), None, fx=factor, fy=factor,
                         interpolation=cv2.INTER_AREA)
        return img

    def show_raw(self, figsize=(8, 6)):
        vis.show_img(self.resized, figsize=figsize)

    def show_processed(self, figsize=(8, 6)):
        if not self.is_preprocessed:
            self.preprocess()
        vis.show_img(self.preprocessed, figsize=figsize)

    def show_corners(self, figsize=(8, 6)):
        if not type(self.corners) == np.ndarray:
            self.find_corners()

        img = self.resized.copy()
        for x, y in self.corners:
            cv2.circle(img, (x, y), 5, (255, 0, 0), 10)
        vis.show_img(img, figsize=figsize)

    def show_grid(self, figsize=(8, 6)):
        if not self.is_transformed:
            self.transform()
        vis.show_img(self.grid,  figsize=figsize)

    def preprocess(self):
        """
        STAGE 1
        Applies multiple steps of preprocessing to image so that corners can be
        found
        """
        self.threshold()
        self.expose_grid()

        self.is_preprocessed = True

    def threshold(self):
        """
        STAGE 1.1
        Converts image to grayscale, blurs, thresholds and then rids it of noise
        using dilation
        """
        # Get parameters
        blur_size = self.preprocessing_options.get('blur_size', 13)
        img = self.preprocessed

        # Convert to grayscale
        if img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Blur
        img = cv2.GaussianBlur(img, (blur_size, blur_size), 0)

        img = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 3)
        # Star shaped kernel
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)

        self.preprocessed = img

    def expose_grid(self):
        """
        STAGE 1.2
        Uses flood fill to determine the largest object in the image (the
        sudoku) and removes everything else from the images leaving just the
        sudoku grid
        """
        img = self.preprocessed

        max_area = 0
        inv = 255 - img
        mask = np.zeros((inv.shape[0] + 2, inv.shape[1] + 2), np.uint8)
        dim = min(inv.shape)
        for x in range(dim // 4, dim // 4 * 3):
            if inv[x, x] == 0:
                area, flood, _, corners = cv2.floodFill(
                    inv, mask, (x + 1, x + 1), 64)
                if area > max_area:
                    max_area = area
                    max_point = (x, x)
        flood[flood != 64] = 0
        mask = np.zeros((inv.shape[0] + 2, inv.shape[1] + 2), np.uint8)
        area, flood, mask, corners = cv2.floodFill(inv, mask, max_point, 255)
        flood[flood != 255] = 0
        kernel = np.array([0, 1, 0, 1, 1, 1, 0, 1, 0],
                          np.uint8).reshape((3, 3))
        flood = cv2.erode(flood, kernel)

        self.preprocessed = flood

    def find_corners(self):
        """
        STAGE 2
        Finds corners of the sudoku by using preprocessed image
        """
        if not self.is_preprocessed:
            self.preprocess()
        img = self.preprocessed

        height, width = img.shape[0] - 1, img.shape[1] - 1
        corners = np.zeros((4, 2), np.int32)
        LU_not, RU_not = True, True
        LD_not, RD_not = True, True
        for dist in range(min(img.shape)):
            for x in range(dist + 1):
                y = dist - x
                if img[y, x] == 255 and LU_not:
                    corners[0] = (x, y)
                    LU_not = False
                if img[y, width - x] == 255 and RU_not:
                    corners[1] = (width - x, y)
                    RU_not = False
                if img[height - y, x] == 255 and LD_not:
                    corners[2] = (x, height - y)
                    LD_not = False
                if img[height - y, width - x] == 255 and RD_not:
                    corners[3] = ((width - x), (height - y))
                    RD_not = False
            if not (LU_not or RU_not or LD_not or RD_not):
                break

        self.corners = corners

    def transform(self):
        """
        STAGE 3
        Transforms image so that the sudoku fills the whole image
        """
        if not self.corners:
            self.find_corners()

        img = self.resized

        size = self.preprocessing_options.get('transform_size', 288)
        boundary = np.float32([[0, 0], [size, 0], [0, size], [size, size]])
        M = cv2.getPerspectiveTransform(
            self.corners.astype(np.float32), boundary)

        self.grid = cv2.warpPerspective(img, M, (size, size))
        self.is_transformed = True

    def preprocess_transformed(self):
        """
        STAGE 4
        Preprocesses the transformed puzzle to make it easier to extract digits
        """
        if not self.is_transformed:
            self.transform()

        img = self.grid.copy()

        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 19, 5)

        self.processed_grid  = img

    def preprocess_digits(self):
        if not self.is_grid_processed:
            self.preprocess_transformed()

        img = self.processed_grid
        print('Processed grid:', img.shape)
        grid_size = img.shape[0]
        digit_size = grid_size // 9
        self.cleaned = img.copy()
        for x in range(0, grid_size, digit_size):
            for y in range(0, grid_size, digit_size):
                digit = img[y:(y + digit_size), x:(x + digit_size)]
                cleaned_digit, prediction = find_number(digit.copy())

                cleaned_digit, prediction = center_number(digit)
                vis.show_img(cleaned_digit)
                vis.show_img(digit)
                self.cleaned[y:(y + digit_size), x:(x + digit_size)] = cleaned_digit


def find_number(img):
    number = False
    dim = img.shape[0]
    min_area = dim
    max_area = dim * 10
    center = dim // 2
    max_center_dist = (3 * dim) // 16
    mask = np.zeros((dim+2, dim+2), np.uint8)
    for x in range(max_center_dist):
        for i in range(2):
            if i == 0:
                pnt = (center + x, center + x)
            elif i == 1:
                pnt = (center - x, center - x)
            if img[pnt] == 0:
                area, flood, _, corners = cv2.floodFill(img, mask, pnt, 128)
                if area < max_area and area > min_area:
                    number = True
                    break
        if number:
            break
    if not number:
        img[:, :] = 255
        return img, 0
    else:
        img[img == 0] = 255
        mask = np.zeros((dim+2, dim+2), np.uint8)
        area, flood, _, corners = cv2.floodFill(img, mask, pnt, 0)
        img[img == 128] = 255
        return img, -1


def find_borders(img):
    dim = img.shape[0] - 1
    top_border = 0

    for y in range(dim):
        if 0 in img[y, :]:
            break
        top_border += 1
    bottom_border = 0
    for y in range(dim):
        if 0 in img[dim - y, :]:
            break
        bottom_border += 1
    left_border = 0
    for x in range(dim):
        if 0 in img[:, x]:
            break
        left_border += 1
    right_border = 0
    for x in range(dim):
        if 0 in img[:, dim - x]:
            break
        right_border += 1

    return (top_border, left_border, bottom_border, right_border)


def center_number(img):
    borders = find_borders(img)
    invert = 255 - img
    x_shift = (borders[3] - borders[1]) // 2
    y_shift = (borders[2] - borders[0]) // 2
    M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    invert = cv2.warpAffine(invert, M, (invert.shape[1], invert.shape[0]))
    img = 255 - invert
    return img, -1


def preprocess_digit(img):
    img, prediction = find_number(img)
    if prediction == -1:
        img, prediction = center_number(img)
    return img, prediction
