"""Functions for processing sudoku images"""
import cv2
import numpy as np


star_kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)


def resize(img, max_side=640):
    """Resizes the image's largest side to 640 and the other proportionally."""
    factor = max_side / max(img.shape)
    img = cv2.resize(img.copy(), None, fx=factor, fy=factor,
                     interpolation=cv2.INTER_AREA)
    return img


def to_grayscale(img):
    """Converts a colored image to grayscale"""
    if img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def threshold(img, blur_size=13):
    """Applies a blur and thresholding to a grayscale image."""

    img = cv2.GaussianBlur(img, ksize=(blur_size, blur_size), sigmaX=0)
    img = cv2.adaptiveThreshold(
        img, maxValue=255, adaptiveMethod=0, thresholdType=1, blockSize=11, C=3)

    img = cv2.dilate(img, kernel=star_kernel)

    return img


def expose_grid(img):
    """
    Finds the biggest connected part in the image, the grid, and removes
    everything else from the images.
    """
    inverse = 255 - img
    mask = get_mask_like(img)
    size = min(inverse.shape)

    biggest_area = 0

    # looks for objects in the image, saving location of the largest
    for x in range(size // 4, 3 * (size // 4)):
        if inverse[x, x] == 0:
            area, img, _, _ = cv2.floodFill(
                inverse, mask, seedPoint=(x + 1, x + 1), newVal=64)
            if area > biggest_area:
                biggest_area = area
                point = (x, x)

    # removes everything but the objects
    img[img != 64] = 0

    # removes every object but the biggest
    mask = get_mask_like(img)
    area, img, _, _ = cv2.floodFill(
        inverse, mask, seedPoint=point, newVal=255)
    img[img != 255] = 0

    img = cv2.erode(img, kernel=star_kernel)

    return img


def find_corners(img):
    """
    Finds corners of the grid by sliding diagonal lines midwards from
    all corners.
    """
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

    return corners


def transform(img, corners, size=288):
    """Zooms into image using corners."""
    assert (size % 9) == 0
    boundary = np.float32([[0, 0], [size, 0], [0, size], [size, size]])
    M = cv2.getPerspectiveTransform(corners.astype(np.float32), boundary)

    img = cv2.warpPerspective(img, M, (size, size))

    return img


def find_digits(img):
    """
    Divides grid into cells, finds and cleans the numbers and saves
    them to a list.
    """
    size = img.shape[0]
    digit_size = size // 9

    digits = []

    for y in range(0, size, digit_size):
        for x in range(0, size, digit_size):
            digit = img[y:(y + digit_size), x:(x + digit_size)].copy()
            digit = clean_digit(digit)

            digits.append(digit)

    return digits


def clean_digit(img):
    """
    Looks for a big blob in the middle. If it finds one, assumes its
    the digit and cleans everything else from the cell
    """
    is_empty = True
    size = img.shape[0]
    min_area = size
    max_area = size * 10
    center = size // 2
    max_center_dist = (3 * size) // 16
    mask = np.zeros((size + 2, size + 2), np.uint8)
    for x in range(max_center_dist):
        for i in range(2):
            if i == 0:
                pnt = (center + x, center + x)
            elif i == 1:
                pnt = (center - x, center - x)
            if img[pnt] == 0:
                area, *_ = cv2.floodFill(img, mask, pnt, 128)
                if area < max_area and area > min_area:
                    is_empty = False
                    break
        if not is_empty:
            break
    if is_empty:
        return None

    img[img == 0] = 255
    mask = np.zeros((size + 2, size + 2), np.uint8)
    area, *_ = cv2.floodFill(img, mask, pnt, 0)
    img[img == 128] = 255

    img = center_digit(img)

    return img


def center_digit(img):
    """Centers the digit to ease inference."""
    borders = find_borders(img)
    invert = 255 - img
    x_shift = (borders[3] - borders[1]) // 2
    y_shift = (borders[2] - borders[0]) // 2
    M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    invert = cv2.warpAffine(invert, M, (invert.shape[1], invert.shape[0]))
    img = 255 - invert
    return img


def find_borders(img):
    """Finds borders of the number by sliding lines midwards from all sides"""
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


def get_mask_like(img):
    """
    Constructs an array of zeros like the input image with one field padding on
    all four sides.
    (Used for some OpenCV functions)
    """
    return np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)
