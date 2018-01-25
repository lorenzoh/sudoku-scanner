import cv2
import numpy as np


star_kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)


def to_grayscale(img):
    if img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def threshold(img, options={}):
    blur_size = options.get('blur_1_size', 13)

    img = cv2.GaussianBlur(img, ksize=(blur_size, blur_size), sigmaX=0)
    img = cv2.adaptiveThreshold(
        img, maxValue=255, adaptiveMethod=0, thresholdType=1, blockSize=11, C=3)

    img = cv2.dilate(img, kernel=star_kernel)

    return img


def expose_grid(img, options={}):
    inverse = 255 - img
    mask = get_mask_like(img)
    dim = min(inverse.shape)

    biggest_area = 0

    # looks for objects in the image, saving location of the largest
    for x in range(dim // 4, 3 * (dim // 4)):
        if inverse[x, x] == 0:
            area, img, _, corners = cv2.floodFill(
                inverse, mask, seedPoint=(x + 1, x + 1), newVal=64)
            if area > biggest_area:
                biggest_area = area
                point = (x, x)

    # removes everything but the objects
    img[img != 64] = 0

    # removes every object but the biggest
    mask = get_mask_like(img)
    area, img, mask, corners = cv2.floodFill(
        inverse, mask, seedPoint=point, newVal=255)
    img[img != 255] = 0

    img = cv2.erode(img, kernel=star_kernel)

    return img


def find_corners(img, options={}):

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

def get_mask_like(img):
    """
    Constructs an array of zeros like the input image with one field padding on
    every side.
    (Used for some OpenCV functions)
    """
    return np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)
