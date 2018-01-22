import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import cv2


def show_img(img, opencv=True, figsize=(8, 6)):
    plt.figure(figsize=figsize)
    plt.axis('off')
    if img.shape[-1] != 3:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.show()
