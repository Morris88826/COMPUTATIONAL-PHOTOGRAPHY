import numpy as np
import matplotlib.pyplot as plt
from helper import crop_image

# Functions for extra credits
def auto_cropping(image):
    plt.imshow(image)
    plt.show()
    raise NotImplementedError
    return

def auto_contrasting(image):
    min_val = np.min(image)
    max_val = np.max(image)
    image = (image - min_val)/(max_val - min_val)
    return image

def auto_white_balancing(image):
    raise NotImplementedError
    return


def preprocess_image(image, border=None, auto_crop=False, auto_contrast=False):
    if auto_crop:
        image = auto_cropping(image)
    elif border is not None:
        image = crop_image(image, border)
    else:
        raise ValueError("Either auto_crop or border should be provided")

    if auto_contrast:
        image = auto_contrasting(image)
    return image
    