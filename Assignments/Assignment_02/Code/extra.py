import cv2
import numpy as np
import matplotlib.pyplot as plt
from helper import crop_image


def find_edge(image):
    # use the Canny edge detector to find the edges
    
    # Convert to 8-bit
    image = np.clip(image * 255, 0, 255).astype(np.uint8)

    # apply GaussianBlur to reduce noise
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # find dynamic threshold
    sigma = 0.33
    v = np.median(blurred_image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(image, lower, upper)
    edges = edges/255

    return edges

# Functions for extra credits
def auto_cropping(image, rShift, gShift):

    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    # crop the image
    # horizontal shift
    if rShift[1] < 0 and gShift[1] < 0:
        max_shift = max(abs(rShift[1]), abs(gShift[1]))
        r = r[:, :max_shift]
        g = g[:, :max_shift]
        b = b[:, :max_shift]

    elif rShift[1] >= 0 and gShift[1] >= 0:
        max_shift = max(rShift[1], gShift[1])
        r = r[:, max_shift:]
        g = g[:, max_shift:]
        b = b[:, max_shift:]
    
    elif rShift[1] >= 0 and gShift[1] < 0:
        r = r[:, rShift[1]:gShift[1]]
        g = g[:, rShift[1]:gShift[1]]
        b = b[:, rShift[1]:gShift[1]]

    elif rShift[1] < 0 and gShift[1] >= 0:
        r = r[:, gShift[1]:rShift[1]]
        g = g[:, gShift[1]:rShift[1]]
        b = b[:, gShift[1]:rShift[1]]

    # vertical shift
    if rShift[0] < 0 and gShift[0] < 0:
        max_shift = max(abs(rShift[0]), abs(gShift[0]))
        r = r[:max_shift, :]
        g = g[:max_shift, :]
        b = b[:max_shift, :]

    elif rShift[0] >= 0 and gShift[0] >= 0:
        max_shift = max(rShift[0], gShift[0])
        r = r[max_shift:, :]
        g = g[max_shift:, :]
        b = b[max_shift:, :]
    
    elif rShift[0] >= 0 and gShift[0] < 0:
        r = r[rShift[0]:gShift[0], :]
        g = g[rShift[0]:gShift[0], :]
        b = b[rShift[0]:gShift[0], :]

    elif rShift[0] < 0 and gShift[0] >= 0:
        r = r[gShift[0]:rShift[0], :]
        g = g[gShift[0]:rShift[0], :]
        b = b[gShift[0]:rShift[0], :]

    return np.stack((r, g, b), axis=2)

def auto_contrasting(r, g, b):
    # find the minimum and maximum value of the image
    min_val = min(np.min(r), np.min(g), np.min(b))
    max_val = max(np.max(r), np.max(g), np.max(b))

    # normalize the image
    r = (r - min_val) / (max_val - min_val)
    g = (g - min_val) / (max_val - min_val)
    b = (b - min_val) / (max_val - min_val)

    # clip the values to be between 0 and 1
    r = np.clip(r, 0, 1)
    g = np.clip(g, 0, 1)
    b = np.clip(b, 0, 1)

    return r, g, b

def auto_white_balancing(image):
    raise NotImplementedError
    return


def preprocess_image(image, border, better_features=False):
    # crop the image
    image = crop_image(image, border)

    if better_features:
        image = find_edge(image)
        return image
    
    return image
    