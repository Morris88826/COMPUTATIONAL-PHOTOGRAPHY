import cv2
import numpy as np
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

    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    # first remove shifts
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

    # then remove white borders
    upper = 95
    mask_white = (r >= np.percentile(r, upper)) | (g >= np.percentile(g, upper)) | (b >= np.percentile(b, upper))
    mask_white = np.logical_not(mask_white)

    top = np.median(np.argmax(mask_white, axis=0))
    bottom = np.median(mask_white.shape[0] - np.argmax(mask_white[::-1], axis=0))
    left = np.median(np.argmax(mask_white, axis=1))
    right = np.median(mask_white.shape[1] - np.argmax(mask_white[::-1], axis=1))

    finalImage = np.stack([r, g, b], axis=2)
    finalImage = finalImage[int(top):int(bottom), int(left):int(right), :]

    # remove the black borders
    lower = 5
    r = finalImage[:, :, 0]
    g = finalImage[:, :, 1]
    b = finalImage[:, :, 2]
    mask_black = (r <= np.percentile(r, lower)) | (g <= np.percentile(g, lower)) | (b <= np.percentile(b, lower))
    mask_black = np.logical_not(mask_black)

    top = np.median(np.argmax(mask_black, axis=0))
    bottom = np.median(mask_black.shape[0] - np.argmax(mask_black[::-1], axis=0))
    left = np.median(np.argmax(mask_black, axis=1))
    right = np.median(mask_black.shape[1] - np.argmax(mask_black[::-1], axis=1))

    finalImage = np.stack([r, g, b], axis=2)
    finalImage = finalImage[int(top):int(bottom), int(left):int(right), :]


    return finalImage

def auto_contrasting(image):

    percentiles = [5, 95]
    # find the minimum and maximum value of the image
    min_val = np.percentile(image, percentiles[0])
    max_val = np.percentile(image, percentiles[1])

    # normalize the image
    image = (image - min_val) / (max_val - min_val)

    # clip the values to be between 0 and 1
    image = np.clip(image, 0, 1)

    return image

def auto_white_balancing(image, rShift, gShift):

    # Find the borders of the image
    image = auto_cropping(image, rShift, gShift)

    # Calculate the illumination of the image
    illumination = np.mean(image, axis=(0, 1))
    neutral_gray = 0.5

    # Normalize the image
    image = image / illumination * neutral_gray
    
    # Clip the values to be between 0 and 1
    image = np.clip(image, 0, 1)

    return image


def preprocess_image(image, border, better_features=False):
    # crop the image
    image = crop_image(image, border)

    if better_features:
        image = find_edge(image)
        return image
    
    return image
    