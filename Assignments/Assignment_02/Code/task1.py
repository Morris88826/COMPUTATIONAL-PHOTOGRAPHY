import numpy as np
from helper import circ_shift

# The main part of the code. Implement the FindShift function
def find_shift(im1, im2, search_range = 20):
    min_ssd = np.Inf

    for i in range(-search_range, search_range + 1):
        for j in range(-search_range, search_range + 1):
            # shifting the image
            shifted_im1 = circ_shift(im1, (i, j))
            
            # calculating the SSD
            ssd = np.sum((shifted_im1 - im2) ** 2)

            if ssd < min_ssd:
                min_ssd = ssd
                min_d_y = i
                min_d_x = j

    return (min_d_y, min_d_x)