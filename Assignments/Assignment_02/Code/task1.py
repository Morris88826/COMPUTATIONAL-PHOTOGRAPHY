import numpy as np
from helper import circ_shift

def main(im1, im2, search_range=20, border = 20):
    cropped_im1 = im1[border:-border, border:-border]
    cropped_im2 = im2[border:-border, border:-border]
    return find_shift(cropped_im1, cropped_im2, search_range)

# The main part of the code. Implement the FindShift function
def find_shift(im1, im2, search_range = 20):
    min_ssd = np.Inf

    for i in range(-search_range, search_range + 1):
        for j in range(-search_range, search_range + 1):
            # shifting the image
            shifted_im1 = circ_shift(im1, (i, j))
            
            # # cropping the invalid border pixels
            # if i >= 0 and j >= 0:
            #     cropped_im1 = shifted_im1[i:, j:]
            #     cropped_im2 = im2[i:, j:]
            # elif i >= 0 and j < 0:
            #     cropped_im1 = shifted_im1[i:, :j]
            #     cropped_im2 = im2[i:, :j]
            # elif i < 0 and j >= 0:
            #     cropped_im1 = shifted_im1[:i, j:]
            #     cropped_im2 = im2[:i, j:]
            # elif i < 0 and j < 0:
            #     cropped_im1 = shifted_im1[:i, :j]
            #     cropped_im2 = im2[:i, :j]
            
            # calculating the SSD
            ssd = np.sum((shifted_im1 - im2) ** 2)

            if ssd < min_ssd:
                min_ssd = ssd
                min_d_y = i
                min_d_x = j

    return (min_d_y, min_d_x)