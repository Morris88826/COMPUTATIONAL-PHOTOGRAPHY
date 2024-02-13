import task1
from skimage.transform import resize
from helper import circ_shift

# The main part of the code. Implement the FindShift function
def find_shift(im1, im2, search_range=20, levels=4):
    '''
    Find the shift between two images using the pyramid method
    Inputs:
     - im1: The first image
     - im2: The target image to which the first image is to be aligned
     - search_range: The range of the search in the minimum level of the pyramid
     - levels: The number of levels in the pyramid
    '''

    # Create the Gaussian pyramid for both images
    im1_pyramid = []
    im2_pyramid = []
    for i in range(levels):
        if i == 0:
            # Initialize the pyramid
            im1_pyramid = [im1]
            im2_pyramid = [im2]
        else:
            im1_pyramid.append(resize(im1_pyramid[-1], (im1_pyramid[-1].shape[0]//2, im1_pyramid[-1].shape[1]//2), anti_aliasing=True))
            im2_pyramid.append(resize(im2_pyramid[-1], (im2_pyramid[-1].shape[0]//2, im2_pyramid[-1].shape[1]//2), anti_aliasing=True))

    # reverse the pyramid
    im1_pyramid.reverse()
    im2_pyramid.reverse()

    shift = (0, 0)
    for i in range(levels):
        im1 = im1_pyramid[i]
        im2 = im2_pyramid[i]

        shift = (shift[0]*2, shift[1]*2)
        shifted_im1 = circ_shift(im1, shift)
        # Calculate the shift at the current level    
        if i == 0:
            (d_y, d_x) = task1.find_shift(shifted_im1, im2, search_range=search_range)
        else: 
            (d_y, d_x) = task1.find_shift(shifted_im1, im2, search_range=2)

        # Update the shift
        shift = (shift[0] + d_y, shift[1] + d_x)

    return shift
