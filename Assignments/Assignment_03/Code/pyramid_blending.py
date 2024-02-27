

# Import required libraries
import os
import argparse
import numpy as np
from skimage.transform import resize
from matplotlib import pyplot as plt
from filters import gaussian_pyramid, laplacian_pyramid, custom_gaussian_pyramid, custom_laplacian_pyramid
from GetMask import GetMask

# Read source, target and mask for a given id
def Read(id, path = ""):
    source = plt.imread(path + "source_" + id + ".jpg")
    info = np.iinfo(source.dtype) # get information about the image type (min max values)
    source = source.astype(np.float32) / info.max # normalize the image into range 0 and 1
    target = plt.imread(path + "target_" + id + ".jpg")
    info = np.iinfo(target.dtype) # get information about the image type (min max values)
    target = target.astype(np.float32) / info.max # normalize the image into range 0 and 1

    if not os.path.exists(path + "mask_" + id + ".jpg"):
        return source, None, target
    
    mask   = plt.imread(path + "mask_" + id + ".jpg")
    info = np.iinfo(mask.dtype) # get information about the image type (min max values)
    mask = mask.astype(np.float32) / info.max # normalize the image into range 0 and 1

    return source, mask, target

# Pyramid Blend
def PyramidBlend(source, mask, target, levels = 5, custom=False):

    # Create Gaussian and Laplacian pyramids for source, mask and target
    if custom:
        L_s = custom_laplacian_pyramid(source, levels=levels, kernel_size=5, sigma=1.0)
        L_t = custom_laplacian_pyramid(target, levels=levels, kernel_size=5, sigma=1.0)
        G_m = custom_gaussian_pyramid(mask, levels=levels, kernel_size=5, sigma=1.0)
    else:
        L_s = laplacian_pyramid(source, levels=levels)
        L_t = laplacian_pyramid(target, levels=levels)
        G_m = gaussian_pyramid(mask, levels=levels)

    # Create the blended pyramid
    blended_pyramid = []
    for i in range(levels):
        blended_pyramid.append(L_s[i] * G_m[i] + L_t[i] * (1 - G_m[i]))

    # Collapsing the blended pyramid
    blended = blended_pyramid[-1]
    for i in range(levels-2, -1, -1):
        blended = resize(blended, (blended_pyramid[i].shape[0], blended_pyramid[i].shape[1]), anti_aliasing=True)
        blended += blended_pyramid[i]

    blended = np.clip(blended, 0, 1)

    return blended


def create_pyramid_blend(source, maskOriginal, target, levels, custom=False):
    # Cleaning up the mask
    mask = np.ones_like(maskOriginal)
    mask[maskOriginal < 0.5] = 0

    # Implement the PyramidBlend function (Task 2)
    pyramidOutput = PyramidBlend(source, mask, target, levels, custom=custom)

    return pyramidOutput

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pyramid Blending')
    parser.add_argument('--input_dir', type=str, default='../Images/', help='Input directory')
    parser.add_argument('--output_dir', type=str, default='../Results/', help='Output directory')
    parser.add_argument('--levels', type=int, default=5, help='Number of levels in the pyramid')
    parser.add_argument('--id', type=int, default=1, help='Image id')
    parser.add_argument('--custom', action='store_true', help='Use custom implementation')
    parser.add_argument('--custom_mask', action='store_true', help='Use custom mask')

    args = parser.parse_args()

    # Setting up the input output paths
    input_dir = args.input_dir
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # main area to specify files and display blended image
    index = args.id
    # Read data and clean mask
    source, maskOriginal, target = Read(str(index).zfill(2), input_dir)

    if args.custom_mask:
        maskOriginal = GetMask(source).astype(np.float32)/255.0
        # save the mask
        plt.imsave("{}mask_{}.jpg".format(input_dir, str(index).zfill(2)), maskOriginal)

    levels = args.levels

    ### The main part of the code ###
    # Implement the PyramidBlend function (Task 2)
    if args.custom:
        pyramidOutput = create_pyramid_blend(source, maskOriginal, target, levels, custom=True)
        plt.imsave("{}custom_pyramid_{}.jpg".format(output_dir, str(index).zfill(2)), pyramidOutput)
    else:
        pyramidOutput = create_pyramid_blend(source, maskOriginal, target, levels)
        plt.imsave("{}pyramid_{}.jpg".format(output_dir, str(index).zfill(2)), pyramidOutput)

