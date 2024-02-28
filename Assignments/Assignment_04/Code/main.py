# Import required libraries
import os
import numpy as np
from matplotlib import pyplot as plt

# Read source, target and mask for a given id
def Read(id, path = ""):
    source = plt.imread(path + "source_" + id + ".jpg")
    info = np.iinfo(source.dtype) # get information about the image type (min max values)
    source = source.astype(np.float32) / info.max # normalize the image into range 0 and 1
    target = plt.imread(path + "target_" + id + ".jpg")
    info = np.iinfo(target.dtype) # get information about the image type (min max values)
    target = target.astype(np.float32) / info.max # normalize the image into range 0 and 1
    mask   = plt.imread(path + "mask_" + id + ".jpg")
    info = np.iinfo(mask.dtype) # get information about the image type (min max values)
    mask = mask.astype(np.float32) / info.max # normalize the image into range 0 and 1

    return source, mask, target

# Adjust parameters, source and mask for negative offsets or out of bounds of offsets
def AlignImages(mask, source, target, offset):
    sourceHeight, sourceWidth, _ = source.shape
    targetHeight, targetWidth, _ = target.shape
    xOffset, yOffset = offset
    
    if (xOffset < 0):
        mask    = mask[abs(xOffset):, :]
        source  = source[abs(xOffset):, :]
        sourceHeight -= abs(xOffset)
        xOffset = 0
    if (yOffset < 0):
        mask    = mask[:, abs(yOffset):]
        source  = source[:, abs(yOffset):]
        sourceWidth -= abs(yOffset)
        yOffset = 0
    # Source image outside target image after applying offset
    if (targetHeight < (sourceHeight + xOffset)):
        sourceHeight = targetHeight - xOffset
        mask    = mask[:sourceHeight, :]
        source  = source[:sourceHeight, :]
    if (targetWidth < (sourceWidth + yOffset)):
        sourceWidth = targetWidth - yOffset
        mask    = mask[:, :sourceWidth]
        source  = source[:, :sourceWidth]
    
    maskLocal = np.zeros_like(target)
    maskLocal[xOffset:xOffset + sourceHeight, yOffset:yOffset + sourceWidth] = mask
    sourceLocal = np.zeros_like(target)
    sourceLocal[xOffset:xOffset + sourceHeight, yOffset:yOffset + sourceWidth] = source

    return sourceLocal, maskLocal

# Poisson Blend
def PoissonBlend(source, mask, target, isMix):
    
    return source * mask + target * (1 - mask)

if __name__ == '__main__':
    # Setting up the input output paths
    inputDir = '../Images/'
    outputDir = '../Results/'

    if not os.path.exists(outputDir):   
        os.makedirs(outputDir)
    
    # False for source gradient, true for mixing gradients
    isMix = False

    # Source offsets in target
    offsets = [[210, 10], [10, 28], [140, 80], [-40, 90], [60, 100], [20, 20], [-28, 88]]

    # main area to specify files and display blended image
    for index in range(len(offsets)):
        # Read data and clean mask
        source, maskOriginal, target = Read(str(index+1).zfill(2), inputDir)

        # Cleaning up the mask
        mask = np.ones_like(maskOriginal)
        mask[maskOriginal < 0.5] = 0

        # Align the source and mask using the provided offest
        source, mask = AlignImages(mask, source, target, offsets[index])

        
        ### The main part of the code ###
    
        # Implement the PoissonBlend function
        poissonOutput = PoissonBlend(source, mask, target, isMix)

        
        # Writing the result
                
        if not isMix:
            plt.imsave("{}poisson_{}.jpg".format(outputDir, str(index+1).zfill(2)), poissonOutput)
        else:
            plt.imsave("{}poisson_{}_Mixing.jpg".format(outputDir, str(index+1).zfill(2)), poissonOutput)
