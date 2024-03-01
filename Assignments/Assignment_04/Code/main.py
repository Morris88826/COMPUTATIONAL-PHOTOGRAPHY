# Import required libraries
import os
import numpy as np
import argparse
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

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

def calculateGradient(image):
    # Calculate the gradient in x and y direction
    gradientX = np.zeros_like(image)
    gradientY = np.zeros_like(image)
    gradientX[:, :-1] = image[:, 1:] - image[:, :-1]
    gradientY[:-1, :] = image[1:, :] - image[:-1, :]
    return gradientX, gradientY

# Poisson Blend
def PoissonBlend(source, mask, target, isMix):
    '''
    source: source image
    mask: binary mask, shape (H, W, 3)
    target: target image
    '''
    
    # padding the source, target, and mask
    source = np.pad(source, ((1,1), (1,1), (0,0)), 'constant')
    target = np.pad(target, ((1,1), (1,1), (0,0)), 'constant')
    mask = np.pad(mask, ((1,1), (1,1), (0,0)), 'constant')


    # duplicate boundary
    source[0, :, :] = source[1, :, :]
    source[-1, :, :] = source[-2, :, :]
    source[:, 0, :] = source[:, 1, :]
    source[:, -1, :] = source[:, -2, :]

    target[0, :, :] = target[1, :, :]
    target[-1, :, :] = target[-2, :, :]
    target[:, 0, :] = target[:, 1, :]
    target[:, -1, :] = target[:, -2, :]

    H, W, _ = mask.shape

    omega_indices = np.argwhere(mask[:,:,0] > 0)
    omega = np.zeros_like(mask)
    omega[omega_indices[:,0], omega_indices[:,1], :] = 1

    # center
    c_row = np.arange(W*H)
    c_col = np.arange(W*H)
    c_data = np.ones(W*H)
    indices = omega_indices[:,0]*W + omega_indices[:,1]
    c_data[indices] = 4

    # right
    r_row = indices
    r_col = omega_indices[:,0]*W + omega_indices[:,1]+1
    r_data = -1 * np.ones(len(r_row))

    # left
    l_row = indices
    l_col = omega_indices[:,0]*W + omega_indices[:,1]-1
    l_data = -1 * np.ones(len(l_row))

    # bottom
    b_row = indices
    b_col = (omega_indices[:,0]+1)*W + omega_indices[:,1]
    b_data = -1 * np.ones(len(b_row))

    # top
    t_row = indices
    t_col = (omega_indices[:,0]-1)*W + omega_indices[:,1]
    t_data = -1 * np.ones(len(t_row))

    row = np.concatenate([c_row, r_row, l_row, b_row, t_row])
    col = np.concatenate([c_col, r_col, l_col, b_col, t_col])
    data = np.concatenate([c_data, r_data, l_data, b_data, t_data])

    A = csr_matrix((data, (row, col)), shape=(W*H, W*H))
    b = np.zeros((W*H, 3))
    b[:, 0] = target[:,:,0].reshape(-1)
    b[:, 1] = target[:,:,1].reshape(-1)
    b[:, 2] = target[:,:,2].reshape(-1)

    if isMix:
        Gs = source[omega_indices[:,0], omega_indices[:,1]] - source[omega_indices[:,0], omega_indices[:,1]-1]
        Gt = target[omega_indices[:,0], omega_indices[:,1]] - target[omega_indices[:,0], omega_indices[:,1]-1]
        G1 = np.where(abs(Gs) > abs(Gt), Gs, Gt)

        Gs = source[omega_indices[:,0], omega_indices[:,1]] - source[omega_indices[:,0], omega_indices[:,1]+1]
        Gt = target[omega_indices[:,0], omega_indices[:,1]] - target[omega_indices[:,0], omega_indices[:,1]+1]
        G2 = np.where(abs(Gs) > abs(Gt), Gs, Gt)

        Gs = source[omega_indices[:,0], omega_indices[:,1]] - source[omega_indices[:,0]-1, omega_indices[:,1]]
        Gt = target[omega_indices[:,0], omega_indices[:,1]] - target[omega_indices[:,0]-1, omega_indices[:,1]]
        G3 = np.where(abs(Gs) > abs(Gt), Gs, Gt)

        Gs = source[omega_indices[:,0], omega_indices[:,1]] - source[omega_indices[:,0]+1, omega_indices[:,1]]
        Gt = target[omega_indices[:,0], omega_indices[:,1]] - target[omega_indices[:,0]+1, omega_indices[:,1]]
        G4 = np.where(abs(Gs) > abs(Gt), Gs, Gt)

        del_sqaure = G1 + G2 + G3 + G4
        b[omega_indices[:,0]*W + omega_indices[:,1]] = del_sqaure

    else:
        del_sqaure_S = 4*source[omega_indices[:,0], omega_indices[:,1]] - source[omega_indices[:,0], omega_indices[:,1]+1] - source[omega_indices[:,0], omega_indices[:,1]-1] - source[omega_indices[:,0]+1, omega_indices[:,1]] - source[omega_indices[:,0]-1, omega_indices[:,1]]
        b[omega_indices[:,0]*W + omega_indices[:,1]] = del_sqaure_S
    
    x = np.zeros((W*H, 3))
    x[:, 0] = spsolve(A, b[:, 0])
    x[:, 1] = spsolve(A, b[:, 1])
    x[:, 2] = spsolve(A, b[:, 2])
    
    x = x.reshape(H, W, 3)
    x = x[1:-1, 1:-1, :]

    # Clipping the result
    x = np.clip(x, 0, 1)

    return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Poisson Blending')
    parser.add_argument('--isMix', action='store_true', help='Use mixing gradients')
    parser.add_argument('--inputDir', type=str, default='../Images/', help='Input directory')
    parser.add_argument('--outputDir', type=str, default='../Results/', help='Output directory')
    args = parser.parse_args()

    # Setting up the input output paths
    inputDir = args.inputDir
    outputDir = args.outputDir

    if not os.path.exists(outputDir):   
        os.makedirs(outputDir)
    
    isMix = args.isMix
    # Source offsets in target
    offsets = [[210, 10], [10, 28], [140, 80], [-40, 90], [60, 100], [20, 20], [-28, 88], [70, -70]]

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
