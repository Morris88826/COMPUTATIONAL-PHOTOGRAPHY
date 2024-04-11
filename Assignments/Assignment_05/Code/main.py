# Import required libraries
import os
import cv2
import glob
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

# Read source and mask (if exists) for a given id
def Read(id, path = ""):
    source = plt.imread(path + "image_" + id + ".jpg") / 255
    maskPath = path + "mask_" + id + ".jpg"
    
    if os.path.isfile(maskPath):
        mask = plt.imread(maskPath)
        assert(mask.shape == source.shape), 'size of mask and image does not match'
        mask = (mask > 128)[:, :, 0].astype(int)
    else:
        mask = np.zeros_like(source)[:, :, 0].astype(int)

    return source, mask

def find_gradient(image):
    gray = cv2.cvtColor((image*255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(float) / 255
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    energy = np.abs(convolve(gray, Gx)) + np.abs(convolve(gray, Gy))
    return energy

def find_seam(energy):
    M = energy.copy()

    for i in range(1, energy.shape[0]):
        for j in range(energy.shape[1]):
            if j == 0:
                M[i, j] = min(M[i-1, j], M[i-1, j+1]) + M[i, j]
            elif j == energy.shape[1]-1:
                M[i, j] = min(M[i-1, j-1], M[i-1, j]) + M[i, j]
            else:
                M[i, j] = min(M[i-1, j-1], M[i-1, j], M[i-1, j+1]) + M[i, j]
    

    # find the optimal seam
    seam = np.zeros(M.shape[0], dtype=int)
    seam[-1] = np.argmin(M[-1])    

    # backtrack to find the seam
    for i in range(M.shape[0]-2, -1, -1):
        j = seam[i+1]
        if j == 0:
            seam[i] = j + np.argmin(M[i, j:j+2])
        elif j == M.shape[1]-1:
            seam[i] = j-1 + np.argmin(M[i, j-1:j+1])
        else:
            seam[i] = j-1 + np.argmin(M[i, j-1:j+2])
    return M, seam

def SeamCarve(input, widthFac, heightFac, mask=None):

    # Main seam carving function. This is done in three main parts: 1)
    # computing the energy function, 2) finding optimal seam, and 3) removing
    # the seam. The three parts are repeated until the desired size is reached.

    assert (widthFac == 1 or heightFac == 1), 'Changing both width and height is not supported!'
    assert (widthFac <= 1 and heightFac <= 1), 'Increasing the size is not supported!'

    inSize = input.shape
    size   = (int(widthFac*inSize[1]), int(heightFac*inSize[0]))
    useMask = True if mask is not None else False

    image = input.copy()
    output_width = size[0]
    if heightFac < 1:
        # rotate the image
        image = np.rot90(image, 1)
        if useMask:
            mask = np.rot90(mask, 1)
        output_width = size[1]
    
    input_width = image.shape[1]
    for i in tqdm.tqdm(range(input_width - output_width)):
        energy = find_gradient(image)
        if useMask:
            energy[mask.astype(bool)] = 1e9

        # find the optimal seam
        _, seam = find_seam(energy)
        
        # remove the seam
        output = np.zeros((image.shape[0], image.shape[1]-1, 3))
        if useMask:
            new_mask = np.zeros((image.shape[0], image.shape[1]-1))

        for i in range(output.shape[0]):
            output[i, :, 0] = np.delete(image[i, :, 0], seam[i])
            output[i, :, 1] = np.delete(image[i, :, 1], seam[i])
            output[i, :, 2] = np.delete(image[i, :, 2], seam[i])

            if useMask:
                new_mask[i, :] = np.delete(mask[i, :], seam[i])

        image = output
        if useMask:
            mask = new_mask

    if heightFac < 1:
        # rotate the image
        image = np.rot90(image, -1)

    return image, size


if __name__ == "__main__":
    # Setting up the input output paths
    inputDir = '../Images/'
    outputDir = '../Results/'

    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    start_idx = 1
    N = len(glob.glob(inputDir + "image_*.jpg"))
    for index in range(start_idx, N + 1):
        for (widthFac, heightFac) in [[0.5, 1], [1, 0.5]]:

            input, mask = Read(str(index).zfill(2), inputDir)
            print("Processing image {} with size {}x{}".format(index, input.shape[1], input.shape[0]))
            print(" - widthFac: {}, heightFac: {}".format(widthFac, heightFac))

            # Performing seam carving. This is the part that you have to implement.
            output, size = SeamCarve(input, widthFac, heightFac)

            if not os.path.exists(outputDir):
                os.makedirs(outputDir)

            # Writing the result
            plt.imsave("{}/result_{}_{}x{}.jpg".format(outputDir, 
                                                    str(index).zfill(2), 
                                                    str(size[0]).zfill(2), 
                                                    str(size[1]).zfill(2)), output)

            print("=====================================")

    # Extra credit part
    sample_id = 4
    for (widthFac, heightFac) in [[0.5, 1], [1, 0.5]]:
        input, mask = Read(str(sample_id).zfill(2), inputDir)
        print("Processing image {} with size {}x{}".format(sample_id, input.shape[1], input.shape[0]))
        print(" - widthFac: {}, heightFac: {}".format(widthFac, heightFac))

        # Performing seam carving. This is the part that you have to implement.
        output, size = SeamCarve(input, widthFac, heightFac, mask)

        if not os.path.exists(outputDir):
            os.makedirs(outputDir)

        # Writing the result
        plt.imsave("{}/result_{}_{}x{}_masked.jpg".format(outputDir, 
                                                str(sample_id).zfill(2), 
                                                str(size[0]).zfill(2), 
                                                str(size[1]).zfill(2)), output)