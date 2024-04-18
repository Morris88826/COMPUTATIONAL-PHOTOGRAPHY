import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from gsolve import gsolve

# Based on code by James Tompkin
#
# reads in a directory and parses out the exposure values
# files should be named like: "xxx_yyy.jpg" where
# xxx / yyy is the exposure in seconds. 
def ParseFiles(calibSetName, dir):
    imageNames = os.listdir(os.path.join(dir, calibSetName))
    
    filePaths = []
    exposures = []
    
    for imageName in imageNames:
        exposure = imageName.split('.')[0].split('_')
        exposures.append(int(exposure[0]) / int(exposure[1]))
        filePaths.append(os.path.join(dir, calibSetName, imageName))
    
    # sort by exposure
    sortedIndices = np.argsort(exposures)[::-1]
    filePaths = [filePaths[i] for i in sortedIndices]
    exposures = [exposures[i] for i in sortedIndices]
    
    return filePaths, exposures

def weight_func(z, zmin=0, zmax=255):
    return np.where(z <= (zmin + zmax) / 2, z - zmin, zmax - z)

def calculate_crf(images, exposures, random_points, _lambda):
    # Sample the images
    N = random_points.shape[0]
    P = len(images)

    Z = np.zeros((N, P)).astype(np.uint8) 
    for j, image in enumerate(images):
        for i in range(N):
            Z[i, j] = image[random_points[i, 1], random_points[i, 0]]
    
    # Create the triangle function
    w = np.zeros(256)
    for i in range(256):
        w[i] = weight_func(i)
    
    # Recover the camera response function (CRF) using Debevec's optimization code (gsolve.m)
    B = np.log(exposures)
    g, _ = gsolve(Z, B, _lambda, w)
    
    
    return Z, g

def global_tone_mapping(E, gamma=0.3):
    E = E / E.max()
    E = np.power(E, gamma)
    E = E * 255
    E = E.astype(np.uint8)
    return E

def local_tone_mapping(E, sd=0.5, gamma=0.5):

    # Compute the intensity (I) by averaging the color channels.
    epsilon = 1e-6
    I = np.mean(E, axis=2)
    I = np.clip(I, epsilon, None)

    # Compute the chrominance channels: (R/I, G/I, B/I)
    chrominance = np.zeros_like(E)
    for c in range(3):
        chrominance[:, :, c] = E[:, :, c] / I

    # Compute the log intensity: L = log2(I)
    L = np.log(I)/np.log(2)

    # Filter L with a Gaussian filter
    B = cv2.GaussianBlur(L, (0, 0), sd)

    # Compute the detail layer: D = L - B
    D = L - B

    # Apply an offset and a scale to the base: B′=(B−o)∗s
    o = np.max(B)
    dR = 5
    s = dR / (np.max(B) - np.min(B))
    B_ = (B - o) * s

    # Reconstruct the log intensity
    O = np.power(2, B_ + D)

    # Put back the colors: R′, G′, B′ = O ∗ (R/I, G/I, B/I)
    R_ = O * chrominance[:, :, 0]
    G_ = O * chrominance[:, :, 1]
    B_ = O * chrominance[:, :, 2]

    result = np.zeros_like(E)
    result[:, :, 0] = R_
    result[:, :, 1] = G_
    result[:, :, 2] = B_

    # Apply gamma correction
    result = result / result.max()
    result = np.power(result, gamma)
    result = (result * 255).astype(np.uint8)

    return result
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate HDR image from a set of images with different exposures.')
    parser.add_argument('--inputDir', type=str, default='../Images/', help='Directory containing the input images.')
    parser.add_argument('--outputDir', type=str, default='../Results/', help='Directory to save the output images.')
    parser.add_argument('--calibSetName', '-c', type=str, default='Chapel', help='Name of the calibration set.')
    args = parser.parse_args()

    # Setting up the input output paths and the parameters
    inputDir = args.inputDir
    outputDir = args.outputDir
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    _lambda = 50

    calibSetName = args.calibSetName

    # Parsing the input images to get the file names and corresponding exposure
    # values
    filePaths, exposures = ParseFiles(calibSetName, inputDir)


    """ Task 1 """
    image = np.array(Image.open(filePaths[0]))
    P = len(filePaths)
    N = int(5*256/(P-1))
    x_points = np.random.randint(0, image.shape[1], N)
    y_points = np.random.randint(0, image.shape[0], N)
    random_points = np.vstack((x_points, y_points)).T

    # Sample the images
    r_images = []
    g_images = []
    b_images = []
    for filePath in filePaths:
        image = Image.open(filePath)
        image = np.array(image)
        r_images.append(image[:, :, 0])
        g_images.append(image[:, :, 1])
        b_images.append(image[:, :, 2])
    # Calculate the CRF for each channel
    r_Z, r_g = calculate_crf(r_images, exposures, random_points, _lambda)
    g_Z, g_g = calculate_crf(g_images, exposures, random_points, _lambda)
    b_Z, b_g = calculate_crf(b_images, exposures, random_points, _lambda)
    Zs = [r_Z, g_Z, b_Z]
    gs = [r_g, g_g, b_g]
    images = [r_images, g_images, b_images]

    # Plot the CRF
    plt.plot(r_g, np.arange(256), c='r', label='Red', linewidth=0.5)
    plt.plot(g_g, np.arange(256), c='g', label='Green', linewidth=0.5)
    plt.plot(b_g, np.arange(256), c='b', label='Blue', linewidth=0.5)
    plt.xlabel('log exposure X')
    plt.ylabel('pixel value Z')
    plt.xlim(-20, 5)
    plt.ylim(0, 300)
    plt.title('Camera Response Function')
    plt.legend()
    plt.savefig(os.path.join(outputDir, '{}_CRF.png'.format(calibSetName)))
    plt.close()

    """ Task 2 """
    # Reconstruct the radiance using the calculated CRF
    hdr_image = np.zeros((image.shape[0], image.shape[1], 3))
    for c in range(3):  
        numerator = np.zeros((image.shape[0], image.shape[1]))
        denominator = np.zeros((image.shape[0], image.shape[1]))
        for p in range(P):
            Z = images[c][p]
            numerator += weight_func(Z) * (gs[c][Z] - np.log(exposures[p]))
            denominator += weight_func(Z)
        
        hdr_image[:, :, c] = np.exp(np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0))

    """ Task 3 """
    # Perform both local and global tone-mapping

    # Global tone-mapping
    global_hdr_image = global_tone_mapping(hdr_image, gamma=0.1)
    global_hdr_image = Image.fromarray(global_hdr_image)
    global_hdr_image.save(os.path.join(outputDir, '{}_Global.png'.format(calibSetName)))

    # Local tone-mapping
    local_hdr_image = local_tone_mapping(hdr_image, sd=0.5, gamma=0.5)
    local_hdr_image = Image.fromarray(local_hdr_image)
    local_hdr_image.save(os.path.join(outputDir, '{}_Local.png'.format(calibSetName)))
