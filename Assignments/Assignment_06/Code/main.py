import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from gsolve import gsolve

def ParseFiles(calibSetName, dir):
    assert os.path.exists(os.path.join(dir, calibSetName)), 'File not found: {}'.format(os.path.join(dir, calibSetName))
    
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

def calculate_crf(images, exposures, _lambda):
    P = len(images)
    N = int(5*256/(P-1))
    im_h, im_w, _ = images[0].shape

    x_points = np.random.randint(0, im_w, N)
    y_points = np.random.randint(0, im_h, N)
    random_points = np.vstack((x_points, y_points)).T

    # Sample the images
    Z = np.zeros((N, P, 3)).astype(np.uint8)
    for j, image in enumerate(images):
        for i in range(N):
            Z[i, j] = image[random_points[i, 1], random_points[i, 0]]
    
    # Create the triangle function
    w = np.zeros(256)
    for i in range(256):
        w[i] = weight_func(i)
    
    # Recover the camera response function (CRF) using Debevec's optimization code (gsolve.m)
    B = np.log(exposures)
    gs = []
    for i in range(3):
        g, _ = gsolve(Z[:,:,i], B, _lambda, w)
        gs.append(g)
    return gs

def calculate_radiance(images, exposures, gs):
    P = len(images)
    im_h, im_w, _ = images[0].shape
    radiance = np.zeros((im_h, im_w, 3))

    for c in range(3):
        numerator = np.zeros((im_h, im_w))
        denominator = np.zeros((im_h, im_w))
        for p in range(P):
            Z = images[p][:, :, c]
            numerator += weight_func(Z) * (gs[c][Z] - np.log(exposures[p]))
            denominator += weight_func(Z)

        radiance[:, :, c] = np.exp(np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0))
    return radiance

def global_tone_mapping(E, gamma=0.3):
    E = E / E.max()
    E = np.power(E, gamma)
    E = (E * 255).astype(np.uint8)
    return E

def local_tone_mapping(E, sd=0.5, dR=5, gamma=0.5):

    # Compute the intensity (I) by averaging the color channels.
    epsilon = 1e-6
    I = np.mean(E, axis=2)
    I = np.clip(I, epsilon, None)

    # Compute the chrominance channels: (R/I, G/I, B/I)
    chrominance = np.zeros_like(E)
    for c in range(3):
        chrominance[:, :, c] = E[:, :, c] / I

    # Compute the log intensity: L = log2(I), since log0 is undefined, we add a small epsilon to I
    L = np.log(I)/np.log(2)

    # Filter L with a Gaussian filter
    B = cv2.GaussianBlur(L, (0, 0), sd)

    # Compute the detail layer: D = L - B
    D = L - B

    # Apply an offset and a scale to the base: B′=(B−o)∗s
    o = np.max(B)
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
    calibSetName = args.calibSetName

    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    _lambda = 50

    # Parsing the input images to get the file names and corresponding exposure values
    filePaths, exposures = ParseFiles(calibSetName, inputDir)

    """ Task 1 """
    images = [np.array(Image.open(filePath)) for filePath in filePaths] # pil image is from 0 to 255
    r_g, g_g, b_g = calculate_crf(images, exposures, _lambda)

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
    radiance = calculate_radiance(images, exposures, [r_g, g_g, b_g])

    """ Task 3 """
    # Perform both local and global tone-mapping

    # Global tone-mapping
    global_hdr_image = global_tone_mapping(radiance, gamma=0.1)
    global_hdr_image = Image.fromarray(global_hdr_image)
    global_hdr_image.save(os.path.join(outputDir, '{}_Global.png'.format(calibSetName)))

    # Local tone-mapping
    local_hdr_image = local_tone_mapping(radiance, sd=0.5, gamma=0.5, dR=4)
    local_hdr_image = Image.fromarray(local_hdr_image)
    local_hdr_image.save(os.path.join(outputDir, '{}_Local.png'.format(calibSetName)))
