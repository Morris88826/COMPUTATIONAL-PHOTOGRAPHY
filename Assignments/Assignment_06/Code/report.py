import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from main import ParseFiles

def show_images(filePaths):
    tmpImage = plt.imread(filePaths[0])
    height, width, _ = tmpImage.shape
    largeImage = np.zeros((height*2, width * (len(filePaths)+1)//2, 3))
    num_col = len(filePaths)//2 + len(filePaths)%2

    for i, filePath in enumerate(filePaths):
        image = plt.imread(filePath)
        largeImage[height*(i//num_col):height*(i//num_col+1), width*(i%num_col):width*(i%num_col+1)] = image
    return largeImage

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate HDR image from a set of images with different exposures.')
    parser.add_argument('--inputDir', type=str, default='../Images/', help='Directory containing the input images.')
    parser.add_argument('--resultsDir', type=str, default='../Results/', help='Directory to the results images.')
    parser.add_argument('--outputDir', type=str, default='../report/', help='Directory to save the report images.')
    parser.add_argument('--calibSetName', '-c', type=str, default='Chapel', help='Name of the calibration set.')
    args = parser.parse_args()

    # Setting up the input output paths and the parameters
    inputDir = args.inputDir
    resultsDir = args.resultsDir
    outputDir = args.outputDir
    calibSetName = args.calibSetName

    filePaths, exposures = ParseFiles(calibSetName, inputDir)

    largeImage = show_images(filePaths)
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
        
    plt.imsave(os.path.join(outputDir, '{}.png'.format(calibSetName)), largeImage)
    plt.close()

    # Plot the CRF
    crf_image = plt.imread(os.path.join(resultsDir, '{}_CRF.png'.format(calibSetName)))
    plt.imsave(os.path.join(outputDir, '{}_CRF.png'.format(calibSetName)), crf_image)
    plt.close()

    # Plot the radiance map with global and local tone mapping
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(plt.imread(os.path.join(resultsDir, '{}_Global.png'.format(calibSetName))))
    ax[0].set_title('Global Tone Mapping')
    ax[0].axis('off')

    ax[1].imshow(plt.imread(os.path.join(resultsDir, '{}_Local.png'.format(calibSetName))))
    ax[1].set_title('Local Tone Mapping')
    ax[1].axis('off')

    plt.tight_layout()

    plt.savefig(os.path.join(outputDir, '{}_ToneMapping.png'.format(calibSetName)))
    plt.close() 
                 
