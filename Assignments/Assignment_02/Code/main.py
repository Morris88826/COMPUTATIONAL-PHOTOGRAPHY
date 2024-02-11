# Import required libraries
import os
import tqdm
import json
import time
import glob
import task1
import task2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from helper import read_strip, circ_shift


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGB Image Alignment')
    parser.add_argument('-a', '--aligned_method', type=str, default="task1", help='Method to use for alignment')
    parser.add_argument('-i', '--input_image', default=None, help='Input image name')
    parser.add_argument('--input_dir', default='../Images', help='Input image directory')
    parser.add_argument('--output_dir', default='../Results', help='Output image directory')
    parser.add_argument('--search_range', type=int, default=20, help='Search range for alignment')
    parser.add_argument('--border', nargs="+", type=int, default=[20, 200], help='Border to crop')
    parser.add_argument('--levels', type=int, default=4, help='Number of levels in the pyramid')
    args = parser.parse_args()


    # Setting the input output file path
    imageDir = args.input_dir
    outDir = os.path.join(args.output_dir, args.aligned_method) 

    if args.input_image is None:
        # Get all the images in the input directory
        if args.aligned_method == "task1":
            images = [os.path.basename(path) for path in glob.glob(imageDir + '/*.jpg')]
        else:
            images = [os.path.basename(path) for path in glob.glob(imageDir + '/*')]
    else:
        images = [os.path.basename(args.input_image)]

    metadata = {}
    for imageName in tqdm.tqdm(images):
        start_t = time.time()
        input_image_path = os.path.join(imageDir, imageName)
        # Get r, g, b channels from image strip
        r, g, b = read_strip(input_image_path)

        # get the hardcoded border value for the image
        border = args.border[1] if ".tif" in imageName else args.border[0]

        # Calculate shift
        if args.aligned_method == "task1":
            rShift = task1.main(r, b, args.search_range, border=border)
            gShift = task1.main(g, b, args.search_range, border=border)
        elif args.aligned_method == "task2":
            rShift = task2.main(r, b, levels=args.levels, search_range=args.search_range, border=border)
            gShift = task2.main(g, b, levels=args.levels, search_range=args.search_range, border=border)
        else:
            raise ValueError(f"Invalid method {args.aligned_method}")
        
        # Shifting the images using the obtained shift values
        finalB = b
        finalG = circ_shift(g, gShift)
        finalR = circ_shift(r, rShift)

        # Putting together the aligned channels to form the color image
        finalImage = np.stack((finalR, finalG, finalB), axis = 2)

        # Writing the image to the Results folder
        if not os.path.exists(outDir):
            os.makedirs(outDir)
        out_path = os.path.join(outDir, imageName.split('.')[0]+'.jpg')
        plt.imsave(out_path, finalImage)

        # Storing the metadata
        metadata[imageName] = {
            "time": time.time() - start_t,
            "rShift": rShift,
            "gShift": gShift
        }

    # Saving the metadata
    metadata_path = os.path.join(outDir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
