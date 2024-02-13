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
from extra import preprocess_image

def show_preprocessed_image(r, g, b):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(r, cmap='gray')
    ax[0].set_title('Red Channel')
    ax[1].imshow(g, cmap='gray')
    ax[1].set_title('Green Channel')
    ax[2].imshow(b, cmap='gray')
    ax[2].set_title('Blue Channel')
    plt.show()
    raise NotImplementedError
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGB Image Alignment')
    parser.add_argument('-a', '--aligned_method', type=str, default="task1", help='Method to use for alignment')
    parser.add_argument('-i', '--input_image', default=None, help='Input image name')
    parser.add_argument('--input_dir', default='../Images', help='Input image directory')
    parser.add_argument('--output_dir', default='../Results', help='Output image directory')
    parser.add_argument('--search_range', type=int, default=20, help='Search range for alignment')
    parser.add_argument('--border', nargs="+", type=int, default=[20, 200], help='Border to crop')
    parser.add_argument('--levels', type=int, default=4, help='Number of levels in the pyramid')

    # Extra credit options
    parser.add_argument('--auto_crop', action='store_true', help='Use auto cropping')
    parser.add_argument('--auto_contrast', action='store_true', help='Use auto contrast')
    parser.add_argument('--auto_white_balance', action='store_true', help='Use auto white balance')
    parser.add_argument('--better_features', action='store_true', help='Use better features for alignment')
    parser.add_argument('--better_transformation', action='store_true', help='Use better transformation for alignment')

    args = parser.parse_args()

    # Setting the input output file path
    imageDir = args.input_dir
    outDir = os.path.join(args.output_dir, args.aligned_method) 
    if args.auto_crop:
        outDir += "_cr"
    if args.auto_contrast:
        outDir += "_ct"
    if args.auto_white_balance:
        outDir += "_wb"
    if args.better_features:
        outDir += "_bf"
    if args.better_transformation:
        outDir += "_bt"
    
    if not os.path.exists(outDir):
        os.makedirs(outDir)

    if args.input_image is None:
        if args.aligned_method == "task1":
            images = [os.path.basename(path) for path in glob.glob(imageDir + '/*.jpg')]
        else:
            images = [os.path.basename(path) for path in glob.glob(imageDir + '/*.tif')]
    else:
        images = [os.path.basename(args.input_image)]

    metadata = {}
    for imageName in tqdm.tqdm(images):
        start_t = time.time()
        input_image_path = os.path.join(imageDir, imageName)
        # Get r, g, b channels from image strip
        r, g, b = read_strip(input_image_path)

        # Preprocessing the images
        border = args.border[1] if ".tif" in imageName else args.border[0]
        r_processed = preprocess_image(r, border, better_features=args.better_features)
        g_processed = preprocess_image(g, border, better_features=args.better_features)
        b_processed = preprocess_image(b, border, better_features=args.better_features)

        show_preprocessed_image(r_processed, g_processed, b_processed)

        # Calculate shift
        if args.aligned_method == "task1":
            rShift = task1.find_shift(r_processed, b_processed, args.search_range)
            gShift = task1.find_shift(g_processed, b_processed, args.search_range)
        elif args.aligned_method == "task2":
            rShift = task2.find_shift(r_processed, b_processed, levels=args.levels, search_range=args.search_range)
            gShift = task2.find_shift(g_processed, b_processed, levels=args.levels, search_range=args.search_range)
        else:
            raise ValueError(f"Invalid method {args.aligned_method}")
        
        # Shifting the images using the obtained shift values
        finalB = b
        finalG = circ_shift(g, gShift)
        finalR = circ_shift(r, rShift)

        # Putting together the aligned channels to form the color image
        finalImage = np.stack((finalR, finalG, finalB), axis = 2)

        # Writing the image to the Results folder
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
