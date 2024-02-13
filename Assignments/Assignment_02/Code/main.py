# Import required libraries
import os
import tqdm
import json
import time
import enum
import glob
import task1
import task2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from helper import read_strip, circ_shift, crop_image
from extra import auto_cropping, auto_contrasting, auto_white_balancing, find_edge

class ExtraCreditOptions(enum.Enum):
    NONE = 0
    AUTO_CROP = 1
    AUTO_CONTRAST = 2
    AUTO_WHITE_BALANCE = 3
    BETTER_FEATURES = 4
    BETTER_TRANSFORMATION = 5

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGB Image Alignment')
    parser.add_argument('-i', '--input_image', default=None, help='Input image name')
    parser.add_argument('--input_dir', default='../Images', help='Input image directory')
    parser.add_argument('--output_dir', default='../Results', help='Output image directory')
    parser.add_argument('--border', nargs="+", type=int, default=[20, 200], help='Border to crop')
    parser.add_argument('--search_range', type=int, default=20, help='Search range for alignment')
    parser.add_argument('--levels', type=int, default=4, help='Number of levels in the pyramid')
    parser.add_argument('--extra', type=int, default=0, help='Extra credit option: 0 for none, 1 for auto crop, 2 for auto contrast, 3 for auto white balance, 4 for better features, 5 for better transformation')

    # Extra credit options
    # parser.add_argument('--auto_crop', action='store_true', help='Use auto cropping')
    # parser.add_argument('--auto_contrast', action='store_true', help='Use auto contrast')
    # parser.add_argument('--auto_white_balance', action='store_true', help='Use auto white balance')
    # parser.add_argument('--better_features', action='store_true', help='Use better features for alignment')
    # parser.add_argument('--better_transformation', action='store_true', help='Use better transformation for alignment')

    args = parser.parse_args()

    extraOption = ExtraCreditOptions(args.extra)

    # Setting the input output file path
    imageDir = args.input_dir
    outDir = args.output_dir

    # Extra credit options
    if extraOption == ExtraCreditOptions.AUTO_CROP:
        print("Extra credit 1: Automatic cropping")
        outDir = os.path.join(outDir, "auto_crop")
    elif extraOption == ExtraCreditOptions.AUTO_CONTRAST:
        print("Extra credit 2: Automatic contrasting")
        outDir = os.path.join(outDir, "auto_contrast")
    elif extraOption == ExtraCreditOptions.AUTO_WHITE_BALANCE:
        print("Extra credit 3: Automatic white balance")
        outDir = os.path.join(outDir, "auto_white_balance")
    elif extraOption == ExtraCreditOptions.BETTER_FEATURES:
        print("Extra credit 4: Better features for alignment")
        outDir = os.path.join(outDir, "better_features")
    elif extraOption == ExtraCreditOptions.BETTER_TRANSFORMATION:
        print("Extra credit 5: Better transformation for alignment")
        outDir = os.path.join(outDir, "better_transformation")
    else: # No extra credit
        outDir = os.path.join(outDir, "default")
    
    if not os.path.exists(outDir):
        os.makedirs(outDir)

    # read images
    if args.input_image is None:
        images_path = [path for path in glob.glob(imageDir + '/*')]
    else:
        images_path = [args.input_image]

    metadata = {}
    for input_image_path in tqdm.tqdm(images_path):
        start_t = time.time()
        imageName = os.path.basename(input_image_path)

        # Get r, g, b channels from image strip
        r, g, b, scalingFactor = read_strip(input_image_path)

        # Preprocessing the images, scalingFactor==255 -> 8-bit image, scalingFactor==65535 -> 16-bit image
        border = args.border[1]
        if scalingFactor == 255:
            border = args.border[0]

        # crop the image
        cropped_r = crop_image(r, border)
        cropped_g = crop_image(g, border)
        cropped_b = crop_image(b, border)

        if extraOption == ExtraCreditOptions.BETTER_FEATURES:
            cropped_r = find_edge(cropped_r)
            cropped_g = find_edge(cropped_g)
            cropped_b = find_edge(cropped_b)

        # Calculate shift
        if scalingFactor == 255:
            rShift = task1.find_shift(cropped_r, cropped_b, args.search_range)
            gShift = task1.find_shift(cropped_g, cropped_b, args.search_range)
        else:
            rShift = task2.find_shift(cropped_r, cropped_b, search_range=args.search_range, levels=args.levels)
            gShift = task2.find_shift(cropped_g, cropped_b, search_range=args.search_range, levels=args.levels)

        # Shifting the images using the obtained shift values
        finalB = b
        finalG = circ_shift(g, gShift)
        finalR = circ_shift(r, rShift)

        # Putting together the aligned channels to form the color image
        finalImage = np.stack((finalR, finalG, finalB), axis = 2)

        # Post processing the image (Extra Credit Options)
        if extraOption == ExtraCreditOptions.AUTO_CROP:
            finalImage = auto_cropping(finalImage, rShift, gShift)
        elif extraOption == ExtraCreditOptions.AUTO_CONTRAST:
            finalImage = auto_contrasting(finalImage)
        elif extraOption == ExtraCreditOptions.AUTO_WHITE_BALANCE:
            finalImage = auto_white_balancing(finalImage)

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
