"""
Credit: Alyosha Efros
""" 
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as sktr
import argparse
from filters import low_pass_filter, high_pass_filter
from scipy.signal import convolve2d



def get_points(im1, im2):
    print('Please select 2 points in each image for alignment.')
    plt.imshow(im1)
    p1, p2 = plt.ginput(2)
    plt.close()
    plt.imshow(im2)
    p3, p4 = plt.ginput(2)
    plt.close()
    return (p1, p2, p3, p4)

def recenter(im, r, c):
    R, C, _ = im.shape
    rpad = (int) (np.abs(2*r+1 - R))
    cpad = (int) (np.abs(2*c+1 - C))
    return np.pad(
        im, [(0 if r > (R-1)/2 else rpad, 0 if r < (R-1)/2 else rpad),
             (0 if c > (C-1)/2 else cpad, 0 if c < (C-1)/2 else cpad),
             (0, 0)], 'constant')

def find_centers(p1, p2):
    cx = np.round(np.mean([p1[0], p2[0]]))
    cy = np.round(np.mean([p1[1], p2[1]]))
    return cx, cy

def align_image_centers(im1, im2, pts):
    p1, p2, p3, p4 = pts
    h1, w1, b1 = im1.shape
    h2, w2, b2 = im2.shape
    
    cx1, cy1 = find_centers(p1, p2)
    cx2, cy2 = find_centers(p3, p4)

    im1 = recenter(im1, cy1, cx1)
    im2 = recenter(im2, cy2, cx2)
    return im1, im2

def rescale_images(im1, im2, pts):
    p1, p2, p3, p4 = pts
    len1 = np.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)
    len2 = np.sqrt((p4[1] - p3[1])**2 + (p4[0] - p3[0])**2)
    dscale = len2/len1
    if dscale < 1:
        im1 = sktr.rescale(im1, dscale, channel_axis=2)
    else:
        im2 = sktr.rescale(im2, 1./dscale, channel_axis=2)
    return im1, im2

def rotate_im1(im1, im2, pts):
    p1, p2, p3, p4 = pts
    theta1 = math.atan2(-(p2[1] - p1[1]), (p2[0] - p1[0]))
    theta2 = math.atan2(-(p4[1] - p3[1]), (p4[0] - p3[0]))
    dtheta = theta2 - theta1
    im1 = sktr.rotate(im1, dtheta*180/np.pi)
    return im1, dtheta

def match_img_size(im1, im2):
    # Make images the same size
    h1, w1, c1 = im1.shape
    h2, w2, c2 = im2.shape
    if h1 < h2:
        im2 = im2[int(np.floor((h2-h1)/2.)) : -int(np.ceil((h2-h1)/2.)), :, :]
    elif h1 > h2:
        im1 = im1[int(np.floor((h1-h2)/2.)) : -int(np.ceil((h1-h2)/2.)), :, :]
    if w1 < w2:
        im2 = im2[:, int(np.floor((w2-w1)/2.)) : -int(np.ceil((w2-w1)/2.)), :]
    elif w1 > w2:
        im1 = im1[:, int(np.floor((w1-w2)/2.)) : -int(np.ceil((w1-w2)/2.)), :]
    assert im1.shape == im2.shape
    return im1, im2

def align_images(im1, im2, pts=None):
    if pts is None:
        pts = get_points(im1, im2)
    im1, im2 = align_image_centers(im1, im2, pts)
    im1, im2 = rescale_images(im1, im2, pts)
    im1, angle = rotate_im1(im1, im2, pts)
    im1, im2 = match_img_size(im1, im2)
    return im1, im2



def normalize_image(im):
    # Check if the image data type is integer
    if np.issubdtype(im.dtype, np.integer):
        info = np.iinfo(im.dtype) # get information about the image type (min max values)
        im = im.astype(np.float32) / info.max # normalize the image into range 0 and 1
    else:
        # If it's already a float, ensure it's in range [0, 1]
        im = im.astype(np.float32)
        if im.max() > 1.0:
            im /= 255.0 # assuming the float image range is 0 to 255
    
    if im.ndim == 3:
        im = im[..., :3]
    return im

def create_hybrid_image(im1, im2, kernel_size_1, sigma_1, kernel_size_2, sigma_2, isGray=False, verbose=False):

    im1 = normalize_image(im1)
    im2 = normalize_image(im2)

    if isGray:
        im1 = np.mean(im1, axis=2)
        im2 = np.mean(im2, axis=2)
	
    # Now you are ready to write your own code for creating hybrid images!
    lp_filter = low_pass_filter(kernel_size_1, sigma_1)
    hp_filter = high_pass_filter(kernel_size_2, sigma_2)

    if isGray:
        im_low = convolve2d(im1, lp_filter, mode='same')
        im_high = convolve2d(im2, hp_filter, mode='same')
    else:
        im_low = np.zeros(im1.shape)
        im_high = np.zeros(im2.shape)
        for i in range(3):
            im_low[:, :, i] = convolve2d(im1[:, :, i], lp_filter, mode='same')
            im_high[:, :, i] = convolve2d(im2[:, :, i], hp_filter, mode='same')

    im = im_low + im_high
    im = (im - im.min())/ (im.max() - im.min())
    
    if verbose:
        return im, (im_low, im_high)
    else:
        return im

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hybrid Images')
    parser.add_argument('--isGray', action='store_true', help='Convert the image to grayscale')
    parser.add_argument('--im1', type=str, default='../Images/Monroe.jpg', help='Input image 1')
    parser.add_argument('--im2', type=str, default='../Images/Einstein.jpg', help='Input image 2')
    parser.add_argument('--output_dir', type=str, default='../Results/', help='Output directory')

    # parameters
    parser.add_argument('--kernal_size', type=int, default=20, help='Size of the kernal')
    parser.add_argument('--sigma', type=float, default=4.0, help='Standard deviation of the Gaussian filter')

    args = parser.parse_args()

    outDir = args.output_dir
    if not os.path.exists(outDir):
        os.makedirs(outDir)
        
    isGray = args.isGray    

    im1_name = os.path.basename(args.im1)
    im2_name = os.path.basename(args.im2)

    # 1. load the images
	
	# Low frequency image
    im1 = plt.imread(args.im1) # read the input image
    
	# High frequency image
    im2 = plt.imread(args.im2) # read the input image

    assert im1.ndim == im2.ndim, "The dimensions of the two images are not the same!"

    if im1.ndim == 2:
        isGray = True


    # 2. align the two images by calling align_images
    im1_aligned, im2_aligned = align_images(im1, im2)
    
    # 3. create the hybrid image by calling create_hybrid_image
    kernel_size = args.kernal_size
    sigma = args.sigma
    im = create_hybrid_image(im1_aligned, im2_aligned, kernel_size, sigma, kernel_size, sigma, isGray)

    if isGray:
        plt.imsave(outDir + im1_name.split('.')[0] + '_' + im2_name.split('.')[0] + '_Hybrid.jpg', im, cmap='gray')
    else:
        plt.imsave(outDir + im1_name.split('.')[0] + '_' + im2_name.split('.')[0] + '_Hybrid.jpg', im)

