import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage.transform import resize

def low_pass_filter(kernel_size=3, sigma=1.0):
    # standard 2D Gaussian filter

    gaussian_1D = np.arange(kernel_size)
    mean = gaussian_1D.mean()

    gaussian_1D = np.exp(-((gaussian_1D - mean)**2) / (2 * sigma**2)) / (np.sqrt(2 * np.pi * sigma**2))
    gaussian_2D = np.outer(gaussian_1D, gaussian_1D)

    return gaussian_2D  

def high_pass_filter(kernel_size=3, sigma=1.0):
    
    # impulse filter
    impulse = np.zeros((kernel_size, kernel_size))
    impulse[kernel_size//2, kernel_size//2] = 1

    # low pass filter
    gaussian_2D = low_pass_filter(kernel_size, sigma)

    # high pass filter
    impulse = impulse - gaussian_2D

    return impulse

def scale_laplacian_image(image):
    # Find the min and max of the image
    img_min = image.min()
    img_max = image.max()
    
    # Scale the image to the range [0, 1]
    img_scaled = (image - img_min) / (img_max - img_min)
    
    # Bias the image so that the zero point maps to 0.5
    img_biased = (img_scaled - 0.5) * 2  # This makes the range [-1, 1]
    img_biased += 0.5  # Now the range is [0, 1] with 0.5 as the midpoint
    
    # Clip values to ensure they're within [0, 1]
    img_biased = np.clip(img_biased, 0, 1)
    
    return img_biased

def custom_gaussian_pyramid(image, levels, kernel_size=3, sigma=1.0):
    pyramid = [image]
    lp_filter = low_pass_filter(kernel_size, sigma)
    
    for _ in range(levels-1):
        # if the image is color
        if len(image.shape) == 3:
            filtered_image = np.zeros_like(image)
            for c in range(3):
                filtered_image[:,:,c] = convolve2d(image[:,:,c], lp_filter, mode='same')
        else:
            filtered_image = convolve2d(image, lp_filter, mode='same')

        # downsample the image
        image = filtered_image[::2, ::2]
        pyramid.append(image)
    
    return pyramid

def custom_laplacian_pyramid(image, levels, kernel_size=3, sigma=1.0):
    gaussian_pyramid_ = custom_gaussian_pyramid(image, levels, kernel_size, sigma)
    pyramid = []

    lp_filter = low_pass_filter(kernel_size, sigma)
    is_color = len(image.shape) == 3
    
    for i in range(levels-1):
        # get the gaussian image
        gaussian_image = gaussian_pyramid_[i+1]

        # upsample the image
        if is_color:
            upsampled_image = np.zeros((gaussian_image.shape[0]*2, gaussian_image.shape[1]*2, 3))
        else:
            upsampled_image = np.zeros((gaussian_image.shape[0]*2, gaussian_image.shape[1]*2))
        upsampled_image[::2, ::2] = gaussian_image

        # gaussian blur to interpolate
        if is_color:
            for c in range(3):
                upsampled_image[:,:,c] = convolve2d(upsampled_image[:,:,c], lp_filter, mode='same')
        else:
            upsampled_image = convolve2d(upsampled_image, lp_filter, mode='same')
        
        # scale correction
        upsampled_image = upsampled_image * 4 # since the filter is normalized, i.e. sum of all elements = 1
        upsampled_image = upsampled_image[:gaussian_pyramid_[i].shape[0], :gaussian_pyramid_[i].shape[1]]

        # subtract the upsampled image from the gaussian image
        image = gaussian_pyramid_[i] - upsampled_image

        pyramid.append(image)

    pyramid.append(gaussian_pyramid_[-1])
    return pyramid

def gaussian_pyramid(image, levels):
    pyramid = [image]
    for _ in range(levels-1):
        image = resize(image, (image.shape[0]//2, image.shape[1]//2), anti_aliasing=True)
        pyramid.append(image)
    return pyramid

def laplacian_pyramid(image, levels):
    gaussian_pyramid_ = gaussian_pyramid(image, levels)
    pyramid = []
    
    for i in range(levels-1):
        # filter the image
        gaussian_image = gaussian_pyramid_[i+1]
        upsampled_image = resize(gaussian_image, (gaussian_pyramid_[i].shape[0], gaussian_pyramid_[i].shape[1]), anti_aliasing=True)
        # subtract the upsampled image from the gaussian image
        image = gaussian_pyramid_[i] - upsampled_image
        pyramid.append(image)

    pyramid.append(gaussian_pyramid_[-1])
    return pyramid

def show_pyramid(pyramid, is_laplacian=False, title="Pyramid"):
    fig, axs = plt.subplots(1, levels, figsize=(10, 3))
    for i in range(levels):
        image = pyramid[i]
        image = scale_laplacian_image(image) if is_laplacian else image
        if len(image.shape) == 3:
            axs[i].imshow(image)
        else:
            axs[i].imshow(image, cmap='gray')
        axs[i].set_title("Level {}: {}x{}".format(i, image.shape[0], image.shape[1]))
        axs[i].axis('off')   
    fig.suptitle(title)
    plt.show()

if __name__ == "__main__":
    kernel_size = 3
    sigma = 1.0
    lp_filter = low_pass_filter(kernel_size, sigma)
    gt_lp_filter = np.array([[0.003, 0.013, 0.022, 0.013, 0.003],
                                [0.013, 0.059, 0.097, 0.059, 0.013],
                                [0.022, 0.097, 0.159, 0.097, 0.022],
                                [0.013, 0.059, 0.097, 0.059, 0.013],
                                [0.003, 0.013, 0.022, 0.013, 0.003]])
    
    # unit test
    assert np.allclose(lp_filter, gt_lp_filter, atol=1e-3), "Test failed!"

    
    hp_filter = high_pass_filter(kernel_size, sigma)
    gt_hp_filter = np.array([[-0.003, -0.013, -0.022, -0.013, -0.003],
                            [-0.013, -0.059, -0.097, -0.059, -0.013],
                            [-0.022, -0.097, 0.841, -0.097, -0.022],
                            [-0.013, -0.059, -0.097, -0.059, -0.013],
                            [-0.003, -0.013, -0.022, -0.013, -0.003]])
    
    # unit test
    assert np.allclose(hp_filter, gt_hp_filter, atol=1e-3), "Test failed!"

    test_image = "../Images/lena.png"
    image = plt.imread(test_image)
        
    grayscale = False
    if grayscale:
        image = np.mean(image, axis=2)
    else:
        image = image[:,:, :3]

    
    levels = 5
    kernel_size = 5
    # Gaussian pyramid
    gaussian_pyramid_ = gaussian_pyramid(image, levels)
    # gaussian_pyramid_ = custom_gaussian_pyramid(image, levels, kernel_size, sigma)
    # Laplacian pyramid
    laplacian_pyramid_ = laplacian_pyramid(image, levels)
    # laplacian_pyramid_ = custom_laplacian_pyramid(image, levels, kernel_size, sigma)

    show_pyramid(gaussian_pyramid_, title="Gaussian Pyramid")
    show_pyramid(laplacian_pyramid_, is_laplacian=True, title="Laplacian Pyramid")

