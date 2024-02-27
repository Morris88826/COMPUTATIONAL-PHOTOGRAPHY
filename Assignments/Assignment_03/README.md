# Assignment 3


## Overview
In this assignment, we focused on two major tasks: generating hybrid images using Gaussian and Laplacian filters and implementing pyramid blending to merge two images more smoothly.

## Prerequisites
For this assignment, I use these packages that are not in Python's standard library. Make sure you pip install these packages first.
* numpy
* matplotlib
* skimage: For resizing the images

## Get Started
In the folder, there are two main items: the Code folder and the Report.pdf. To reproduce the result shown in the Report.pdf, please execute the following command from the root directory. Make sure you have the **Images** folder under the main directory. 

```
cd Code
python main.py 
```

main.py accepts several arguments.

*  **task_id:** This is the flag of which task you run. 0 for task1, 1 for extra credit, 2 for task2, and 3 ablation studies on task1 with different sd
    *  Default is 0, which is task1
*  **input_dir:** Input image directory. 
    *  Default is '../Images'.
*  **output_dir:** Output result directory. 
    *  Default is '../Results'.


One can also choose to run "hybrid_images.py" for task1 and "pyramid_blending.py" for task2 directly.

### Hybrid Images
```
python hybrid_images.py
```
*  **isGray:** Whether use grayscale images to generate the hybrid image
    *  Default is False.
*  **im1:** Path to im1.
    *  Default is '../Images/Monroe.jpg'.
*  **im2:** Path to im2.
    *  Default is '../Images/Einstein.jpg'.
*  **output_dir:** Output result directory. 
    *  Default is '../Results'.
*  **kernal_size** Specify the kernel size you want to use for filtering
    *  Default is 20
*  **sigma** Standard deviation of the Gaussian filter
    *  Default is 4.0

### Pyramid blending
```
python pyramid_blending.py
```
*  **input_dir:** Input image directory. 
    *  Default is '../Images'.
*  **output_dir:** Output result directory. 
    *  Default is '../Results'.
*  **levels:** Number of levels in the pyramid.
    *  Default is 5. 
*  **id:** Image id. Use the select the source, target, and mask. Ex: id = 1 will find source_01, target_01, mask_01
    *  Default is 1.
