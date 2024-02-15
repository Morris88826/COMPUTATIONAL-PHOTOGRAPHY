# Assignment 2


## Overview
In this assignment, we are required to merge the images from the Prokudin-Gorskii glass plates, which represent the Blue (B), Green (G), and Red \(R) channels of the image, arranged in a top-down order. We accomplish this by aligning the R and G channels with the B channel, identifying the alignment that minimizes the Sum of Squared Differences (SSD). Task 1 involves creating a straightforward \textbf{FindShift} function capable of handling small JPEG images. In contrast, Task 2 calls for a more efficient search method using an image pyramid. This approach significantly enhances execution time by conducting the broadest search at the lowest pyramid level, i.e. the smallest image. As we ascend to higher levels, the required search range for shifts narrows. Detailed explanations of the configuration and implementation will be provided in subsequent sections.

## Prerequisites
For this assignment, I uses these packages that are not in Python's standard library. Make sure you pip install these packages first.
* numpy
* matplotlib
* opencv: For finding edges (Canny edge detector)
* skimage: For resizing the images
* tdqm: Progress bar for keeping track of the progress

## Get Started
In the folder, there are two main items: the Code folder and the Report.pdf. To reproduce the result shown in the Report.pdf, please execute the following command from the root directory. Make sure you have the **Images** folder under the main directory. 

```
cd Code
python main.py 
```

main.py accepts several arguments.

*  **input_image:** Path to the image. 
    *  Default is None, which is to process on all images in the input_dir folder
*  **input_dir:** Input image directory. 
    *  Default is '../Images'.
*  **output_dir:** Output result directory. 
    *  Default is '../Results'.
*  **border:** Border to crop. 
    *  Default is [20, 200] where 20 the first element is used to crop .jpg images and the second one is used to crop .tif images.
*  **search_range:** Search range for alignment. 
    *  Default is 20
*  **levels:** Number of levels in the image pyramid
    *  Default is 4
*  **extra:** Extra credit option: 0 for none, 1 for auto crop, 2 for auto contrast, 3 for auto white balance, 4 for better features, 5 for better transformation
    *  Default is 0


I also provide code for generate the figures in the report (make sure you generate results first). Simply run:
```
python demo.py --task <task_id>
```
* task_id: this is the flag of which task you want to see for demo. 0 for task1, 1 for task2, 2 for auto crop, 3 for auto contrast, 4 for auto white balance, 5 for better features, 6 for better transformation
