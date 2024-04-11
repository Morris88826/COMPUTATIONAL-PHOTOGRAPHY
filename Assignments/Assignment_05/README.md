# Assignment 5

## Overview
In this assignment, we perform seam carving to resize the image. This technique selectively removes parts of the image that are less important, while maintaining the aspect ratio of the most important objects.

## Prerequisites
For this assignment, I use these packages that are not in Python's standard library. Make sure you pip install these packages first.
* cv2
* numpy
* matplotlib
* scipy - convolve

## Get Started
In the folder, there are two main items: the Code folder and the Report.pdf. To reproduce the result shown in the Report.pdf, please execute the following command from the root directory. Make sure you have the **Images** folder under the main directory. 

```
cd Code
python main.py 
```
By running the main program directly, it will process all images with the filename 'image_X.jpg' in the Images folder. The results will be saved in the Results folder under the root directory. Additionally, for extra points, it will generate the corresponding seam carving with the mask on sample image 'image_04.jpg'.


### Unit Test
I also create a unit_test.py file for testing the function of finding the seam. One can verify it by running
```
cd Code
pytest unit_test.py 
```
