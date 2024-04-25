# Assignment 6

## Overview
In this assignment, we automatically combine multiple exposures into a single high dynamic range radiance map and then convert this radiance map to an image suitable for display through tone mapping.

## Prerequisites
For this assignment, I use these packages that are not in Python's standard library. Make sure you pip install these packages first.
* cv2
* numpy
* pillow
* matplotlib

## Get Started
In the folder, there are two main items: the Code folder and the Report.pdf. To reproduce the result shown in the Report.pdf, please execute the following command from the root directory. Make sure you have the **Images** folder under the main directory. 

```
cd Code
python main.py
```
main.py accepts several arguments.

*  **calibSetName:** This is the name of the example name, i.e. 'Chapel' or 'Office'.
    *  Default is 'Chapel'
*  **inputDir:** Input image directory. 
    *  Default is '../Images'.
*  **outputDir:** Output result directory. 
    *  Default is '../Results'.

There is also a report.py for generating the figures shown in the report. This can only work after you executed main.py.

