# Assignment 4

## Overview
In this assignment, we perform gradient-domain image blending to reduce the color mismatch to seamlessly blend the masked areas of the source image into a target photograph. There are two approaches: the Poisson blending approach, and the mixing gradient approach.

## Prerequisites
For this assignment, I use these packages that are not in Python's standard library. Make sure you pip install these packages first.
* numpy
* matplotlib
* scipy - csr_matrix, spsolve

## Get Started
In the folder, there are two main items: the Code folder and the Report.pdf. To reproduce the result shown in the Report.pdf, please execute the following command from the root directory. Make sure you have the **Images** folder under the main directory. 

```
cd Code
python main.py 
```

main.py accepts several arguments.

*  **isMix:** This is the flag of whether using mixing gradient approach
    *  Default is False, which is just use Poisson Blending
*  **input_dir:** Input image directory. 
    *  Default is '../Images'.
*  **output_dir:** Output result directory. 
    *  Default is '../Results'.


There is also a report.py for generating the figures shown in the report. It uses the arguments as main.py.
