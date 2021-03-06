This folder contains scripts that:

Go through a satellite image, isolate building rooftops using connected components, and automatically
crop a rectangle from each roof. This is then used to generate synthetic textures.

Input: satellite imagery, and ground truth (INRIA's buildings dataset). The input can even be a satellite image and it's binary segmentation (black-white) mask denoting a pixel as a roof/non-roof pixel.

Output: Rectangular patches/entire building rooftops for all buildings in the image

Works in conjunction with Models for Remote Sensing (https://github.com/bohaohuang/mrs)

Project website: https://bass-connections-2019.github.io/
@Author: Aneesh Gupta
