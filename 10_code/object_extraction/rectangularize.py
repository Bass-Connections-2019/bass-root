"""
@Author: Aneesh Gupta
Automate rectangular patch extraction from building rooftops
This works in conjunction with the Models for Remote Sensing codebase
https://github.com/bohaohuang/mrs

References: https://stackoverflow.com/questions/2478447/find-largest-rectangle-containing-only-zeros-in-an-n%C3%97n-binary-matrix/30418912#30418912
"""
import numpy as np
import cv2
from PIL import Image
from scipy import ndimage
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import sys
from tqdm import tqdm
from statistics import mean
sys.path.append(r'/hdd/2019-bass-connections-aatb/mrs_new')
from mrs_utils import misc_utils, vis_utils, eval_utils
from collections import namedtuple
from operator import mul

Info = namedtuple('Info', 'start height')

def max_rect(mat, value=0):
    """returns (height, width, left_column, bottom_row) of the largest rectangle 
    containing all `value`'s.
    """
    it = iter(mat)
    hist = [(el==value) for el in next(it, [])]
    max_rect = max_rectangle_size(hist) + (0,)
    for irow,row in enumerate(it):
        hist = [(1+h) if el == value else 0 for h, el in zip(hist, row)]
        max_rect = max(max_rect, max_rectangle_size(hist) + (irow+1,), key=area)
        # irow+1, because we already used one row for initializing max_rect
    return max_rect

def max_rectangle_size(histogram):
    stack = []
    top = lambda: stack[-1]
    max_size = (0, 0, 0) # height, width and start position of the largest rectangle
    pos = 0 # current position in the histogram
    for pos, height in enumerate(histogram):
        start = pos # position where rectangle starts
        while True:
            if not stack or height > top().height:
                stack.append(Info(start, height)) # push
            elif stack and height < top().height:
                max_size = max(max_size, (top().height, (pos - top().start), top().start), key=area)
                start, _ = stack.pop()
                continue
            break # height == top().height goes here

    pos += 1
    for start, height in stack:
        max_size = max(max_size, (height, (pos - start), start), key=area)

    return max_size

def area(size):
    return size[0] * size[1]

def get_stats_from_group(reg_group, conf_img=None):
    """
    Get the coordinates of all pixels within the group and also mean of confidence
    :param reg_group:
    :param conf_img:
    :return:
    """
    coords = []
    for g in reg_group:
        coords.extend(g.coords)
    coords = np.array(coords)
    return coords

city = "austin"
city_name = "austin3"
targ_rgb_dir = r"/hdd/inria/train/images/"+city_name+".tif"
targ_gt_dir = r"/hdd/inria/train/gt/"+city_name+".tif"


targ_rgb = misc_utils.load_file(targ_rgb_dir)
targ_lbl = misc_utils.load_file(targ_gt_dir)/255

osc = eval_utils.ObjectScorer(min_th=0.5, link_r=10, eps=2)

#Get object groups
print("Extracting the building as objects...")
lbl_groups = osc.get_object_groups(targ_lbl)
rgb_groups = osc.get_object_groups(targ_lbl)

print("Number of building 'groups': ", len(lbl_groups))

# Get the colors for everything in every building
grp_id = 0
size_dict = dict()

# print("\n Replacing roofs for eligible buildings...")
counter = 0
score = dict()

file = open(r"/hdd/2019-bass-connections-aatb/mrs_new/object_exctraction/Aneesh/rooftops/dictionary/good_roofs_"+city_name+".txt", "w") 

for g_lbl in tqdm(lbl_groups):
# for g_lbl in lbl_groups:
    grp_id+=1
    grp1 = get_stats_from_group(g_lbl)
    area1 = sum([k.area for k in g_lbl])
    # if area1 >=85000:
    #     continue
    size_dict[grp_id] = area1
    grp1 = grp1.tolist()
    min_x = min([k[0] for k in list(grp1)])
    min_y = min([k[1] for k in list(grp1)])
    max_x = max([k[0] for k in list(grp1)])
    max_y = max([k[1] for k in list(grp1)])
    box_area = (max_x - min_x+1)*(max_y-min_y+1)
    if area1>3000:
        score[grp_id] = area1/box_area
    output = np.zeros((max_x-min_x+1,max_y-min_y+1,4), dtype = int)
    rect_map = np.zeros((max_x-min_x+1,max_y-min_y+1), dtype = int)
    count = 0
    try:
        for x in range(min_x, max_x+1):
            for y in range(min_y, max_y+1):
                if([x,y] in grp1):
                    count+=1
                    output[x-min_x][y-min_y] = np.append(np.flip(targ_rgb[x][y]), 255)
                    rect_map[x-min_x][y-min_y] = 255
                    # print(targ_rgb[x][y])
                else:
                    output[x-min_x][y-min_y] = np.array([0, 0, 0, 0])
                    rect_map[x-min_x][y-min_y] = 0
        # print("Count = ", count)
        output = np.array(output)
        rect_map = rect_map.tolist()
        (height, width, left_column, bottom_row) = max_rect(rect_map, value = 255)
        output2 = np.zeros((height,width,4), dtype = int)

        start_row = bottom_row-height+1
        for i in range(width):
            for j in range (height):
                output2[j][i] = output[start_row +j][left_column+i]
        cv2.imwrite("/hdd/2019-bass-connections-aatb/mrs_new/object_exctraction/Aneesh/rooftops/"+city+"_cropped/"+city_name+"_"+str(grp_id)+"_"+str(area1)+".png", output2)
        if(height*width>=800):
            file.write("/hdd/2019-bass-connections-aatb/mrs_new/object_exctraction/Aneesh/rooftops/"+city+"_cropped/"+city_name+"_"+str(grp_id)+"_"+str(area1)+".png" + "\n")
    except:
        print("ID failed:", grp_id)
        pass

file.close()

