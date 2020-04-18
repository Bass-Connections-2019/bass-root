"""
@Author: Aneesh Gupta
This works in conjunction with the Models for Remote Sensing codebase
https://github.com/bohaohuang/mrs

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

targ_rgb_dir = r"/hdd/inria/train/images/austin1.tif"
targ_gt_dir = r"/hdd/inria/train/gt/austin1.tif"
city_name = "austin"

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
    count = 0
    try:
        for x in range(min_x, max_x+1):
            for y in range(min_y, max_y+1):
                if([x,y] in grp1):
                    count+=1
                    output[x-min_x][y-min_y] = np.append(np.flip(targ_rgb[x][y]), 255)
                    # print(targ_rgb[x][y])
                else:
                    output[x-min_x][y-min_y] = np.array([0, 0, 0, 0])
        # print("Count = ", count)
        output = np.array(output)
        # print(output.shape)
        # output = cv2.bitwise_not(output)
        cv2.imwrite("/hdd/2019-bass-connections-aatb/mrs_new/object_exctraction/Aneesh/rooftops/"+city_name+"/"+city_name+"_"+str(grp_id)+"_"+str(area1)+".png", output)
    except:
        print("ID failed:", grp_id)
        pass

best_ids = sorted(list(score.keys()), key = lambda x: score[x], reverse = True)

file = open(r"/hdd/2019-bass-connections-aatb/mrs_new/object_exctraction/Aneesh/rooftops/best_roofs_"+city_name+".txt", "w") 
print("Calculating best roofs for cropping")
for i in range(30):
    id = best_ids[i]
    print("ID:", id, "Score:", score[id])
    file.write("ID: " + str(id)+" Score: " + str(score[id]) + "\n")
file.close() 
# print("Building id:", grp_id)
# print("Number of pixels: ", len(grp1))
# grp1 = grp1.tolist()
# min_x = min([k[0] for k in list(grp1)])
# min_y = min([k[1] for k in list(grp1)])
# max_x = max([k[0] for k in list(grp1)])
# max_y = max([k[1] for k in list(grp1)])
# box_area = (max_x - min_x)*(max_y-min_y)
# print("Anchor points: " , min_x , min_y , max_x, max_y)
# print("Actual area:" , area1)
# print("Box area: ", box_area)
# print("City size", targ_rgb.shape)

# output = np.zeros((max_x-min_x+1,max_y-min_y+1,4), dtype = int)
# print(output.shape)
# count = 0
# for x in tqdm(range(min_x, max_x+1)):
#     for y in range(min_y, max_y+1):
#         if([x,y] in grp1):
#             count+=1
#             output[x-min_x][y-min_y] = np.append(targ_rgb[x][y].astype(int), 255)
#             # output[x-min_x][y-min_y] = np.array([int(k) for k in list(targ_rgb[x][y])])
#         else:
#             output[x-min_x][y-min_y] = np.array([0, 0, 0, 0])

# print("Count = ", count)
# output = np.array(output)
# print(output.shape)
# cv2.imwrite("/hdd/2019-bass-connections-aatb/mrs_new/object_exctraction/Aneesh/rooftops/test_output.png", output)