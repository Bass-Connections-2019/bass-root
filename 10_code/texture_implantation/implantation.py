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
import os
import random


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


def create_texture_lists(texture_directory, small_threshold, large_threshold):
    file_list = [f for f in os.listdir(texture_directory)]
    small_texture_list = []
    medium_texture_list = []
    large_texture_list = []
    for f in file_list:
        if f.endswith(".png"):
            size = int(f.split("_")[2].split(".")[0])
            img = cv2.imread(texture_directory + f)
            if size <= small_threshold:
                small_texture_list.append(img)
            elif size >= large_threshold:
                large_texture_list.append(img)
            else:
                medium_texture_list.append(img)
    return small_texture_list, medium_texture_list, large_texture_list


def implant_textures(rgb_input_path, gt_input_path, stl, mtl, ltl, small_threshold, large_threshold):
    targ_rgb = cv2.imread(rgb_input_path)
    targ_gt = cv2.imread(gt_input_path)

    #load in file
    targ_rgb = misc_utils.load_file(rgb_input_path)
    cv2.imwrite(r"/hdd/2019-bass-connections-aatb/mrs_new/object_exctraction/Aneesh/implantation_with_size/original_satellite.png", targ_rgb)
    # print("RGB Shape", rgb.shape)
    targ_lbl = misc_utils.load_file(gt_input_path)/255

    #Prepare to extract objects (Using Bohao's class)
    osc = eval_utils.ObjectScorer(min_th=0.5, link_r=10, eps=2)

    #Get object groups
    print("Extracting the building as objects...")
    lbl_groups = osc.get_object_groups(targ_lbl)

    print("Number of building 'groups': ", len(lbl_groups))
    # print("\n Replacing roofs for eligible buildings...")
    counter = 0
    # Get the colors for everything in every building
    i = 0
    size_dict = dict()
    for g_lbl in tqdm(lbl_groups):
        i+=1
        # coords_lbl is the list of coords, [[x1 y1] [x2 y2] ....]
        coords_lbl = get_stats_from_group(g_lbl)
        #area = num of pixels in roof
        group_area = sum([k.area for k in g_lbl])
        grp_id = i

        if group_area <= small_threshold:
            roof_tex = random.choice(stl)
        elif group_area >= large_threshold:
            roof_tex = random.choice(ltl)
        else:
            roof_tex = random.choice(mtl)
        texture_area = roof_tex.shape[0]*roof_tex.shape[1]
        #getting top left of the target roof
        xi, yi = coords_lbl[0]

        # This is the 'logic' is used.
        # I just match up top left of target roof with top left of texture.
        # Not entirely correct, here I just say if target building area is less than texture area
        # But this fails for complex geometries, where target area is less than textures, but 
        # still falls out of the texture box
        if(group_area<=texture_area):
            counter+=1
            for k in coords_lbl:
                # k is a single coordinate, ie, [x y]
                # matching top left to top left.
                try:
                    targ_rgb[k[0]][k[1]] = roof_tex[k[0]-xi][k[1]-yi]
                except:
                    print("an exception has occured")
        size_dict[i] = group_area
    print(counter, " building roofs changed...")
    cv2.imwrite(r"/hdd/2019-bass-connections-aatb/mrs_new/object_exctraction/Aneesh/implantation_with_size/austin_implanted.png", targ_rgb)

def main():
    small_building_threshold = 1500
    large_building_threshold = 4000
    stl, mtl, ltl = create_texture_lists("/hdd/2019-bass-connections-aatb/texture_v1/texture_nets/automated_resources/output/final_output/", small_building_threshold, large_building_threshold)
    implant_textures("/hdd/inria/train/images/austin1.tif", "/hdd/inria/train/gt/austin1.tif", stl, mtl, ltl, small_building_threshold, large_building_threshold)

main()
