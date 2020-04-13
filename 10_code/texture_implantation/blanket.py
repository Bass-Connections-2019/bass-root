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

# blur_radius = 0.00001
# threshold = 200


# texture_dir = r"/hdd/2019-bass-connections-aatb/texture_v1/texture_nets/data/output_austin_brown_roof/train5_1000.png"
# synth_rgb_dir = r"/hdd/2019-bass-connections-aatb/Synthetic Data/Synthinel/random_city_s_i_62_RGB.tif"
# synth_gt_dir = r"/hdd/2019-bass-connections-aatb/Synthetic Data/Synthinel/random_city_s_i_62_GT.tif"

# synth_gt = cv2.imread(synth_gt_dir)
# synth_rgb = cv2.imread(synth_rgb_dir)
# texture = cv2.imread(texture_dir)
# print("Texture size: ", texture.shape)
# print("Synth size: ", synth_rgb.shape)

# only_synth_bldngs = np.multiply(synth_rgb, synth_gt)
# cv2.imwrite("./Output/test_img.png", only_synth_bldngs)
# cv2.imwrite("./Output/gt_img_test.png", synth_gt*255)
# cv2.imwrite("./Output/rgb_img_test.png", synth_rgb)

# fname='./Output/test_img.png'
# img = Image.open(fname).convert('L')
# # img = Image.open(fname)
# img = np.asarray(img)
# # print("Values in tempimg:", np.unique(img))
# print(img.shape)

# # smooth the image (to remove small objects)
# imgf = ndimage.gaussian_filter(img, blur_radius)

# # find connected components
# labeled, nr_objects = ndimage.label(imgf > threshold) 
# print("Number of objects is {}".format(nr_objects))

# plt.imsave('./Output/extractedbuildings.png', labeled)
# print(np.unique(labeled))


# imglist = synth_rgb.tolist()

# got = False
# desired_group = 3

# for k in range(572):
#     for i in range(572):
#         #now img is 5000*5000*4 (4 things: r,g,b and building label)
#         if(labeled[k][i]) == desired_group and not got:
#             xi,yi = k, i
#             got = True
#         imglist[k][i].append(labeled[k][i])
# synth_rgb_labels = np.array(imglist)
# print("New synth size", synth_rgb.shape)

# count = 0
# for k in range(572):
#     for i in range(572):
#         if(synth_rgb_labels[k][i][3] == desired_group):
#             count+=1
#             synth_rgb[k][i] = texture[k-xi][i-yi]

# print("Count: ", count)
# cv2.imwrite("./Output/rgb_img_updated.png", synth_rgb)

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

roof_tex_dir = r"/hdd/2019-bass-connections-aatb/mrs_new/object_exctraction/Aneesh/roof_texture/austin_brown_roof.jpg"
targ_rgb_dir = r"/hdd/inria/train/images/vienna1.tif"
targ_gt_dir = r"/hdd/inria/train/gt/vienna1.tif"

roof_tex = cv2.imread(roof_tex_dir)
print("Texture size", roof_tex.shape)
texture_area = roof_tex.shape[0]*roof_tex.shape[1]
targ_rgb = cv2.imread(targ_rgb_dir)
targ_gt = cv2.imread(targ_gt_dir)

#load in file
targ_rgb = misc_utils.load_file(targ_rgb_dir)
cv2.imwrite(r"/hdd/2019-bass-connections-aatb/mrs_new/object_exctraction/Aneesh/output/original_satellite.png", targ_rgb)
# print("RGB Shape", rgb.shape)
targ_lbl = misc_utils.load_file(targ_gt_dir)/255

#Prepare to extract objects (Using Bohao's class)
osc = eval_utils.ObjectScorer(min_th=0.5, link_r=10, eps=2)

#Get object groups
print("Extracting the building as objects...")
lbl_groups = osc.get_object_groups(targ_lbl)

print("Number of building 'groups': ", len(lbl_groups))

# Get the colors for everything in every building
i = 0
size_dict = dict()

# print("\n Replacing roofs for eligible buildings...")
counter = 0
for g_lbl in tqdm(lbl_groups):
    i+=1
    # coords_lbl is the list of coords, [[x1 y1] [x2 y2] ....]
    coords_lbl = get_stats_from_group(g_lbl)
    #area = num of pixels in roof
    group_area = sum([k.area for k in g_lbl])
    grp_id = i
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
            targ_rgb[k[0]][k[1]] = roof_tex[k[0]-xi][k[1]-yi]
    size_dict[i] = group_area


print(counter, " building roofs changed...")
cv2.imwrite(r"/hdd/2019-bass-connections-aatb/mrs_new/object_exctraction/Aneesh/implanted/vienna_implanted.png", targ_rgb)






