import sys
import numpy as np
from tqdm import tqdm
from statistics import mean
import matplotlib.pyplot as plt
plt.switch_backend('agg')
sys.path.append(r'/hdd/2019-bass-connections-aatb/mrs_new')
from mrs_utils import misc_utils, vis_utils, eval_utils

city_names = [r"austin", r"chicago", r"vienna", r"kitsap", r"tyrol-w"]

# Get centroid of all coordinates (added by Gaurav)
def getcentroid(coords):
    coords = np.array(coords)
    length = coords.shape[0]
    sum_x = np.sum(coords[:, 0])
    sum_y = np.sum(coords[:, 1])
    return sum_x/length, sum_y/length


#Get coordinates out
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

for city in city_names:
# Specify paths
    print("\nCity: ", city)
    rgb_file = r"/hdd/inria/train/images/"+ city + r"1.tif"
    lbl_file = r"/hdd/inria/train/gt/" + city + r"1.tif"
    conf_file = r"/hdd/2019-bass-connections-aatb/mrs_new/results/ecresnet50_dcunet_dsinria_lre1e-03_lrd1e-02_ep25_bs5_ds50_dr0p1/" + city + r"1.npy"


    #load in file
    rgb = misc_utils.load_file(rgb_file)
    lbl_img, conf_img = misc_utils.load_file(lbl_file)/255, misc_utils.load_file(conf_file)

    #Prepare to extracto objects (Using Bohao's class)
    osc = eval_utils.ObjectScorer(min_th=0.5, link_r=10, eps=2)

    #Get object groups
    print("Extracting the objects...")
    lbl_groups = osc.get_object_groups(lbl_img)
    conf_groups = osc.get_object_groups(conf_img)

    print("Number of object 'groups': ", len(lbl_groups))

    # Get the colors for everything in every building
    i = 0
    colors_dict = dict()
    size_dict = dict()
    print("\n Getting all colors for all objects...")
    # Loop through all "object groups" (ie buildings)
    centroid_list = []
    for g_lbl in tqdm(lbl_groups):
        i+=1
        #Array of coords for current building
        #format: [[x1,y1], [x2,y2]]
        coords_lbl = get_stats_from_group(g_lbl)
        print(coords_lbl)
        print(getcentroid(coords_lbl))
        centroid_list.append(getcentroid(coords_lbl))
        rgb_list = []
        for k in coords_lbl:
            # k is a single coordinate, ie, [x y]
            rgb_list.append(np.flip(rgb[k[0]][k[1]]))
        size_dict[i] = sum([k.area for k in g_lbl])
        colors_dict[i] = rgb_list


    """
    colors_dict: key: building/object number
                value: list of [r,g,b] for each pixel in this building/object

    size_dict: key: building/object number
            value: area (size/# of pixels) in this building
    """
    # print centroids (Gaurav)
    np.save("/hdd/2019-bass-connections-aatb/mrs_new/texture_descriptors/similarityindex/centroids/"+ city + ".npy",centroid_list)
    
    # print(size_dict[291])
    # #Aggregate colors at building level
    # aggregated_dict = dict()
    # i = 0
    # print("\n Aggregating colors at building (object) level...")
    # for k in tqdm(colors_dict.keys()):
    #     i+=1
    #     avg_r = mean([p[0] for p in colors_dict[k]])
    #     avg_g = mean([p[1] for p in colors_dict[k]])
    #     avg_b = mean([p[2] for p in colors_dict[k]])
    #     aggregated_dict[i] = [avg_r, avg_g, avg_b]

    # """
    # aggregated_dict: key: building/object number
    #                 value: avg [r,g,b] for this building/object
    # """
    # x_axis = list(colors_dict.keys())
    # red = [aggregated_dict[k][0] for k in x_axis]
    # green = [aggregated_dict[k][1] for k in x_axis]
    # blue = [aggregated_dict[k][2] for k in x_axis]
    # size = [size_dict[k] for k in x_axis]


    # # Plot r,g,b and size
    # plt.figure()
    # plt.hist(red, bins = 20, color='r')
    # plt.xlabel('Average Red Pixel Value')
    # plt.ylabel('Number of Buildings')
    # plt.title('Red distribution for '+city + ' 1')
    # plt.savefig('/hdd/2019-bass-connections-aatb/mrs_new/object_exctraction/Aneesh/' + city +'_red-hist1.png')

    # plt.figure()
    # plt.hist(green, bins = 20, color='g')
    # plt.xlabel('Average Green Pixel Value')
    # plt.ylabel('Number of Buildings')
    # plt.title('Green distribution for '+city + ' 1')
    # plt.savefig('/hdd/2019-bass-connections-aatb/mrs_new/object_exctraction/Aneesh/' + city +'_green-hist1.png')

    # plt.figure()
    # plt.hist(blue, bins = 20, color='b')
    # plt.xlabel('Average Blue Pixel Value')
    # plt.ylabel('Number of Buildings')
    # plt.title('Blue distribution for '+city + ' 1')
    # plt.savefig('/hdd/2019-bass-connections-aatb/mrs_new/object_exctraction/Aneesh/' + city +'_blue-hist1.png')

    # print("Max building size: ", max(size))
    # print("Largest building is building #",size.index(max(size)+1))
    # plt.figure()
    # plt.plot(x_axis, size, color = 'grey')
    # plt.xlabel('Building number')
    # plt.ylabel('Building size')
    # plt.title('Size distribution for '+city + ' 1')
    # plt.savefig('/hdd/2019-bass-connections-aatb/mrs_new/object_exctraction/Aneesh/' + city +'_size-hist1.png')
