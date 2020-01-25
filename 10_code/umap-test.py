import umap
import numpy as np
import pandas as pd
import pickle
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm
import random
from sklearn.datasets import load_digits

"""
Code
"""

"""
Importing input data files
"""
# initialize empty list to save activations at 2nd last layer
act_layer = []

# define path to all pickle files
files = list(glob.glob(os.path.join('/hdd/2019-bass-connections-aatb/mrs_new/results_all_patches/ecresnet50_dcunet_dsinria_lre1e-03_lrd1e-02_ep25_bs5_ds50_dr0p1','*.pkl')))
random.shuffle(files)
# define function for opening and getting information from pickle file
def pic(file):
    infile = open(str(file), 'rb')
    inter = pickle.load(infile)
    infile.close()
    return inter

count = 0
labels = []
print("Starting to read...")
# for file in tqdm(files):
for file in tqdm(files):
    count+=1
    patch_act = pic(file)
    layer = patch_act[7].reshape(-1)
    act_layer.append(layer)
    label = str(file)
    # label = label.split('/')[-1].split('_')[0]
    # label = label[0]+label[-1]
    label = label.split('/')[-1].split('_')
    label = label[0][0] + label[0][-1] + '_' + label[-2] + '_' + label[-1]
    label = label.split('.')[0].split('_')[0] + label.split('.')[0].split('_')[-1]
    labels.append(label)

print("Done Reading...")
print("Read", count, "files")

activation_array = np.array(act_layer)

print("Starting UMap transformation")
embedding = umap.UMAP().fit_transform(activation_array)
print("Finished UMap transformation")

mapper = umap.UMAP().fit(activation_array)
umap.plot.points(mapper, label="test label")
