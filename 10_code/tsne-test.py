# import libraries
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
perp = 50
print(activation_array.shape)
# tsne = TSNE(n_components =2, perplexity = perp, random_state=0)
tsne = TSNE(n_components =2, perplexity = perp, random_state=0, n_iter = 1000)

# print("labels", str(len(labels)))
# print(files[0:5])
# print(labels[0:5])

print("Starting TSNE...")
activated_tsne = tsne.fit_transform(activation_array)
print("Done With TSNE...")



plt.figure(figsize=(15, 15))
plt.xlim(activated_tsne[:, 0].min(), activated_tsne[:, 0].max() + 1)
plt.ylim(activated_tsne[:, 1].min(), activated_tsne[:, 1].max() + 1)
# colors = ["#476A2A", "#7851B8", "#BD3430", "#4A2D4E", "#875525",
#           "#A83683", "#4E655E", "#853541", "#3A3120", "#535D8E"]

colors = {'t': 'b', 'a': 'g', 'k': 'm' ,'v': 'c' ,'c': 'y'}

for i in range(len(activated_tsne)):
        # actually plot the points
        #plt.scatter(activated_tsne[i, 0], activated_tsne[i, 1])
        plt.text(activated_tsne[i, 0], activated_tsne[i, 1], str(labels[i]), color = colors[str(labels[i])[0]], fontdict={'weight': 'bold', 'size': 9})

plt.xlabel("t-SNE feature 0")
plt.xlabel("t-SNE feature 1")
markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in colors.values()]
plt.legend(markers, colors.keys(), numpoints=1)
plt.show()
plt.savefig('/hdd/2019-bass-connections-aatb/mrs_new/tSNE_out_all_patches_labelled_'+str(perp)+'.png')

plt.figure(figsize=(15, 15))
plt.xlim(activated_tsne[:, 0].min(), activated_tsne[:, 0].max() + 1)
plt.ylim(activated_tsne[:, 1].min(), activated_tsne[:, 1].max() + 1)

for i in range(len(activated_tsne)):
        # actually plot the points
        #plt.scatter(activated_tsne[i, 0], activated_tsne[i, 1])
        plt.scatter(activated_tsne[i, 0], activated_tsne[i, 1], color = colors[str(labels[i])[0]])

plt.xlabel("t-SNE feature 0")
plt.xlabel("t-SNE feature 1")
markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in colors.values()]
plt.legend(markers, colors.keys(), numpoints=1)
plt.show()
plt.savefig('/hdd/2019-bass-connections-aatb/mrs_new/tSNE_out_all_patches_scatter_'+str(perp)+'.png')

