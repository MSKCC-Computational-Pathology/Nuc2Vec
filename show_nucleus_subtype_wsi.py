import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import argparse
import random
import openslide

IMG_SIZE = 128
FSIZE = 1024

parser = argparse.ArgumentParser(description='show nearest neighbour')
parser.add_argument('-c', '--clustering-result', type=str, help='path to data file')
parser.add_argument('-r', '--img-dir', type=str, help='path to image file')
parser.add_argument('-o', '--root', type=str, help='save root')
args = parser.parse_args()


data = pd.read_csv(args.clustering_result)
subtypes = pd.unique(data['subtype'])

for idx in subtypes:
    selected = data[data['subtype'] == idx]
    print("cluster"+str(idx)+": "+str(len(selected))+" instances...")
    if len(selected) > 10000:
        selected = selected.sample(min(10000, len(selected)))
    elif len(selected) < 100:
        continue

    patches = []
    for i, sample in selected.iterrows():
        try:
            slide = openslide.OpenSlide(os.path.join(args.img_dir, sample['slide_id']))
        except Exception:
            continue
        center = np.array([sample['x'], sample['y']]).astype(np.int)
        reg = slide.read_region(center - int(IMG_SIZE / 4), 0,
                                (int(IMG_SIZE / 2), int(IMG_SIZE / 2)))
        img = reg.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
        patches.append(img)
        if len(patches) == 100:
            break

    r = np.floor(np.sqrt(len(patches))).astype(np.int)
    fig, ax = plt.subplots(nrows=r, ncols=r, figsize=(r, r))
    i = 0
    for row in ax:
        for col in row:
            item = selected.iloc[i]
            col.imshow(patches[i])
            col.axis('off')
            # col.set_title(item['type'], fontsize=8, pad=0)
            i = i + 1

    fig.suptitle("subtype" + str(idx+1), y=0.9, fontsize='x-large', fontweight='bold')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(os.path.join(args.root, "subtype" + str(idx+1) + '.png'), dpi=1000, bbox_inches='tight')
    plt.close()

