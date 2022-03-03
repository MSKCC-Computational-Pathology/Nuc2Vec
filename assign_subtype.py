# Copyright (c) 2021 MSK
import faiss
import argparse
import os
import numpy as np
from tqdm import tqdm
from scipy import stats
import pandas as pd
import torch


parser = argparse.ArgumentParser(description='assign subtype labels according to Nuc2Vec')
parser.add_argument('-n', '--nuclei', type=str, help='path to nuclei file')
parser.add_argument('-t', '--trained', type=str, help='path to reference nuclei subtyping file', default='./original_reference_subtyping.csv')
parser.add_argument('-r', '--root', type=str, help='path for saving', default='./')
parser.add_argument('-f', '--feature', type=str, help='path to features')
parser.add_argument('-q', '--query', type=str, help='path to query features')
parser.add_argument('-k', '--topk', type=int, help='number of nearest neighbours', default=1023)
args = parser.parse_args()

if args.query is None:
    args.query = args.feature
else:
    fk = np.load(args.feature)['arr_0']
    fq = torch.load(args.query).numpy()
save_dir = os.path.join(args.root, args.query.split("/")[-1].split(".")[0])
dim = fk.shape[1]
index = faiss.IndexFlatL2(dim)
index = faiss.index_cpu_to_all_gpus(index)
index.add(fk)
indices = []
distance = []
chunk = int(len(fq)/20) + 1
for i in tqdm(range(chunk)):
    if i == chunk-1:
       qk = fq[i * 20:]
    else:
       qk = fq[i*20:(i+1)*20]
    dist, idx = index.search(qk, args.topk + 1)
    indices.append(idx.astype(np.int32))
indices = np.concatenate(indices, 0)
labels = pd.read_csv(args.trained)['subtype'].to_numpy()
data = pd.read_csv(args.nuclei)
assignments = []
for i in indices:
    ass = labels[i]
    cluster_assignment = stats.mode(ass, axis=None)[0][0]
    assignments.append(cluster_assignment)
data['subtype'] = assignments
data['slide_id'] = args.query.split("/")[-1].split(".")[0]+'.svs'
data.to_csv(os.path.join(args.root, args.query.split("/")[-1].split(".")[0])+'.csv', index=False)

