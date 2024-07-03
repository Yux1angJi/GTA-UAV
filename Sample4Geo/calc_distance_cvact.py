import numpy as np
from sklearn.metrics import DistanceMetric
from sample4geo.dataset.cvact import CVACTDatasetTrain
import scipy.io as sio
import pickle
import torch

TOP_K = 128

dataset = CVACTDatasetTrain(data_folder = "./data/CVACT")

anuData = sio.loadmat('./data/CVACT/ACT_data.mat')

utm = anuData["utm"]
ids = anuData['panoIds']

idx2numidx = dataset.idx2numidx


train_ids_set = set(dataset.train_ids)
train_idsnum_list = []
 

utm_coords = dict()
utm_coords_list = []

for i, idx in enumerate(ids):
    
    idx = str(idx)
    
    if idx in train_ids_set:
    
        coordinates = (float(utm[i][0]), float(utm[i][1]))
        utm_coords[idx] = coordinates
        utm_coords_list.append(coordinates) 
        train_idsnum_list.append(idx2numidx[idx])
    
    
print("Length Train Ids:", len(utm_coords_list))

train_idsnum_lookup = np.array(train_idsnum_list)
    

print("Length of gps coords : " +str(len(utm_coords_list)))
print("Calculation...")

dist = DistanceMetric.get_metric("euclidean")
dm = dist.pairwise(utm_coords_list, utm_coords_list)
print("Distance Matrix:", dm.shape)


dm_torch = torch.from_numpy(dm)
dm_torch = dm_torch.fill_diagonal_(dm.max())


values, ids = torch.topk(dm_torch, k=TOP_K, dim=1, largest=False)

values_near_numpy = values.numpy()
ids_near_numpy = ids.numpy()

near_neighbors = dict()

for i, idnum in enumerate(train_idsnum_list):
    
    near_neighbors[idnum] = train_idsnum_lookup[ids_near_numpy[i]].tolist()

print("Saving...") 
with open("./data/CVACT/gps_dict.pkl", "wb") as f:
    pickle.dump(near_neighbors, f)
