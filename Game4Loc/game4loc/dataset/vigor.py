import cv2
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import random
import copy
import torch
from tqdm import tqdm
from collections import defaultdict
import time
 
class VigorDatasetTrain(Dataset):
    
    def __init__(self,
                 data_folder,
                 same_area=True,
                 transforms_query=None,
                 transforms_reference=None,
                 prob_flip=0.0,
                 prob_rotate=0.0,
                 shuffle_batch_size=128,
                 ):
        
        super().__init__()
 
        self.data_folder = data_folder
        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        self.shuffle_batch_size = shuffle_batch_size
        
        self.transforms_query = transforms_query           # ground
        self.transforms_reference = transforms_reference   # satellite
        
        if same_area:
            self.cities = ['Chicago', 'NewYork', 'SanFrancisco', 'Seattle']
        else:
            self.cities = ['NewYork', 'Seattle'] 

        # load sat list 
        sat_list = []
        for city in self.cities:
            df_tmp = pd.read_csv(f'{data_folder}/splits/{city}/satellite_list.txt', header=None, delim_whitespace=True)
            df_tmp = df_tmp.rename(columns={0: "sat"})
            df_tmp["path"] = df_tmp.apply(lambda x: f'{data_folder}/satellite/{city}/{x.sat}', axis=1)
            sat_list.append(df_tmp)
        self.df_sat = pd.concat(sat_list, axis=0).reset_index(drop=True)
        
        # idx for complete train and test independent of mode = train or test
        sat2idx = dict(zip(self.df_sat.sat, self.df_sat.index))
        self.idx2sat = dict(zip(self.df_sat.index, self.df_sat.sat))
        self.idx2sat_path = dict(zip(self.df_sat.index, self.df_sat.path))
        
        
        # ground dependent on mode 'train' or 'test'
        ground_list = []
        for city in self.cities:

            if same_area:
                df_tmp = pd.read_csv(f'{data_folder}/splits/{city}/same_area_balanced_train.txt', header=None, delim_whitespace=True)
            else:
                df_tmp = pd.read_csv(f'{data_folder}/splits/{city}/pano_label_balanced.txt', header=None, delim_whitespace=True)
            
            df_tmp = df_tmp.loc[:, [0, 1, 4, 7, 10]].rename(columns={0:  "ground",
                                                                     1:  "sat",
                                                                     4:  "sat_np1",
                                                                     7:  "sat_np2",
                                                                     10: "sat_np3"})
            
            df_tmp["path_ground"] = df_tmp.apply(lambda x: f'{data_folder}/ground/{city}/{x.ground}', axis=1)
            df_tmp["path_sat"] = df_tmp.apply(lambda x: f'{data_folder}/satellite/{city}/{x.sat}', axis=1)
            
            for sat_n in ["sat", "sat_np1", "sat_np2", "sat_np3"]:
                df_tmp[f'{sat_n}'] = df_tmp[f'{sat_n}'].map(sat2idx)
                
            ground_list.append(df_tmp) 
        self.df_ground = pd.concat(ground_list, axis=0).reset_index(drop=True)
        
        # idx for split train or test dependent on mode = train or test
        self.idx2ground = dict(zip(self.df_ground.index, self.df_ground.ground))
        self.idx2ground_path = dict(zip(self.df_ground.index, self.df_ground.path_ground))
                
      
        self.pairs = list(zip(self.df_ground.index, self.df_ground.sat))
        self.idx2pairs = defaultdict(list)
        
        # for a unique sat_id we can have 1 or 2 ground views as gt
        for pair in self.pairs:      
            self.idx2pairs[pair[1]].append(pair)
            
            
        self.label = self.df_ground[["sat", "sat_np1", "sat_np2", "sat_np3"]].values 
        
        self.samples = copy.deepcopy(self.pairs)
            

    def __getitem__(self, index):
        
        idx_ground, idx_sat = self.samples[index]
        
        # load query -> ground image
        query_img = cv2.imread(self.idx2ground_path[idx_ground])
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        
        # load reference -> satellite image
        reference_img = cv2.imread(self.idx2sat_path[idx_sat])
        reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)

            
        # Flip simultaneously query and reference
        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            reference_img = cv2.flip(reference_img, 1) 
        
        # image transforms
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']
            
        if self.transforms_reference is not None:
            reference_img = self.transforms_reference(image=reference_img)['image']
                
        # Rotate simultaneously query and reference
        if np.random.random() < self.prob_rotate:
        
            r = np.random.choice([1,2,3])
            
            # rotate sat img 90 or 180 or 270
            reference_img = torch.rot90(reference_img, k=r, dims=(1, 2)) 
            
            # use roll for ground view if rotate sat view
            c, h, w = query_img.shape
            shifts = - w//4 * r
            query_img = torch.roll(query_img, shifts=shifts, dims=2)   
                   
            
        label = torch.tensor(idx_sat, dtype=torch.long)  
        
        return query_img, reference_img, label
    
    def __len__(self):
        return len(self.samples)
        
        
            
    def shuffle(self, sim_dict=None, neighbour_select=8, neighbour_range=16):

            '''
            custom shuffle function for unique class_id sampling in batch
            '''
            
            print("\nShuffle Dataset:")
            
            pair_pool = copy.deepcopy(self.pairs)
            idx2pair_pool = copy.deepcopy(self.idx2pairs)
            
            neighbour_split = neighbour_select // 2
                        
            if sim_dict is not None:
                similarity_pool = copy.deepcopy(sim_dict)                
            
            # Shuffle pairs order
            random.shuffle(pair_pool)
           
            
            # Lookup if already used in epoch
            pairs_epoch = set()   
            idx_batch = set()
     
            
            # buckets
            batches = []
            current_batch = []
            
            
            # counter
            break_counter = 0
            
            # progressbar
            pbar = tqdm()
    
            while True:
                
                pbar.update()
                
                if len(pair_pool) > 0:
                    pair = pair_pool.pop(0)
                    
                    _, idx = pair
                    
                    if idx not in idx_batch and pair not in pairs_epoch and len(current_batch) < self.shuffle_batch_size:
                        
                        idx_batch.add(idx)
                        current_batch.append(pair)
                        pairs_epoch.add(pair)
                        
                        # remove from pool used for sim-sampling
                        idx2pair_pool[idx].remove(pair)

                        if sim_dict is not None and len(current_batch) < self.shuffle_batch_size:
                            
                            near_similarity = copy.deepcopy(similarity_pool[idx][:neighbour_range])
                            near_always = copy.deepcopy(near_similarity[:neighbour_split]) 
                            near_random = copy.deepcopy(near_similarity[neighbour_split:])
                            random.shuffle(near_random)
                            near_random = near_random[:neighbour_split]
                            near_similarity_select = near_always + near_random

                            
                            for idx_near in near_similarity_select:
                            
                            
                                # check for space in batch
                                if len(current_batch) >= self.shuffle_batch_size:
                                    break
                            
                                # no check for pair in epoch necessary cause all we add is removed from pool
                                if idx_near not in idx_batch:
                            
                                    near_pairs = copy.deepcopy(idx2pair_pool[idx_near])
                                    
                                    # up to 2 for one sat view 
                                    random.shuffle(near_pairs)
                                
                                    for near_pair in near_pairs:
                                                                                    
                                        idx_batch.add(idx_near)
                                        current_batch.append(near_pair)
                                        pairs_epoch.add(near_pair)
                                        
                                        idx2pair_pool[idx_near].remove(near_pair)
                                        similarity_pool[idx].remove(idx_near)
                                        
                                        # only select one view
                                        break
                             
                        break_counter = 0
                        
                    else:
                        # if pair fits not in batch and is not already used in epoch -> back to pool
                        if pair not in pairs_epoch:
                            pair_pool.append(pair)
                            
                        break_counter += 1
                        
                    if break_counter >= 1024:
                        break
                   
                else:
                    break

                if len(current_batch) >= self.shuffle_batch_size:
                
                    # empty current_batch bucket to batches
                    batches.extend(current_batch)
                    idx_batch = set()
                    current_batch = []
       
            pbar.close()
            
            # wait before closing progress bar
            time.sleep(0.3)
            
            self.samples = batches
            print("pair_pool:", len(pair_pool))
            print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples))) 
            print("Break Counter:", break_counter)
            print("Pairs left out of last batch to avoid creating noise:", len(self.pairs) - len(self.samples))
            print("First Element ID: {} - Last Element ID: {}".format(self.samples[0][1], self.samples[-1][1]))  

            
       
class VigorDatasetEval(Dataset):
    
    def __init__(self,
                 data_folder,
                 split,
                 img_type,
                 same_area=True,
                 transforms=None,
                 ):
        
        super().__init__()
 
        self.data_folder = data_folder
        self.split = split
        self.img_type = img_type
        self.transforms = transforms
        
            
        if same_area:
            self.cities = ['Chicago', 'NewYork', 'SanFrancisco', 'Seattle']
        else:
            if split == "train":
                self.cities = ['NewYork', 'Seattle'] 
            else:
                self.cities = ['Chicago', 'SanFrancisco'] 
               
        # load sat list 
        sat_list = []
        for city in self.cities:
            df_tmp = pd.read_csv(f'{data_folder}/splits/{city}/satellite_list.txt', header=None, delim_whitespace=True)
            df_tmp = df_tmp.rename(columns={0: "sat"})
            df_tmp["path"] = df_tmp.apply(lambda x: f'{data_folder}/satellite/{city}/{x.sat}', axis=1)
            sat_list.append(df_tmp)
        self.df_sat = pd.concat(sat_list, axis=0).reset_index(drop=True)
        
        # idx for complete train and test independent of mode = train or test
        sat2idx = dict(zip(self.df_sat.sat, self.df_sat.index))
        self.idx2sat = dict(zip(self.df_sat.index, self.df_sat.sat))
        self.idx2sat_path = dict(zip(self.df_sat.index, self.df_sat.path))
        
        
        # ground dependent on mode 'train' or 'test'
        ground_list = []
        for city in self.cities:
            
            if same_area:
                df_tmp = pd.read_csv(f'{data_folder}/splits/{city}/same_area_balanced_{split}.txt', header=None, delim_whitespace=True)
            else:
                df_tmp = pd.read_csv(f'{data_folder}/splits/{city}/pano_label_balanced.txt', header=None, delim_whitespace=True)
  
            
            df_tmp = df_tmp.loc[:, [0, 1, 4, 7, 10]].rename(columns={0:  "ground",
                                                                     1:  "sat",
                                                                     4:  "sat_np1",
                                                                     7:  "sat_np2",
                                                                     10: "sat_np3"})
            
            df_tmp["path_ground"] = df_tmp.apply(lambda x: f'{data_folder}/ground/{city}/{x.ground}', axis=1)
            df_tmp["path_sat"] = df_tmp.apply(lambda x: f'{data_folder}/satellite/{city}/{x.sat}', axis=1)
            
            df_tmp["path_sat_np1"] = df_tmp.apply(lambda x: f'{data_folder}/satellite/{city}/{x.sat_np1}', axis=1)
            df_tmp["path_sat_np2"] = df_tmp.apply(lambda x: f'{data_folder}/satellite/{city}/{x.sat_np2}', axis=1)
            df_tmp["path_sat_np3"] = df_tmp.apply(lambda x: f'{data_folder}/satellite/{city}/{x.sat_np3}', axis=1)

            
            for sat_n in ["sat", "sat_np1", "sat_np2", "sat_np3"]:
                df_tmp[f'{sat_n}'] = df_tmp[f'{sat_n}'].map(sat2idx)
                
            ground_list.append(df_tmp) 
        self.df_ground = pd.concat(ground_list, axis=0).reset_index(drop=True)
        
        # idx for split train or test dependent on mode = train or test
        self.idx2ground = dict(zip(self.df_ground.index, self.df_ground.ground))
        self.idx2ground_path = dict(zip(self.df_ground.index, self.df_ground.path_ground))
        
        
        if self.img_type == "reference":
            if split == "train":
                # only sat images we really train on
                self.label = self.df_ground["sat"].unique()
                self.images = []
                for idx in self.label:
                    self.images.append(self.idx2sat_path[idx])
            else:
                # all sat images of cities in split
                self.images = self.df_sat["path"].values
                self.label = self.df_sat.index.values
            
        elif self.img_type == "query":
            self.images = self.df_ground["path_ground"].values
            self.label = self.df_ground[["sat", "sat_np1", "sat_np2", "sat_np3"]].values 

        else:
            raise ValueError("Invalid 'img_type' parameter. 'img_type' must be 'query' or 'reference'")
                

    def __getitem__(self, index):
        
        img_path = self.images[index]
        label = self.label[index]
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
            
        label = torch.tensor(label, dtype=torch.long)

        return img, label

    def __len__(self):
        return len(self.images)

            





