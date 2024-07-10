from torch.utils.data import Dataset
import numpy as np
import cv2
import random


class MixDatasetTrain(Dataset):
    
    def __init__(self,
                 batch_size,
                 transforms_query=None,
                 transforms_gallery=None,
                 prob_flip=0.5,
                 ):
        super().__init__()
        
        self.transforms_query = transforms_query
        self.transforms_gallery = transforms_gallery
        self.prob_flip = prob_flip
        self.batch_size = batch_size
        self.samples = []

    def update(self, samples_list):
        chunks = []
        for samples in samples_list:
            chunks.extend([samples[i:i + self.batch_size] for i in range(0, len(samples), self.batch_size)])
        random.shuffle(chunks)
        self.samples = [item for chunk in chunks for item in chunk]


    def __getitem__(self, index):
        
        query_img_path, gallery_img_path, positive_weight = self.samples[index]
        
        # for query there is only one file in folder
        query_img = cv2.imread(query_img_path)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        
        gallery_img = cv2.imread(gallery_img_path)
        gallery_img = cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB)
        
        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            gallery_img = cv2.flip(gallery_img, 1) 
        
        # image transforms
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']
        
        if self.transforms_gallery is not None:
            gallery_img = self.transforms_gallery(image=gallery_img)['image']
        
        return query_img, gallery_img, positive_weight

    def __len__(self):
        return len(self.samples)