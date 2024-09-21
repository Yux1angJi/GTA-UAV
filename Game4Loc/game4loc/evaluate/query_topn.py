import torch
import numpy as np
from tqdm import tqdm
import gc
import time
from torch.cuda.amp import autocast
import torch.nn.functional as F
import matplotlib.pyplot as plt


def predict(train_config, model, dataloader):
    
    model.eval()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
        
    img_features_list = []

    with torch.no_grad():
        
        for img in bar:
        
            with autocast():
         
                img = img.to(train_config.device)
                img_feature = model(img)
            
                # normalize is calculated in fp32
                if train_config.normalize_features:
                    img_feature = F.normalize(img_feature, dim=-1)
            
            # save features in fp32 for sim calculation
            img_features_list.append(img_feature.to(torch.float32))
      
        # keep Features on GPU
        img_features = torch.cat(img_features_list, dim=0) 
        
    if train_config.verbose:
        bar.close()
        
    return img_features


def QueryTopN(
        config,
        model,
        query_loader,
        query_path_list,
        gallery_loader,
        gallery_path_list,
        top_n=1):
    
    print("Extract Features:")
    img_features_query = predict(config, model, query_loader)
    img_features_gallery = predict(config, model, gallery_loader)
    
    print("Compute Scores:")
    results = []
    for i in tqdm(range(len(img_features_query))):

        img_features_query_i = img_features_query[i]
        img_features_query_i = img_features_query_i.unsqueeze(-1)

        similarity = torch.matmul(img_features_gallery, img_features_query_i)

        sorted_values, sorted_indices = torch.sort(similarity.squeeze(), descending=True)
        top_indices = sorted_indices[:top_n].cpu().numpy()

        result_tmp = [query_path_list[i]]
        for index in top_indices:
            result_tmp.append(gallery_path_list[index])

        results.append(result_tmp)

    return results



    
