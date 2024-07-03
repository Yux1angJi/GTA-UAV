import os
import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader

from sample4geo.dataset.university import get_transforms
from sample4geo.dataset.custom_query import CustomData
from sample4geo.model import TimmModel

from sample4geo.evaluate.query_topn import QueryTopN

import matplotlib.pyplot as plt

import cv2

@dataclass
class Configuration:

    # Model
    model: str = 'convnext_base.fb_in22k_ft_in1k_384'
    
    # Override model image size
    img_size: int = 384
    
    # Evaluation
    batch_size: int = 128
    verbose: bool = True
    gpu_ids: tuple = (0,)
    normalize_features: bool = True
    eval_gallery_n: int = -1             # -1 for all or int
    
    # Dataset
    dataset: str = 'U1652-D2S'           # 'U1652-D2S' | 'U1652-S2D'
    data_folder: str = "./data/U1652"
    
    # Checkpoint to start from
    # checkpoint_start = 'pretrained/university/convnext_base.fb_in22k_ft_in1k_384/weights_e1_0.9515.pth'
    checkpoint_start = '/home/xmuairmud/jyx/ExtenGeo/Sample4Geo/university/convnext_base.fb_in22k_ft_in1k_384/0627211134/weights_end.pth'
  
    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 4 
    
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 


#-----------------------------------------------------------------------------#
# Config                                                                      #
#-----------------------------------------------------------------------------#

config = Configuration() 

# if config.dataset == 'U1652-D2S':
#     config.query_folder_train = './data/U1652/train/satellite'
#     config.gallery_folder_train = './data/U1652/train/drone'   
#     config.query_folder_test = './data/U1652/test/query_drone' 
#     config.gallery_folder_test = './data/U1652/test/gallery_satellite'    
# elif config.dataset == 'U1652-S2D':
#     config.query_folder_train = './data/U1652/train/satellite'
#     config.gallery_folder_train = './data/U1652/train/drone'    
#     config.query_folder_test = './data/U1652/test/query_satellite'
#     config.gallery_folder_test = './data/U1652/test/gallery_drone'
 
config.query_folder_test = '/home/xmuairmud/data/work/map_data/map2/query_drone' 
config.gallery_folder_test = '/home/xmuairmud/data/work/map_data/map2/gallery_satellite'    


if __name__ == '__main__':

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
        
    print("\nModel: {}".format(config.model))


    model = TimmModel(config.model,
                          pretrained=True,
                          img_size=config.img_size)
                          
    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = (config.img_size, config.img_size)
    

    # load pretrained Checkpoint    
    if config.checkpoint_start is not None:  
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)  
        model.load_state_dict(model_state_dict, strict=False)     

    # Data parallel
    print("GPUs available:", torch.cuda.device_count())  
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
            
    # Model to device   
    model = model.to(config.device)

    print("\nImage Size Query:", img_size)
    print("Image Size Ground:", img_size)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std)) 


    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#

    # Transforms
    val_transforms, train_sat_transforms, train_drone_transforms = get_transforms(img_size, mean=mean, std=std)

    # Reference Satellite Images
    query_dataset_test = CustomData(root_dir=config.query_folder_test,
                                               transforms=val_transforms,
                                               )
    query_path_list = query_dataset_test.imgs_list
    
    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    # Query Ground Images Test
    gallery_dataset_test = CustomData(root_dir=config.gallery_folder_test,
                                               transforms=val_transforms,
                                               )
    gallery_path_list = gallery_dataset_test.imgs_list
    
    gallery_dataloader_test = DataLoader(gallery_dataset_test,
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)                                                                                                               
    
    

    results = QueryTopN(config, model, query_dataloader_test, query_path_list, gallery_dataloader_test, gallery_path_list)

    rows = len(results)
    fig, axes = plt.subplots(nrows=rows, ncols=11, figsize=(20, 2))
    for i, result in enumerate(results):
        print(result)
        size = 256, 256

        query_img = result[0]
        query_img = cv2.imread(query_img)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        query_img = cv2.resize(query_img, size)
        
        ax = axes[i, 0]
        ax.imshow(query_img, cmap='gray')
        ax.set_title("Query")
        ax.axis('off')

        for j in range(10):
            ax = axes[i, j+1]
            topj_img = result[j+1]
            topj_img = cv2.imread(topj_img)
            topj_img = cv2.cvtColor(topj_img, cv2.COLOR_BGR2RGB)
            topj_img = cv2.resize(topj_img, size)
            ax.imshow(topj_img, cmap='gray')
            ax.set_title(f"Top {j+1}")
            ax.axis('off')
    fig.set_dpi(100)

    plt.tight_layout()
    plt.savefig('demo_extend_des.png')


        

 
