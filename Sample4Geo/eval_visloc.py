import os
import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader

from sample4geo.dataset.visloc import VisLocDatasetEval, get_transforms
from sample4geo.evaluate.gta import evaluate
from sample4geo.model import TimmModel


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
    dataset: str = 'VisLoc-D2S'           # 'U1652-D2S' | 'U1652-S2D'

    # Checkpoint to start from
    # checkpoint_start = 'pretrained/university/convnext_base.fb_in22k_ft_in1k_384/weights_e1_0.9515.pth'
    # checkpoint_start = 'work_dir/gta/convnext_base.fb_in22k_ft_in1k_384/0703171314/weights_end.pth'
    checkpoint_start = None

    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 4 
    
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 


#-----------------------------------------------------------------------------#
# Config                                                                      #
#-----------------------------------------------------------------------------#

config = Configuration() 


if config.dataset == 'VisLoc-D2S':
    config.train_pairs_meta_file = '/home/xmuairmud/data/UAV_VisLoc_dataset/data1234_z3/train_pair_meta.pkl'
    config.test_pairs_meta_file = '/home/xmuairmud/data/UAV_VisLoc_dataset/data1234_z3/test_pair_meta.pkl'
    # config.data_root_dir = '/home/xmuairmud/data/UAV_VisLoc_dataset/data_1_2/test/satellite'
    config.data_root_dir = '/home/xmuairmud/data/UAV_VisLoc_dataset/data1234_z3/all_satellite'


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


    # Test query
    query_dataset_test = VisLocDatasetEval(pairs_meta_file=config.test_pairs_meta_file,
                                        mode="drone",
                                        transforms=val_transforms,
                                        )
    query_img_list = query_dataset_test.images
    pairs_drone2sate_dict = query_dataset_test.pairs_drone2sate_dict
    
    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    # Test gallery
    gallery_dataset_test = VisLocDatasetEval(pairs_meta_file=config.test_pairs_meta_file,
                                               mode="sate",
                                               transforms=val_transforms,
                                               data_root_dir=config.data_root_dir,
                                               )
    gallery_img_list = gallery_dataset_test.images
    
    gallery_dataloader_test = DataLoader(gallery_dataset_test,
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    print("Query Images Test:", len(query_dataset_test))
    print("Gallery Images Test:", len(gallery_dataset_test))
    
    print("\n{}[{}]{}".format(30*"-", "UAV-VisLoc", 30*"-"))  

    r1_test = evaluate(config=config,
                        model=model,
                        query_loader=query_dataloader_test,
                        gallery_loader=gallery_dataloader_test, 
                        query_list=query_img_list,
                        gallery_list=gallery_img_list,
                        pairs_dict=pairs_drone2sate_dict,
                        ranks_list=[1, 5, 10],
                        step_size=1000,
                        cleanup=True)
 
