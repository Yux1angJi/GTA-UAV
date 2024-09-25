import os
import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader

from game4loc.dataset.gta import GTADatasetEval, get_transforms
from game4loc.evaluate.gta import evaluate
from game4loc.model import DesModel


@dataclass
class Configuration:

    # Model
    # model: str = 'convnext_base.fb_in22k_ft_in1k_384'
    model: str = 'vit_base_patch16_rope_reg1_gap_256.sbb_in1k'
    
    # Override model image size
    img_size: int = 384
    
    # Evaluation
    batch_size: int = 128
    verbose: bool = True
    gpu_ids: tuple = (0,1)
    normalize_features: bool = True
    eval_gallery_n: int = -1             # -1 for all or int

    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 4 
    
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 

    # Dataset
    query_mode: str = 'D2S'
    # query_mode: str = 'S2D'

    # Checkpoint to start from
    # checkpoint_start = 'pretrained/university/convnext_base.fb_in22k_ft_in1k_384/weights_e1_0.9515.pth'
    # checkpoint_start = 'work_dir/denseuav/convnext_base.fb_in22k_ft_in1k_384/0630155817/weights_end.pth'
    # checkpoint_start = 'work_dir/sues/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/0810002619/weights_end.pth'
    # checkpoint_start = 'work_dir/sues/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/0809045532/weights_end.pth'
    checkpoint_start = 'work_dir/gta/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/0912030534/weights_end.pth'

    data_root: str = "/home/xmuairmud/data/GTA-UAV-data/randcam2_5area"

    train_pairs_meta_file = 'offset13_cross-area-drone2sate-train.json'
    test_pairs_meta_file = 'offset13_cross-area-drone2sate-test.json'
    sate_img_dir = 'satellite_overlap/offset_13'


#-----------------------------------------------------------------------------#
# Config                                                                      #
#-----------------------------------------------------------------------------#

config = Configuration()


if __name__ == '__main__':

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
    
    print("\nModel: {}".format(config.model))


    model = DesModel(config.model,
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
    if config.query_mode == 'D2S':
        query_dataset_test = GTADatasetEval(data_root=config.data_root,
                                            pairs_meta_file=config.test_pairs_meta_file,
                                            view="drone",
                                            transforms=val_transforms,
                                            mode='pos',
                                            query_mode=config.query_mode,
                                            )
        gallery_dataset_test = GTADatasetEval(data_root=config.data_root,
                                            pairs_meta_file=config.test_pairs_meta_file,
                                            view="sate",
                                            transforms=val_transforms,
                                            sate_img_dir=config.sate_img_dir,
                                            mode='pos',
                                            query_mode=config.query_mode,
                                            )
        pairs_dict = query_dataset_test.pairs_drone2sate_dict
    elif config.query_mode == 'S2D':
        gallery_dataset_test = GTADatasetEval(data_root=config.data_root,
                                            pairs_meta_file=config.test_pairs_meta_file,
                                            view="drone",
                                            transforms=val_transforms,
                                            mode='pos',
                                            query_mode=config.query_mode,
                                            )
        pairs_dict = gallery_dataset_test.pairs_sate2drone_dict
        query_dataset_test = GTADatasetEval(data_root=config.data_root,
                                            pairs_meta_file=config.test_pairs_meta_file,
                                            view="sate",
                                            transforms=val_transforms,
                                            query_mode=config.query_mode,
                                            pairs_sate2drone_dict=pairs_dict,
                                            sate_img_dir=config.sate_img_dir,
                                            mode='pos',
                                        )
    query_img_list = query_dataset_test.images_name
    query_loc_xy_list = query_dataset_test.images_loc_xy

    gallery_loc_xy_list = gallery_dataset_test.images_loc_xy
    gallery_img_list = gallery_dataset_test.images_name

    query_dataloader_test = DataLoader(query_dataset_test,
                                    batch_size=config.batch_size,
                                    num_workers=config.num_workers,
                                    shuffle=False,
                                    pin_memory=True)
    gallery_dataloader_test = DataLoader(gallery_dataset_test,
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    print("Query Images Test:", len(query_dataset_test))
    print("Gallery Images Test:", len(gallery_dataset_test))
    
    print("\n{}[{}]{}".format(30*"-", "GTA-VisLoc", 30*"-"))  

    r1_test = evaluate(config=config,
                           model=model,
                           query_loader=query_dataloader_test,
                           gallery_loader=gallery_dataloader_test, 
                           query_list=query_img_list,
                           gallery_list=gallery_img_list,
                           pairs_dict=pairs_dict,
                           ranks_list=[1, 5, 10],
                           query_loc_xy_list=query_loc_xy_list,
                           gallery_loc_xy_list=gallery_loc_xy_list,
                           step_size=1000,
                           cleanup=True,
                           plot_acc_threshold=True,
                           top10_log=False)
 