import os
import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader

from game4loc.dataset.gta_mm import GTAMMDatasetEval, get_transforms
from game4loc.evaluate.gta_mm import evaluate
from game4loc.models.model_mm import DesModelWithMM


@dataclass
class Configuration:

    # Model
    # model: str = 'convnext_base.fb_in22k_ft_in1k_384'
    model: str = 'vit_base_patch16_rope_reg1_gap_256.sbb_in1k'
    
    # Override model image size
    img_size: int = 384
    
    # Evaluation
    batch_size: int = 32
    verbose: bool = True
    gpu_ids: tuple = (0,)
    normalize_features: bool = True
    eval_gallery_n: int = -1             # -1 for all or int

    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 4 
    
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 

    # Dataset
    query_mode: str = 'D2S'
    # query_mode: str = 'S2D'
    test_mode: str = 'pos'

    with_depth = False
    with_text = False
    with_pc = True

    # Checkpoint to start from
    # checkpoint_start = 'pretrained/university/convnext_base.fb_in22k_ft_in1k_384/weights_e1_0.9515.pth'
    # checkpoint_start = 'work_dir/denseuav/convnext_base.fb_in22k_ft_in1k_384/0630155817/weights_end.pth'
    # checkpoint_start = 'work_dir/sues/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/0810002619/weights_end.pth'
    # checkpoint_start = 'work_dir/sues/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/0809045532/weights_end.pth'
    checkpoint_start = 'pretrained/gta_12/with_pc.pth'

    data_root: str = "/home/xmuairmud/data/GTA-UAV-data/GTA-UAV-Lidar/GTA-UAV-Lidar-LR"

    train_pairs_meta_file = 'cross-area-drone2sate-train-textqw2b-12.json'
    test_pairs_meta_file = 'cross-area-drone2sate-test-textqw2b-12.json'
    sate_img_dir = 'satellite'

    ####### Cross-area
    dis_threshold_list = [10*(i+1) for i in range(50)]

    ####### Same-area
    # dis_threshold_list = [4*(i+1) for i in range(50)]


#-----------------------------------------------------------------------------#
# Config                                                                      #
#-----------------------------------------------------------------------------#

config = Configuration()


if __name__ == '__main__':

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
    
    print("\nModel: {}".format(config.model))

    model = DesModelWithMM(config.model,
                          pretrained=True,
                          img_size=config.img_size,
                          with_depth=config.with_depth,
                          with_text=config.with_text,
                          with_pc=config.with_pc)
                          
    data_config = model.get_config()
    print(data_config)
    mean = list(data_config["mean"])
    std = list(data_config["std"])
    img_size = (config.img_size, config.img_size)
    

    # load pretrained Checkpoint    
    if config.checkpoint_start is not None:  
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)  
        model.load_state_dict(model_state_dict, strict=True)     

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
    val_sat_transforms, val_drone_rgb_transforms, val_drone_depth_transforms, train_sat_transforms, \
        train_drone_geo_transforms, train_drone_rgb_transforms, train_drone_depth_transforms \
         = get_transforms(img_size, mean=mean, std=std, eval_robust=False)

    query_view = 'drone'
    gallery_view = 'sate'

    # Test query
    query_dataset_test = GTAMMDatasetEval(data_root=config.data_root,
                                        pairs_meta_file=config.test_pairs_meta_file,
                                        view=query_view,
                                        transforms_rgb=val_drone_rgb_transforms,
                                        transforms_depth=val_drone_depth_transforms,
                                        mode=config.test_mode,
                                        sate_img_dir=config.sate_img_dir,
                                        query_mode=config.query_mode,
                                        )
    query_img_list = query_dataset_test.drone_img_names
    query_loc_xy_list = query_dataset_test.drone_loc_xys
    pairs_drone2sate_dict = query_dataset_test.pairs_drone2sate_dict
    
    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    # Test gallery
    gallery_dataset_test = GTAMMDatasetEval(data_root=config.data_root,
                                          pairs_meta_file=config.test_pairs_meta_file,
                                          view=gallery_view,
                                          transforms_rgb=val_sat_transforms,
                                          mode=config.test_mode,
                                          sate_img_dir=config.sate_img_dir,
                                          query_mode=config.query_mode,
                                         )
    gallery_loc_xy_list = gallery_dataset_test.satellite_loc_xys
    gallery_img_list = gallery_dataset_test.satellite_img_names
    
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
                           pairs_dict=pairs_drone2sate_dict,
                           ranks_list=[1, 5, 10],
                           query_loc_xy_list=query_loc_xy_list,
                           gallery_loc_xy_list=gallery_loc_xy_list,
                           step_size=1000,
                           dis_threshold_list=config.dis_threshold_list,
                           cleanup=True,
                           plot_acc_threshold=True,
                           top10_log=True)


