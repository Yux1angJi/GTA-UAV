import os
import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader

from game4loc.dataset.visloc_rgbd import VisLocRGBDDatasetEval, get_transforms
from game4loc.evaluate.visloc_rgbd import evaluate
from game4loc.models.model_rgbd import DesModelWithRGBD


@dataclass
class Configuration:

    # Model
    # model: str = 'convnext_base.fb_in22k_ft_in1k_384'
    model: str = 'vit_base_patch16_rope_reg1_gap_256.sbb_in1k'
    
    # Override model image size
    img_size: int = 384
    
    # Evaluation
    batch_size: int = 128
    batch_size_eval: int = 128

    verbose: bool = True
    gpu_ids: tuple = (0,1)
    normalize_features: bool = True
    eval_gallery_n: int = -1             # -1 for all or int

    test_mode: str = 'pos'

    # Dataset
    dataset: str = 'VisLoc-D2S'           # 'U1652-D2S' | 'U1652-S2D'

    # Checkpoint to start from
    # checkpoint_start = 'work_dir/university/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/162436/weights_end.pth'
    # checkpoint_start = 'work_dir/sues/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/0723160833/weights_end.pth'
    # checkpoint_start = 'work_dir/denseuav/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/0723164458/weights_end.pth'
    # checkpoint_start = 'work_dir/gta/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/0722110449/weights_end.pth'
    
    # checkpoint_start = 'work_dir/visloc/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/0723224548/weights_end.pth' ## GTA-UAV
    # checkpoint_start = 'work_dir/visloc/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/0724003322/weights_end.pth' ## DenseUAV
    # checkpoint_start = 'work_dir/visloc/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/0724022105/weights_end.pth' ## SUES-200
    # checkpoint_start = 'work_dir/visloc/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/0723205823/weights_end.pth' ## ImageNet
    # checkpoint_start = 'work_dir/visloc/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/0724145818/weights_end.pth' ## University

    checkpoint_start = './pretrained/visloc/cross_area/licogeo.pth'

    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 4 
    
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 

    query_mode = 'D2S'

    data_root = '/root/lanyun-tmp/UAV_VisLoc_dataset_Lidar'

    train_pairs_meta_file = 'cross-area-drone2sate-train-z31.json'
    test_pairs_meta_file = 'cross-area-drone2sate-test-z31.json'
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

    model = DesModelWithRGBD(config.model,
                          pretrained=True,
                          img_size=config.img_size)
                          
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
         = get_transforms(img_size, mean=mean, std=std)

    query_view = 'drone'
    gallery_view = 'sate'

    # Test query
    query_dataset_test = VisLocRGBDDatasetEval(data_root=config.data_root,
                                        pairs_meta_file=config.test_pairs_meta_file,
                                        view=query_view,
                                        transforms_rgb=val_drone_rgb_transforms,
                                        transforms_depth=val_drone_depth_transforms,
                                        mode=config.test_mode,
                                        sate_img_dir=config.sate_img_dir,
                                        query_mode=config.query_mode,
                                        )
    query_img_list = query_dataset_test.images_name
    pairs_drone2sate_dict = query_dataset_test.pairs_drone2sate_dict
    query_loc_xy_list = query_dataset_test.images_loc_xy
    
    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)

    # Test gallery
    gallery_dataset_test = VisLocRGBDDatasetEval(data_root=config.data_root,
                                          pairs_meta_file=config.test_pairs_meta_file,
                                          view=gallery_view,
                                          transforms_rgb=val_sat_transforms,
                                          mode=config.test_mode,
                                          sate_img_dir=config.sate_img_dir,
                                          query_mode=config.query_mode,
                                         )
    gallery_img_list = gallery_dataset_test.images_name
    gallery_loc_xy_list = gallery_dataset_test.images_loc_xy
    
    gallery_dataloader_test = DataLoader(gallery_dataset_test,
                                       batch_size=config.batch_size_eval,
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
                           query_loc_xy_list=query_loc_xy_list,
                           gallery_loc_xy_list=gallery_loc_xy_list,
                           pairs_dict=pairs_drone2sate_dict,
                           ranks_list=[1, 5, 10],
                           step_size=1000,
                           cleanup=True,
                           dis_threshold_list=config.dis_threshold_list,
                           plot_acc_threshold=True,
                           top10_log=True)
 
