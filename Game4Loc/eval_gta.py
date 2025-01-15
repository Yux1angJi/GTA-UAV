import os
import sys
import torch
import argparse
from dataclasses import dataclass
from torch.utils.data import DataLoader

from game4loc.dataset.gta import GTADatasetEval, get_transforms
from game4loc.evaluate.gta import evaluate
from game4loc.models.model import DesModel


def parse_tuple(s):
    try:
        return tuple(map(int, s.split(',')))
    except ValueError:
        raise argparse.ArgumentTypeError("Tuple must be integers separated by commas")


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
    gpu_ids: tuple = (0)
    normalize_features: bool = True

    # With Fine Matching
    with_match: bool = False

    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 4 
    
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 

    # Dataset
    query_mode: str = 'D2S'
    # query_mode: str = 'S2D'

    # Checkpoint to start from
    # checkpoint_start = '/home/xmuairmud/jyx/GTA-UAV/Game4Loc/pretrained/gta/same_area/selavpr.pth'
    checkpoint_start = 'pretrained/gta/cross_area/game4loc.pth'

    # data_root: str = "/home/xmuairmud/data/GTA-UAV-data/GTA-UAV-Lidar/GTA-UAV-Lidar"
    data_root: str = "/home/xmuairmud/data/GTA-UAV-data/GTA-UAV-official/GTA-UAV-LR-hf"

    train_pairs_meta_file = 'cross-area-drone2sate-train.json'
    test_pairs_meta_file = 'cross-area-drone2sate-test.json'
    sate_img_dir = 'satellite'


def eval_script(config):

    if config.log_to_file:
        f = open(config.log_path, 'w')
        sys.stdout = f

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
    
    print("\nModel: {}".format(config.model))


    model = DesModel(config.model,
                    pretrained=True,
                    img_size=config.img_size,
                    share_weights=config.share_weights)
                          
    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
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
    query_center_loc_xy_list = query_dataset_test.images_center_loc_xy

    gallery_center_loc_xy_list = gallery_dataset_test.images_center_loc_xy
    gallery_topleft_loc_xy_list = gallery_dataset_test.images_topleft_loc_xy
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

    # For Test Log (distance threshold) 
    dis_threshold_list = None
    if 'cross' in config.test_pairs_meta_file:
        ####### Cross-area for total 500m/10m
        print("cross-area eval")
        dis_threshold_list = [10*(i+1) for i in range(50)]
    else:
        ####### Same-area for total 200m/4m
        print("same-area eval")
        dis_threshold_list = [4*(i+1) for i in range(50)]
    
    print("\n{}[{}]{}".format(30*"-", "Evaluating GTA-UAV", 30*"-"))  

    r1_test = evaluate(config=config,
                           model=model,
                           query_loader=query_dataloader_test,
                           gallery_loader=gallery_dataloader_test, 
                           query_list=query_img_list,
                           gallery_list=gallery_img_list,
                           pairs_dict=pairs_dict,
                           ranks_list=[1, 5, 10],
                           query_center_loc_xy_list=query_center_loc_xy_list,
                           gallery_center_loc_xy_list=gallery_center_loc_xy_list,
                           gallery_topleft_loc_xy_list=gallery_topleft_loc_xy_list,
                           step_size=1000,
                           dis_threshold_list=dis_threshold_list,
                           cleanup=True,
                           plot_acc_threshold=False,
                           top10_log=False,
                           with_match=config.with_match)

    if config.log_to_file:
        f.close()
        sys.stdout = sys.__stdout__  
 


def parse_args():
    parser = argparse.ArgumentParser(description="Training script for gta.")

    parser.add_argument('--log_to_file', action='store_true', help='Log saving to file')

    parser.add_argument('--log_path', type=str, default=None, help='Log file path')

    parser.add_argument('--data_root', type=str, default='./data/GTA-UAV-data', help='Data root')
   
    parser.add_argument('--test_pairs_meta_file', type=str, default='cross-area-drone2sate-test.json', help='Test metafile path')

    parser.add_argument('--model', type=str, default='vit_base_patch16_rope_reg1_gap_256.sbb_in1k', help='Model architecture')

    parser.add_argument('--no_share_weights', action='store_true', help='Model not sharing wieghts')

    parser.add_argument('--with_match', action='store_true', help='Test with post-process image matching (GIM, etc)')

    parser.add_argument('--gpu_ids', type=parse_tuple, default=(0,1), help='GPU ID')

    parser.add_argument('--batch_size', type=int, default=40, help='Batch size')

    parser.add_argument('--checkpoint_start', type=str, default=None, help='Training from checkpoint')

    parser.add_argument('--test_mode', type=str, default='pos', help='Test with positive pairs')

    parser.add_argument('--query_mode', type=str, default='D2S', help='Retrieval with drone to satellite')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    config = Configuration()
    config.data_root = args.data_root
    config.test_pairs_meta_file = args.test_pairs_meta_file
    config.log_to_file = args.log_to_file
    config.log_path = args.log_path
    config.batch_size = args.batch_size
    config.gpu_ids = args.gpu_ids
    config.checkpoint_start = args.checkpoint_start
    config.model = args.model
    config.share_weights = not(args.no_share_weights)
    config.test_mode = args.test_mode
    config.query_mode = args.query_mode
    config.with_match = args.with_match

    eval_script(config)