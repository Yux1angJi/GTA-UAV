import os
import time
import math
import shutil
import sys
import torch
import argparse
from dataclasses import dataclass
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup

from game4loc.dataset.gta_mm import GTAMMDatasetEvalUni, GTAMMDatasetTrain, get_transforms
from game4loc.utils import setup_system, Logger
from game4loc.trainer.trainer_mm_uni import train_mm_with_weight
from game4loc.evaluate.gta_mm_uni import evaluate
from game4loc.loss import InfoNCE, MMWeightedInfoNCE, WeightedInfoNCE, GroupInfoNCE, TripletLoss
from game4loc.models.model_mm import DesModelWithMM


def parse_tuple(s):
    try:
        return tuple(map(int, s.split(',')))
    except ValueError:
        raise argparse.ArgumentTypeError("Tuple must be integers separated by commas")


@dataclass
class Configuration:
    
    # Model
    model: str = 'convnext_base.fb_in22k_ft_in1k_384'
    model_hub: str = 'timm'
    
    # Override model image size
    img_size: int = 384
 
    # Please Ignore
    freeze_layers: bool = False
    frozen_stages = [0,0,0,0]

    # Training with sharing weights
    share_weights: bool = True
    
    # Training with weighted-InfoNCE
    with_weight: bool = True

    # Please Ignore
    train_in_group: bool = True
    group_len = 2
    # Please Ignore
    loss_type = ["whole_slice", "part_slice"]

    # Please Ignore
    train_with_mix_data: bool = False
    # Please Ignore
    train_with_recon: bool = False
    recon_weight: float = 0.1
    
    
    # Training 
    mixed_precision: bool = True
    custom_sampling: bool = True         # use custom sampling instead of random
    seed = 1
    epochs: int = 10
    batch_size: int = 40                # keep in mind real_batch_size = 2 * batch_size
    verbose: bool = False
    gpu_ids: tuple = (0,1)           # GPU ids for training

    # Training with sparse data
    train_ratio: float = 1.0

    # Eval
    batch_size_eval: int = 128
    eval_every_n_epoch: int = 1          # eval every n Epoch
    normalize_features: bool = True
    eval_gallery_n: int = -1             # -1 for all or int

    # Optimizer 
    clip_grad = 100.                     # None | float
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = False     # Gradient Checkpointing
    
    # Loss
    label_smoothing: float = 0.1
    k: float = 3

    # Multi-modal setting
    with_text: bool = False
    with_depth: bool = True
    with_pc: bool = False
    
    # Learning Rate
    lr: float = 0.001                    # 1 * 10^-4 for ViT | 1 * 10^-1 for CNN
    scheduler: str = "cosine"            # "polynomial" | "cosine" | "constant" | None
    warmup_epochs: int = 0.1
    lr_end: float = 0.0001               #  only for "polynomial"

    # Augment Images
    prob_flip: float = 0.5               # flipping the sat image and drone image simultaneously
    
    # Savepath for model checkpoints
    model_path: str = "./work_dir/gta"

    query_mode: str = "D2S"               # Retrieval in Drone to Satellite

    train_mode: str = "pos_semipos"       # Train with positive + semi-positive pairs
    test_mode: str = "pos"                # Test with positive pairs

    # Eval before training
    zero_shot: bool = True
    
    # Checkpoint to start from
    checkpoint_start = None

    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 4 
    
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 
    
    # for better performance
    cudnn_benchmark: bool = True
    
    # make cudnn deterministic
    cudnn_deterministic: bool = False

    data_root: str = "./data/GTA-UAV-data"

    train_pairs_meta_file = 'cross-area-drone2sate-train.json'
    test_pairs_meta_file = 'cross-area-drone2sate-test.json'
    sate_img_dir = 'satellite'


#-----------------------------------------------------------------------------#
# Train Config                                                                #
#-----------------------------------------------------------------------------#

def train_script(config):

    if config.log_to_file:
        f = open(config.log_path, 'w')
        sys.stdout = f

    save_time = "{}".format(time.strftime("%m%d%H%M%S"))
    model_path = "{}/{}/{}".format(config.model_path,
                                       config.model,
                                       save_time)

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    shutil.copyfile(os.path.basename(__file__), "{}/train.py".format(model_path))

    # Redirect print to both console and log file
    sys.stdout = Logger(os.path.join(model_path, 'log.txt'))

    setup_system(seed=config.seed,
                 cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)
    
    print("training save in path: {}".format(model_path))

    print("training start from", config.checkpoint_start)

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
    print("\nModel: {}".format(config.model))
    print(f"Train with depth? {config.with_depth}")
    print(f"Train with text? {config.with_text}")
    print(f"Train with point cloud? {config.with_pc}")

    model = DesModelWithMM(
                    uni_modal=True,
                    model_name=config.model, 
                    pretrained=True,
                    img_size=config.img_size,
                    share_weights=config.share_weights,
                    with_text=config.with_text,
                    with_depth=config.with_depth,
                    with_pc=config.with_pc,
                )
                        
    data_config = model.get_config()
    print(data_config)
    mean = list(data_config["mean"])
    std = list(data_config["std"])
    img_size = (config.img_size, config.img_size)
    
    # Activate gradient checkpointing
    if config.grad_checkpointing:
        model.set_grad_checkpointing(True)
    
    # Load pretrained Checkpoint    
    if config.checkpoint_start is not None:  
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)
        model_state_dict_new = {}
        for k, v in model_state_dict.items():
            model_state_dict_new[k.replace('model.', '')] = v
        model.img_model.load_state_dict(model_state_dict_new, strict=False)
        for param in model.img_model.parameters():
            param.requires_grad = False 

    print("Freeze model:", config.freeze_layers, config.frozen_stages)
    if config.freeze_layers:
        model.freeze_layers(config.frozen_stages)

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

    print("Use custom sampling: {}".format(config.custom_sampling))


    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#

    # Transforms
    val_sat_transforms, val_drone_rgb_transforms, val_drone_depth_transforms, train_sat_transforms, \
        train_drone_geo_transforms, train_drone_rgb_transforms, train_drone_depth_transforms \
         = get_transforms(img_size, mean=mean, std=std)
                                                                                                              
    # Train
    train_dataset = GTAMMDatasetTrain(data_root=config.data_root,
                                    pairs_meta_file=config.train_pairs_meta_file,
                                    transforms_drone_geo=train_drone_geo_transforms,
                                    transforms_drone_img=train_drone_rgb_transforms,
                                    transforms_drone_depth=train_drone_depth_transforms,
                                    transforms_satellite=train_sat_transforms,
                                    prob_flip=config.prob_flip,
                                    shuffle_batch_size=config.batch_size,
                                    mode=config.train_mode,
                                    train_ratio=config.train_ratio,
                                    )
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  shuffle=not config.custom_sampling,
                                  pin_memory=True)
    
    # Test query
    if config.query_mode == 'DImg2SImg':
        query_view = 'drone_img'
        gallery_view = 'sate_img'
    elif config.query_mode == 'DPC2SImg':
        query_view = 'drone_pc'
        gallery_view = 'sate_img'
    elif config.query_mode == 'DText2SImg':
        query_view = 'drone_desc'
        gallery_view = 'sate_img'
    elif config.query_mode == 'DDepth2SImg':
        query_view = 'drone_depth'
        gallery_view = 'sate_img'
    elif config.query_mode == 'DDepth2DImg':
        query_view = 'drone_depth'
        gallery_view = 'drone_img'
    elif config.query_mode == 'DImg2DDepth':
        query_view = 'drone_img'
        gallery_view = 'drone_depth'
    elif config.query_mode == 'DText2DImg':
        query_view = 'drone_desc'
        gallery_view = 'drone_img'
    elif config.query_mode == 'DPC2DImg':
        query_view = 'drone_pc'
        gallery_view = 'drone_img'
    elif config.query_mode == 'SText2SImg':
        query_view = 'sate_desc'
        gallery_view = 'sate_img'
    query_dataset_test = GTAMMDatasetEvalUni(data_root=config.data_root,
                                        pairs_meta_file=config.test_pairs_meta_file,
                                        view=query_view,
                                        transforms=val_drone_depth_transforms,
                                        mode=config.test_mode,
                                        sate_img_dir=config.sate_img_dir,
                                        query_mode=config.query_mode,
                                        )
    if 'DImg2' in config.query_mode or 'DText2' in config.query_mode or 'DPC2' in config.query_mode or 'DDepth2' in config.query_mode:
        query_img_list = query_dataset_test.drone_img_names
        query_loc_xy_list = query_dataset_test.drone_loc_xys
        pairs_drone2sate_dict = query_dataset_test.pairs_drone2sate_dict
    else:
        query_img_list = query_dataset_test.satellite_img_names
        query_loc_xy_list = query_dataset_test.satellite_loc_xys
        pairs_drone2sate_dict = None
    
    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    # Test gallery
    gallery_dataset_test = GTAMMDatasetEvalUni(data_root=config.data_root,
                                          pairs_meta_file=config.test_pairs_meta_file,
                                          view=gallery_view,
                                          transforms=val_sat_transforms,
                                          mode=config.test_mode,
                                          sate_img_dir=config.sate_img_dir,
                                          query_mode=config.query_mode,
                                         )

    if '2SImg' in config.query_mode:
        gallery_img_list = gallery_dataset_test.satellite_img_names
        gallery_loc_xy_list = gallery_dataset_test.satellite_loc_xys
    elif '2DImg' in config.query_mode or '2DDepth' in config.query_mode:
        gallery_img_list = gallery_dataset_test.drone_img_names
        gallery_loc_xy_list = gallery_dataset_test.drone_loc_xys
    
    gallery_dataloader_test = DataLoader(gallery_dataset_test,
                                       batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    print("Query Images Test:", len(query_dataset_test))
    print("Gallery Images Test:", len(gallery_dataset_test))
    
    #-----------------------------------------------------------------------------#
    # Loss                                                                        #
    #-----------------------------------------------------------------------------#
    print("Train with weight?", config.with_weight, "k=", config.k)
    
    loss_function_normal = MMWeightedInfoNCE(
        device=config.device,
        label_smoothing=config.label_smoothing,
        k=config.k,
        dimg2simg=(config.query_mode=="DImg2SImg"),
        dpc2simg=(config.query_mode=="DPC2SImg"),
        dimg2dpc=(config.query_mode=="DPC2DImg"),
        ddepth2dimg=(config.query_mode=="DDepth2DImg"),
        ddepth2simg=(config.query_mode=="DDepth2SImg"),
        ddesc2simg=(config.query_mode=="DText2SImg"),
        ddesc2dimg=(config.query_mode=="DText2DImg"),
        dpc2dimg=(config.query_mode=="DPC2DImg"),
        sdesc2simg=(config.query_mode=='SText2SImg'),
        with_depth=config.with_depth,
        with_pc=config.with_pc,
        with_text=config.with_text,
    )
    ## For TripletLoss
    # loss_function_normal = TripletLoss(device=config.device)

    if config.mixed_precision:
        scaler = GradScaler(init_scale=2.**10)
    else:
        scaler = None
        
    #-----------------------------------------------------------------------------#
    # optimizer                                                                   #
    #-----------------------------------------------------------------------------#

    if config.decay_exclue_bias:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=config.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)


    #-----------------------------------------------------------------------------#
    # Scheduler                                                                   #
    #-----------------------------------------------------------------------------#

    train_steps = len(train_dataloader) * config.epochs
    warmup_steps = len(train_dataloader) * config.warmup_epochs
       
    if config.scheduler == "polynomial":
        print("\nScheduler: polynomial - max LR: {} - end LR: {}".format(config.lr, config.lr_end))  
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                              num_training_steps=train_steps,
                                                              lr_end = config.lr_end,
                                                              power=1.5,
                                                              num_warmup_steps=warmup_steps)
        
    elif config.scheduler == "cosine":
        print("\nScheduler: cosine - max LR: {}".format(config.lr))   
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_training_steps=train_steps,
                                                    num_warmup_steps=warmup_steps)
        
    elif config.scheduler == "constant":
        print("\nScheduler: constant - max LR: {}".format(config.lr))   
        scheduler =  get_constant_schedule_with_warmup(optimizer,
                                                       num_warmup_steps=warmup_steps)
           
    else:
        scheduler = None
        
    print("Warmup Epochs: {} - Warmup Steps: {}".format(str(config.warmup_epochs).ljust(2), warmup_steps))
    print("Train Epochs:  {} - Train Steps:  {}".format(config.epochs, train_steps))
        
    #-----------------------------------------------------------------------------#
    # Query Mode                                                                  #
    #-----------------------------------------------------------------------------#

    self_dict = {}
    if pairs_drone2sate_dict is not None:
        for k in pairs_drone2sate_dict.keys():
            self_dict[k] = [k]
    else:
        for sate_img in query_img_list:
            self_dict[sate_img] = [sate_img] 

    if config.query_mode == 'DImg2SImg':
        query_feature = 'drone_img_features'
        gallery_feature = 'satellite_img_features'
        pairs_dict = pairs_drone2sate_dict
    elif config.query_mode == 'DPC2SImg':
        query_feature = 'drone_pc_features'
        gallery_feature = 'satellite_img_features'
        pairs_dict = pairs_drone2sate_dict
    elif config.query_mode == 'DDepth2SImg':
        query_feature = 'drone_depth_features'
        gallery_feature = 'satellite_img_features'
        pairs_dict = pairs_drone2sate_dict
    elif config.query_mode == 'DText2SImg':
        query_feature = 'drone_desc_features'
        gallery_feature = 'satellite_img_features'
        pairs_dict = pairs_drone2sate_dict
    elif config.query_mode == 'DDepth2DImg':
        query_feature = 'drone_depth_features'
        gallery_feature = 'drone_img_features'
        pairs_dict = self_dict
    elif config.query_mode == 'DImg2DDepth':
        query_feature = 'drone_img_features'
        gallery_feature = 'drone_depth_features'
        pairs_dict = self_dict
    elif config.query_mode == 'DText2DImg':
        query_feature = 'drone_desc_features'
        gallery_feature = 'drone_img_features'
        pairs_dict = self_dict
    elif config.query_mode == 'DPC2DImg':
        query_feature = 'drone_pc_features'
        gallery_feature = 'drone_img_features'
        pairs_dict = self_dict
    elif config.query_mode == 'SText2SImg':
        query_feature = 'satellite_desc_features'
        gallery_feature = 'satellite_img_features'
        pairs_dict = self_dict
        
    #-----------------------------------------------------------------------------#
    # Zero Shot                                                                   #
    #-----------------------------------------------------------------------------#

    if config.zero_shot:
        print("\n{}[{}]{}".format(30*"-", "Zero Shot", 30*"-"))  

        r1_test = evaluate(config=config,
                           model=model,
                           query_loader=query_dataloader_test,
                           gallery_loader=gallery_dataloader_test, 
                           query_list=query_img_list,
                           query_feature=query_feature,
                           gallery_list=gallery_img_list,
                           gallery_feature=gallery_feature,
                           pairs_dict=pairs_dict,
                           ranks_list=[1, 5, 10],
                           query_loc_xy_list=query_loc_xy_list,
                           gallery_loc_xy_list=gallery_loc_xy_list,
                           step_size=1000,
                           cleanup=True)
           
            
    #-----------------------------------------------------------------------------#
    # Train                                                                       #
    #-----------------------------------------------------------------------------#
    start_epoch = 0   
    best_score = 0
    

    for epoch in range(1, config.epochs+1):
        
        print("\n{}[Epoch: {}]{}".format(30*"-", epoch, 30*"-"))

        if config.custom_sampling:
            train_dataloader.dataset.shuffle()
        
        train_loss = train_mm_with_weight(config,
                           model,
                           dataloader=train_dataloader,
                           loss_function=loss_function_normal,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           scaler=scaler,
                           with_weight=config.with_weight)
        
        print("Epoch: {}, Train Loss = {:.3f}, Lr = {:.6f}".format(epoch,
                                                                   train_loss,
                                                                   optimizer.param_groups[0]['lr']))
        
        # evaluate
        if (epoch % config.eval_every_n_epoch == 0 and epoch != 0) or epoch == config.epochs:
        
            print("\n{}[{}]{}".format(30*"-", "Evaluate", 30*"-"))
        
            r1_test = evaluate(config=config,
                                model=model,
                                query_loader=query_dataloader_test,
                                gallery_loader=gallery_dataloader_test, 
                                query_list=query_img_list,
                                gallery_list=gallery_img_list,
                                query_feature=query_feature,
                                gallery_feature=gallery_feature,
                                pairs_dict=pairs_dict,
                                ranks_list=[1, 5, 10],
                                query_loc_xy_list=query_loc_xy_list,
                                gallery_loc_xy_list=gallery_loc_xy_list,
                                step_size=1000,
                                cleanup=True)
                
            if r1_test > best_score:

                best_score = r1_test

                if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
                    torch.save(model.module.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))
                else:
                    torch.save(model.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))
                
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        torch.save(model.module.state_dict(), '{}/weights_end.pth'.format(model_path))
    else:
        torch.save(model.state_dict(), '{}/weights_end.pth'.format(model_path))  

    if config.log_to_file:
        f.close()
        sys.stdout = sys.__stdout__          


def parse_args():
    parser = argparse.ArgumentParser(description="Training script for gta.")

    parser.add_argument('--log_to_file', action='store_true', help='Log saving to file')

    parser.add_argument('--log_path', type=str, default=None, help='Log file path')

    parser.add_argument('--data_root', type=str, default='./data/GTA-UAV-data', help='Data root')

    parser.add_argument('--train_pairs_meta_file', type=str, default='cross-area-drone2sate-train.json', help='Training metafile path')
   
    parser.add_argument('--test_pairs_meta_file', type=str, default='cross-area-drone2sate-test.json', help='Test metafile path')

    parser.add_argument('--model', type=str, default='vit_base_patch16_rope_reg1_gap_256.sbb_in1k', help='Model architecture')

    parser.add_argument('--no_share_weights', action='store_true', help='Train without sharing wieghts')

    parser.add_argument('--freeze_layers', action='store_true', help='Freeze layers for training')

    parser.add_argument('--frozen_stages', type=int, nargs='+', default=[0,0,0,0], help='Frozen stages for training')

    parser.add_argument('--epochs', type=int, default=5, help='Epochs')

    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')

    parser.add_argument('--gpu_ids', type=parse_tuple, default=(0,1), help='GPU ID')

    parser.add_argument('--batch_size', type=int, default=40, help='Batch size')

    parser.add_argument('--checkpoint_start', type=str, default=None, help='Training from checkpoint')

    parser.add_argument('--train_mode', type=str, default='pos_semipos', help='Train with positive or positive+semi-positive pairs')

    parser.add_argument('--test_mode', type=str, default='pos', help='Test with positive pairs')

    parser.add_argument('--query_mode', type=str, default='DImg2SImg', help='Retrieval with drone image to satellite image')

    parser.add_argument('--train_with_recon', action='store_true', help='Train with reconstruction')

    parser.add_argument('--recon_weight', type=float, default=0.1, help='Loss weight for reconstruction')

    parser.add_argument('--train_in_group', action='store_true', help='Train in group')
    
    parser.add_argument('--group_len', type=int, default=2, help='Group length')

    parser.add_argument('--train_with_mix_data', action='store_true', help='Train with mix data')

    parser.add_argument('--loss_type', type=str, nargs='+', default=['part_slice', 'whole_slice'], help='Loss type for group train')

    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing value for loss')

    parser.add_argument('--with_weight', action='store_true', help='Train with weight')

    parser.add_argument('--k', type=float, default=5, help='weighted k')

    parser.add_argument('--with_text', action='store_true', help='Train with text collaboration')

    parser.add_argument('--with_depth', action='store_true', help='Train with depth collaboration')

    parser.add_argument('--with_pc', action='store_true', help='Train with point cloud collaboration')
    
    parser.add_argument('--no_custom_sampling', action='store_true', help='Train without custom sampling')
    
    parser.add_argument('--train_ratio', type=float, default=1.0, help='Train on ratio of data')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    config = Configuration()
    config.data_root = args.data_root
    config.train_pairs_meta_file = args.train_pairs_meta_file
    config.test_pairs_meta_file = args.test_pairs_meta_file
    config.log_to_file = args.log_to_file
    config.log_path = args.log_path
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.train_in_group = args.train_in_group
    config.train_with_recon = args.train_with_recon
    config.recon_weight = args.recon_weight
    config.group_len = args.group_len
    config.train_with_mix_data = args.train_with_mix_data
    config.loss_type = args.loss_type
    config.gpu_ids = args.gpu_ids
    config.label_smoothing = args.label_smoothing
    config.with_weight = args.with_weight
    config.k = args.k
    config.checkpoint_start = args.checkpoint_start
    config.model = args.model
    config.lr = args.lr
    config.share_weights = not(args.no_share_weights)
    config.custom_sampling = not(args.no_custom_sampling)
    config.freeze_layers = args.freeze_layers
    config.frozen_stages = args.frozen_stages
    config.train_mode = args.train_mode
    config.test_mode = args.test_mode
    config.query_mode = args.query_mode
    config.train_ratio = args.train_ratio
    config.with_text = args.with_text
    config.with_depth = args.with_depth
    config.with_pc = args.with_pc

    train_script(config)