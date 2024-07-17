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

from sample4geo.dataset.gta import GTADatasetEval, GTADatasetTrain, get_transforms
from sample4geo.utils import setup_system, Logger
from sample4geo.trainer import train, train_with_weight
from sample4geo.evaluate.gta import evaluate
from sample4geo.loss import InfoNCE, ContrastiveLoss
from sample4geo.model import TimmModel


def parse_tuple(s):
    try:
        return tuple(map(int, s.split(',')))
    except ValueError:
        raise argparse.ArgumentTypeError("Tuple must be integers separated by commas")


@dataclass
class Configuration:
    
    # Model
    model: str = 'convnext_base.fb_in22k_ft_in1k_384'
    
    # Override model image size
    img_size: int = 384
 
    freeze_layers: bool = False

    frozen_stages = [0,0,0,0]

    share_weights: bool = True
    
    train_with_weight: bool = True

    train_in_group: bool = True
    group_len = 2

    loss_type = ["whole_slice", "part_slice"]

    train_with_mix_data: bool = False

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
    
    # Learning Rate
    lr: float = 0.001                    # 1 * 10^-4 for ViT | 1 * 10^-1 for CNN
    scheduler: str = "cosine"           # "polynomial" | "cosine" | "constant" | None
    warmup_epochs: int = 0.1
    lr_end: float = 0.0001               #  only for "polynomial"

    # Augment Images
    prob_flip: float = 0.5              # flipping the sat image and drone image simultaneously
    
    # Savepath for model checkpoints
    model_path: str = "./work_dir/gta"

    dataset: str= "GTA-D2S"
    
    # Eval before training
    zero_shot: bool = True
    
    # Checkpoint to start from
    checkpoint_start = None
    # checkpoint_start = "/home/xmuairmud/jyx/ExtenGeo/Sample4Geo/pretrained/university/convnext_base.fb_in22k_ft_in1k_384/weights_e1_0.9515.pth"


    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 4 
    
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 
    
    # for better performance
    cudnn_benchmark: bool = True
    
    # make cudnn deterministic
    cudnn_deterministic: bool = False

    train_pairs_meta_file = '/home/xmuairmud/data/GTA-UAV-data/randcam2_std0_stable/train_h23456_iou4/train_pair_meta.pkl'
    test_pairs_meta_file = '/home/xmuairmud/data/GTA-UAV-data/randcam2_std0_stable/train_h23456_iou4/test_pair_meta.pkl'
    sate_img_dir = '/home/xmuairmud/data/GTA-UAV-data/randcam2_std0_stable/satellite'


#-----------------------------------------------------------------------------#
# Train Config                                                                #
#-----------------------------------------------------------------------------#

def train_script(config):

    config.train_pairs_meta_file = f'/home/xmuairmud/data/GTA-UAV-data/{config.data_dir}/train_pair_meta.pkl'
    config.test_pairs_meta_file = f'/home/xmuairmud/data/GTA-UAV-data/{config.data_dir}/test_pair_meta.pkl'
    config.sate_img_dir = f'/home/xmuairmud/data/GTA-UAV-data/randcam2_std0_stable/satellite'

    f = open(config.log_path, 'w')
    if config.log_to_file:
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
    
    # Activate gradient checkpointing
    if config.grad_checkpointing:
        model.set_grad_checkpointing(True)
    
    # Load pretrained Checkpoint    
    if config.checkpoint_start is not None:  
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)  
        model.load_state_dict(model_state_dict, strict=False)     

    print("Freeze model layers:", config.freeze_layers, config.frozen_stages)
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


    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#

    # Transforms
    val_transforms, train_sat_transforms, train_drone_transforms = get_transforms(img_size, mean=mean, std=std)
                                                                                                                                 
    # Train
    train_dataset = GTADatasetTrain(pairs_meta_file=config.train_pairs_meta_file,
                                      transforms_query=train_sat_transforms,
                                      transforms_gallery=train_drone_transforms,
                                      prob_flip=config.prob_flip,
                                      shuffle_batch_size=config.batch_size,
                                      )
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  shuffle=not config.custom_sampling,
                                  pin_memory=True)
    
    # Test query
    query_dataset_test = GTADatasetEval(pairs_meta_file=config.test_pairs_meta_file,
                                        view="drone",
                                        transforms=val_transforms,
                                        )
    query_img_list = query_dataset_test.images
    query_loc_xy_list = query_dataset_test.images_loc_xy
    pairs_drone2sate_dict = query_dataset_test.pairs_drone2sate_dict
    
    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    # Test gallery
    gallery_dataset_test = GTADatasetEval(pairs_meta_file=config.test_pairs_meta_file,
                                               view="sate",
                                               transforms=val_transforms,
                                               sate_img_dir=config.sate_img_dir,
                                               )
    gallery_loc_xy_list = gallery_dataset_test.images_loc_xy
    gallery_img_list = gallery_dataset_test.images
    
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

    # loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    loss_function = ContrastiveLoss(
        device=config.device,
        label_smoothing=config.label_smoothing,
        k=config.k,
    )

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
    # Zero Shot                                                                   #
    #-----------------------------------------------------------------------------#
    if config.zero_shot:
        print("\n{}[{}]{}".format(30*"-", "Zero Shot", 30*"-"))  

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
                           cleanup=True)

    #-----------------------------------------------------------------------------#
    # Shuffle                                                                     #
    #-----------------------------------------------------------------------------#            
    if config.custom_sampling:
        train_dataloader.dataset.shuffle()
            
    #-----------------------------------------------------------------------------#
    # Train                                                                       #
    #-----------------------------------------------------------------------------#
    start_epoch = 0   
    best_score = 0
    

    for epoch in range(1, config.epochs+1):
        
        print("\n{}[Epoch: {}]{}".format(30*"-", epoch, 30*"-"))
        
        train_loss = train_with_weight(config,
                           model,
                           dataloader=train_dataloader,
                           loss_function=loss_function,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           scaler=scaler,
                           with_weight=config.train_with_weight)
        
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
                                pairs_dict=pairs_drone2sate_dict,
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
                

        if config.custom_sampling:
            train_dataloader.dataset.shuffle()
                
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        torch.save(model.module.state_dict(), '{}/weights_end.pth'.format(model_path))
    else:
        torch.save(model.state_dict(), '{}/weights_end.pth'.format(model_path))            


def parse_args():
    parser = argparse.ArgumentParser(description="Training script for gta.")

    parser.add_argument('--log_to_file', action='store_true', help='Log saving to file')

    parser.add_argument('--log_path', type=str, default=None, help='Log file path')

    parser.add_argument('--data_dir', type=str, default='randcam2_std0_stable/test', help='Data path')

    parser.add_argument('--no_share_weights', action='store_true', help='Train without sharing wieghts')

    parser.add_argument('--freeze_layers', action='store_true', help='Freeze layers for training')

    parser.add_argument('--frozen_stages', type=int, nargs='+', default=[0,0,0,0], help='Frozen stages for training')

    parser.add_argument('--epochs', type=int, default=5, help='Epochs')

    parser.add_argument('--gpu_ids', type=parse_tuple, default=(0,1), help='GPU ID')

    parser.add_argument('--batch_size', type=int, default=40, help='Batch size')

    parser.add_argument('--checkpoint_start', type=str, default=None, help='Training from checkpoint')

    parser.add_argument('--train_with_recon', action='store_true', help='Train with reconstruction')

    parser.add_argument('--recon_weight', type=float, default=0.1, help='Loss weight for reconstruction')

    parser.add_argument('--train_in_group', action='store_true', help='Train in group')
    
    parser.add_argument('--group_len', type=int, default=2, help='Group length')

    parser.add_argument('--train_with_mix_data', action='store_true', help='Train with mix data')

    parser.add_argument('--loss_type', type=str, nargs='+', default=['part_slice', 'whole_slice'], help='Loss type for group train')

    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing value for loss')

    parser.add_argument('--k', type=float, default=5, help='weighted k')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    config = Configuration()
    config.data_dir = args.data_dir
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
    config.k = args.k
    config.checkpoint_start = args.checkpoint_start
    config.share_weights = not(args.no_share_weights)
    config.freeze_layers = args.freeze_layers
    config.frozen_stages = args.frozen_stages

    train_script(config)