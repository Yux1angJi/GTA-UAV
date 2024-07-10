import os
import time
import math
import shutil
import sys
import torch
from dataclasses import dataclass
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup

from sample4geo.dataset.visloc import VisLocDatasetEval, VisLocDatasetTrain, get_transforms
from sample4geo.dataset.gta import GTADatasetTrain
from sample4geo.dataset.mix_data import MixDatasetTrain
from sample4geo.utils import setup_system, Logger
from sample4geo.trainer import train, train_with_weight
from sample4geo.evaluate.gta import evaluate
from sample4geo.loss import InfoNCE, ContrastiveLoss, GroupInfoNCE
from sample4geo.model import TimmModel


@dataclass
class Configuration:
    
    # Model
    model: str = 'convnext_base.fb_in22k_ft_in1k_384'
    
    # Override model image size
    img_size: int = 384

    num_chunks: int = 1

    ### GTA setting
    
    train_with_weight: bool = False

    train_in_group: bool = False
    group_len = 2

    train_with_mix_data: bool = True
    
    # Training 
    mixed_precision: bool = True
    custom_sampling: bool = True         # use custom sampling instead of random
    seed = 1
    epochs: int = 1
    batch_size: int = 64                # keep in mind real_batch_size = 2 * batch_size
    verbose: bool = False
    gpu_ids: tuple = (0)           # GPU ids for training
    
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
    
    # Learning Rate
    lr: float = 0.001                    # 1 * 10^-4 for ViT | 1 * 10^-1 for CNN
    scheduler: str = "cosine"           # "polynomial" | "cosine" | "constant" | None
    warmup_epochs: int = 0.1
    lr_end: float = 0.0001               #  only for "polynomial"
    
    # Augment Images
    prob_flip: float = 0.5              # flipping the sat image and drone image simultaneously
    
    # Savepath for model checkpoints
    model_path: str = "./work_dir/visloc"

    dataset: str= "VisLoc-D2S"
    
    # Eval before training
    zero_shot: bool = False
    
    # Checkpoint to start from
    checkpoint_start = None
    # checkpoint_start = "pretrained/university/convnext_base.fb_in22k_ft_in1k_384/weights_e1_0.9515.pth"
    # checkpoint_start = 'work_dir/gta/convnext_base.fb_in22k_ft_in1k_384/0708211147/weights_end.pth'

    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 4 
    
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 
    
    # for better performance
    cudnn_benchmark: bool = True
    
    # make cudnn deterministic
    cudnn_deterministic: bool = False


#-----------------------------------------------------------------------------#
# Train Config                                                                #
#-----------------------------------------------------------------------------#

config = Configuration() 

if config.dataset == 'VisLoc-D2S':
    config.train_pairs_meta_file = '/home/xmuairmud/data/UAV_VisLoc_dataset/data1234_z3/train_pair_meta.pkl'
    config.test_pairs_meta_file = '/home/xmuairmud/data/UAV_VisLoc_dataset/data1234_z3/test_pair_meta.pkl'
    # config.data_root_dir = '/home/xmuairmud/data/UAV_VisLoc_dataset/data_1_2/test/satellite'
    config.sate_img_dir = '/home/xmuairmud/data/UAV_VisLoc_dataset/data1234_z3/all_satellite'

    config.extra_train_pairs_meta_file = '/home/xmuairmud/data/GTA-UAV-data/randcam2_std0_stable/train_h23456_z567/train_pair_meta.pkl'


if __name__ == '__main__':

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
    train_dataset = VisLocDatasetTrain(pairs_meta_file=config.train_pairs_meta_file,
                                      transforms_query=train_sat_transforms,
                                      transforms_gallery=train_drone_transforms,
                                      group_len=config.group_len,
                                      prob_flip=config.prob_flip,
                                      shuffle_batch_size=config.batch_size,
                                      )
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  shuffle=not config.custom_sampling,
                                  pin_memory=True)
    
    if config.train_with_mix_data:
        train_dataset_extra = GTADatasetTrain(pairs_meta_file=config.extra_train_pairs_meta_file,
                                      transforms_query=train_sat_transforms,
                                      transforms_gallery=train_drone_transforms,
                                      prob_flip=config.prob_flip,
                                      shuffle_batch_size=config.batch_size,
                                      )
        train_dataset_mix = MixDatasetTrain(batch_size=config.batch_size,
                                            transforms_query=train_sat_transforms,
                                            transforms_gallery=train_drone_transforms,
                                            prob_flip=config.prob_flip,
                                            )
        train_dataset_mix.update([train_dataset.samples, train_dataset_extra.samples])
        train_dataloader = DataLoader(train_dataset_mix,
                                        batch_size=config.batch_size,
                                        num_workers=config.num_workers,
                                        shuffle=not config.custom_sampling,
                                        pin_memory=True)
        

    # Test query
    query_dataset_test = VisLocDatasetEval(pairs_meta_file=config.test_pairs_meta_file,
                                        mode="drone",
                                        transforms=val_transforms,
                                        )
    query_img_list = query_dataset_test.images
    pairs_drone2sate_dict = query_dataset_test.pairs_drone2sate_dict
    
    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    # Test gallery
    gallery_dataset_test = VisLocDatasetEval(pairs_meta_file=config.test_pairs_meta_file,
                                               mode="sate",
                                               transforms=val_transforms,
                                               sate_img_dir=config.sate_img_dir,
                                               )
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

    if config.train_in_group:
        loss_function = GroupInfoNCE(
            group_len=config.group_len,
            label_smoothing=config.label_smoothing,
            device=config.device,
        )
    else:
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        loss_function = ContrastiveLoss(
            loss_function=loss_fn,
            device=config.device,
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
                           step_size=1000,
                           cleanup=True)

    #-----------------------------------------------------------------------------#
    # Shuffle                                                                     #
    #-----------------------------------------------------------------------------#
    if config.train_with_mix_data:  
        train_dataset.shuffle()
        train_dataset_extra.shuffle()
        train_dataloader.dataset.update([train_dataset.samples, train_dataset_extra.samples])          
    elif config.train_in_group:
        train_dataloader.dataset.shuffle_group()
    else:
        train_dataloader.dataset.shuffle()

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
                           num_chunks=config.num_chunks,
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
                                step_size=1000,
                                cleanup=True)
                
            if r1_test > best_score:

                best_score = r1_test

                if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
                    torch.save(model.module.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))
                else:
                    torch.save(model.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))
                
        if config.train_with_mix_data:  
            train_dataset.shuffle()
            train_dataset_extra.shuffle()
            train_dataloader.dataset.update([train_dataset.samples, train_dataset_extra.samples])          
        elif config.train_in_group:
            train_dataloader.dataset.shuffle_group()
        else:
            train_dataloader.dataset.shuffle()
                
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        torch.save(model.module.state_dict(), '{}/weights_end.pth'.format(model_path))
    else:
        torch.save(model.state_dict(), '{}/weights_end.pth'.format(model_path))            