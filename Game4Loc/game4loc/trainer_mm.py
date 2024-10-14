import time
import torch
from tqdm import tqdm
from .utils import AverageMeter
from torch.cuda.amp import autocast
import torch.nn.functional as F



def train_mm_with_weight(train_config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None, recon_weight=0.1, loss_recon=None, train_with_recon=False, with_weight=False):

    # set model train mode
    model.train()
    
    losses = AverageMeter()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)
    
    step = 1
    start_training_time = time.time()
    total_step = len(dataloader)
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
    
    # for loop over one epoch
    for sample in bar:
        # print('jyxjyxjyx', query.shape)
        start_time = time.time()
        
        if scaler:
            with autocast():
                # data (batches) to device
                drone_img = sample['drone_img'].to(train_config.device)
                drone_lidar_pts = sample['drone_lidar_pts'].to(train_config.device)
                drone_lidar_clr = sample['drone_lidar_clr'].to(train_config.device)
                satellite_img = sample['satellite_img'].to(train_config.device)
                weight = sample['positive_weight'].to(train_config.device)
            
                # # Forward pass
                loss = {}

                x = model(drone_img=drone_img, 
                          drone_lidar_pts=drone_lidar_pts,
                          drone_lidar_clr=drone_lidar_clr,
                          satellite_img=satellite_img)
                drone_img_features = x['drone_img_features']
                drone_lidar_features = x['drone_lidar_features']
                satellite_img_features = x['satellite_img_features']

                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                    if with_weight:
                        loss_sim = loss_function(drone_img_features, drone_lidar_features, satellite_img_features, model.module.logit_scale.exp(), weight)
                    else:
                        loss_sim = loss_function(drone_img_features, drone_lidar_features, satellite_img_features, model.module.logit_scale.exp())
                else:
                    if with_weight:
                        loss_sim = loss_function(drone_img_features, drone_lidar_features, satellite_img_features, model.logit_scale.exp(), weight)
                    else: 
                        loss_sim = loss_function(drone_img_features, drone_lidar_features, satellite_img_features, model.logit_scale.exp())
                
                loss.update(loss_sim)
                
                loss_total = sum(loss.values())
                losses.update(loss_total.item())
            
            scaler.scale(loss_total).backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad) 
            
            # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
    
        # else:
        
        #     query = query.to(train_config.device)
            # reference = reference.to(train_config.device)
            # weight = weight.to(train_config.device)
        
            # # # Forward pass
            # loss = {}

            # features1, features2 = model(img1=query, img2=reference)

            # if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
            #     if with_weight:
            #         loss_sim = loss_function(features1, features2, model.module.logit_scale.exp(), weight)
            #     else:
            #         loss_sim = loss_function(features1, features2, model.module.logit_scale.exp())
            # else:
            #     if with_weight:
            #         loss_sim = loss_function(features1, features2, model.logit_scale.exp(), weight)
            #     else: 
            #         loss_sim = loss_function(features1, features2, model.logit_scale.exp()) 
            
            # loss.update(loss_sim)
            
            # loss_total = sum(loss.values())
            # losses.update(loss_total.item())
            
            # loss_total.backward()
            
            # # Gradient clipping 
            # if train_config.clip_grad:
            #     torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)                  
            
            # # Update model parameters (weights)
            # optimizer.step()
            # # Zero gradients for next step
            # optimizer.zero_grad()
            
            # # Scheduler
            # if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
            #     scheduler.step()
        
        if train_config.verbose:
            
            monitor = {"loss": "{:.4f}".format(loss_total.item()),
                       "loss_avg": "{:.4f}".format(losses.avg),
                       "lr" : "{:.6f}".format(optimizer.param_groups[0]['lr'])}
            loss = {k: "{:.4f}".format(v) for k,v in loss.items()}
            monitor.update(loss)
            
            bar.set_postfix(ordered_dict=monitor)
        
        elif step % 50 == 1:
            end_time = time.time()
            iter_time = end_time - start_time
            elapsed_time = end_time - start_training_time
            eta = (elapsed_time / step) * (total_step - step)

            loss_value = "{:.4f}".format(loss_total.item())
            loss_avg = "{:.4f}".format(losses.avg),
            lr = "{:.6f}".format(optimizer.param_groups[0]['lr'])

            print_log = f"Iteration {step}/{total_step} took {iter_time:.4f} s, ETA: {eta:.2f} s, loss: {loss_value}, loss_avg: {loss_avg}, lr: {lr}, "
            for k, v in loss.items():
                print_log += "{}: {:.4f}, ".format(k, v)

            print(print_log, flush=True)
        
        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg

