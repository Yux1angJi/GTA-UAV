import time
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast
import torch.nn.functional as F

from ..utils import AverageMeter


def train_with_weight_dis(train_config, model, dataloader, loss_function, optimizer, 
    scheduler=None, scaler=None, 
    recon_weight=0.1, loss_recon=None, train_with_recon=False, 
    with_weight=False, 
    train_with_offset=False,
    loss_offset=None):

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
    for query, reference, weight, query_loc_xy, ref_loc_xy in bar:
        # print('jyxjyxjyx', query.shape)
        start_time = time.time()
        
        if scaler:
            with autocast():
                # data (batches) to device   
                query = query.to(train_config.device)
                reference = reference.to(train_config.device)
                weight = weight.to(train_config.device)
            
                # # Forward pass
                loss = {}

                if train_with_recon:
                    features1, features1_recon, features2, features2_recon = model(img1=query, img2=reference, forward_features=True)
                    img1_recon, img2_recon = model.decode(features1_recon, features2_recon)
                    loss_recon1 = loss_recon(img1_recon, query)
                    loss_recon2 = loss_recon(img2_recon, reference)
                    loss_recon_value = {"recon": recon_weight * (loss_recon1["recon"] + loss_recon2["recon"])}
                    loss.update(loss_recon_value)
                else:
                    features1, features2 = model(img1=query, img2=reference)
                
                if train_with_offset:
                    offset = model.predict_offset(features1, features2)
                    loss_offset = loss_offset(offset, query_loc_xy, ref_loc_xy)
                    loss.update(loss_offset)

                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                    if with_weight:
                        loss_sim = loss_function(features1, features2, model.module.logit_scale.exp(), weight)
                    else:
                        loss_sim = loss_function(features1, features2, model.module.logit_scale.exp())
                else:
                    if with_weight:
                        loss_sim = loss_function(features1, features2, model.logit_scale.exp(), weight)
                    else: 
                        loss_sim = loss_function(features1, features2, model.logit_scale.exp()) 
                
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

        else:
            raise NotImplementedError
        
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


def train_with_weight(train_config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None, recon_weight=0.1, loss_recon=None, train_with_recon=False, with_weight=False):

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
    for query, reference, weight in bar:
        # print('jyxjyxjyx', query.shape)
        start_time = time.time()
        
        if scaler:
            with autocast():
                # data (batches) to device   
                query = query.to(train_config.device)
                reference = reference.to(train_config.device)
                weight = weight.to(train_config.device)
            
                # # Forward pass
                loss = {}

                if train_with_recon:
                    features1, features1_recon, features2, features2_recon = model(img1=query, img2=reference, forward_features=True)
                    img1_recon, img2_recon = model.decode(features1_recon, features2_recon)
                    loss_recon1 = loss_recon(img1_recon, query)
                    loss_recon2 = loss_recon(img2_recon, reference)
                    loss_recon_value = {"recon": recon_weight * (loss_recon1["recon"] + loss_recon2["recon"])}
                    loss.update(loss_recon_value)

                else:
                    features1, features2 = model(query, reference)

                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                    if with_weight:
                        loss_sim = loss_function(features1, features2, model.module.logit_scale.exp(), weight)
                    else:
                        loss_sim = loss_function(features1, features2, model.module.logit_scale.exp())
                else:
                    if with_weight:
                        loss_sim = loss_function(features1, features2, model.logit_scale.exp(), weight)
                    else: 
                        loss_sim = loss_function(features1, features2, model.logit_scale.exp()) 
                
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
   
        else:
        
            query = query.to(train_config.device)
            reference = reference.to(train_config.device)
            weight = weight.to(train_config.device)
        
            # # Forward pass
            loss = {}

            if train_with_recon:
                features1, features1_recon, features2, features2_recon = model(img1=query, img2=reference, forward_features=True)
                img1_recon, img2_recon = model.decode(features1_recon, features2_recon)
                loss_recon1 = loss_recon(img1_recon, query)
                loss_recon2 = loss_recon(img2_recon, reference)
                loss_recon_value = {"recon": recon_weight * (loss_recon1["recon"] + loss_recon2["recon"])}
                loss.update(loss_recon_value)

            else:
                features1, features2 = model(img1=query, img2=reference)

            if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                if with_weight:
                    loss_sim = loss_function(features1, features2, model.module.logit_scale.exp(), weight)
                else:
                    loss_sim = loss_function(features1, features2, model.module.logit_scale.exp())
            else:
                if with_weight:
                    loss_sim = loss_function(features1, features2, model.logit_scale.exp(), weight)
                else: 
                    loss_sim = loss_function(features1, features2, model.logit_scale.exp()) 
            
            loss.update(loss_sim)
            
            loss_total = sum(loss.values())
            losses.update(loss_total.item())
            
            loss_total.backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)                  
            
            # Update model parameters (weights)
            optimizer.step()
            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
        
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


def train(train_config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None, num_chunks=1):

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
    for query, reference, ids in bar:
        # print('jyxjyxjyx', query.shape)
        start_time = time.time()
        
        if scaler:
            with autocast():
            
                # data (batches) to device   
                query = query.to(train_config.device)
                reference = reference.to(train_config.device)

                query_chunks = torch.chunk(query, num_chunks)
                reference_chunks = torch.chunk(reference, num_chunks)
                features1_all = []
                features2_all = []

                for (query, reference) in zip(query_chunks, reference_chunks):
                    features1, features2 = model(query, reference)
                    features1_all.append(features1)
                    features2_all.append(features2)
                    # print(features1.shape, features2.shape)
                features1 = torch.cat(features1_all, dim=0)
                features2 = torch.cat(features2_all, dim=0)

                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                    loss = loss_function(features1, features2, model.module.logit_scale.exp())
                else:
                    loss = loss_function(features1, features2, model.logit_scale.exp()) 
                losses.update(loss.item())
                
                  
            scaler.scale(loss).backward()
            
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
   
        else:
            raise NotImplementedError
        
        if train_config.verbose:
            
            monitor = {"loss": "{:.4f}".format(loss.item()),
                       "loss_avg": "{:.4f}".format(losses.avg),
                       "lr" : "{:.6f}".format(optimizer.param_groups[0]['lr'])}
            
            bar.set_postfix(ordered_dict=monitor)
        
        elif step % 50 == 1:
            end_time = time.time()
            iter_time = end_time - start_time
            elapsed_time = end_time - start_training_time
            eta = (elapsed_time / step) * (total_step - step)

            loss = "{:.4f}".format(loss.item())
            loss_avg = "{:.4f}".format(losses.avg),
            lr = "{:.6f}".format(optimizer.param_groups[0]['lr'])

            print(f"Iteration {step}/{total_step} took {iter_time:.4f} s, ETA: {eta:.2f} s, loss: {loss}, loss_avg: {loss_avg}, lr: {lr}", flush=True)
        
        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg


def predict(train_config, model, dataloader):
    
    model.eval()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
        
    img_features_list = []
    
    ids_list = []
    with torch.no_grad():
        
        for img, ids in bar:
        
            ids_list.append(ids)
            
            with autocast():
         
                img = img.to(train_config.device)
                img_feature = model(img)
            
                # normalize is calculated in fp32
                if train_config.normalize_features:
                    img_feature = F.normalize(img_feature, dim=-1)
            
            # save features in fp32 for sim calculation
            img_features_list.append(img_feature.to(torch.float32))
      
        # keep Features on GPU
        img_features = torch.cat(img_features_list, dim=0) 
        ids_list = torch.cat(ids_list, dim=0).to(train_config.device)
        
    if train_config.verbose:
        bar.close()
        
    return img_features, ids_list