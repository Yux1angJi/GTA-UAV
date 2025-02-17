import torch
import numpy as np
from tqdm import tqdm
import gc
import time
from torch.cuda.amp import autocast
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from geopy.distance import geodesic


def sdm(query_loc, sdmk_list, index, gallery_loc_xy_list, s=0.001):
    query_lat, query_lon = query_loc

    sdm_list = []

    for k in sdmk_list:
        sdm_nom = 0.0
        sdm_den = 0.0
        for i in range(k):
            idx = index[i]
            gallery_lat, gallery_lon = gallery_loc_xy_list[idx]
            d = geodesic((query_lat, query_lon), (gallery_lat, gallery_lon)).meters
            sdm_nom += (k - i) / np.exp(s * d)
            sdm_den += (k - i)
        sdm_list.append(sdm_nom/sdm_den)
    return sdm_list


def get_dis(query_loc, index, gallery_loc_xy_list, disk_list):
    query_lat, query_lon = query_loc
    dis_list = []
    for k in disk_list:
        dis_sum = 0.0
        for i in range(k):
            idx = index[i]
            gallery_lat, gallery_lon = gallery_loc_xy_list[idx]
            dis = geodesic((query_lat, query_lon), (gallery_lat, gallery_lon)).meters
            dis_sum += dis
        dis_list.append(dis_sum / k)

    return dis_list


def get_top10(index, gallery_list):
    top10 = []
    for i in range(10):
        idx = index[i]
        top10.append(gallery_list[idx])
    return top10


def predict(train_config, model, dataloader):
    
    model.eval()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
        
    query_feature_list = []
    
    with torch.no_grad():
        
        for sample in bar:
                    
            with autocast():
            
                drone_img = sample['drone_img'].to(train_config.device)
                drone_lidar_pts = sample['drone_lidar_pts'].to(train_config.device)
                drone_lidar_clr = sample['drone_lidar_clr'].to(train_config.device)
                drone_depth = sample['drone_depth'].to(train_config.device)
                drone_desc = sample['drone_desc']
                drone_desc = {key: value.to(train_config.device) for key, value in drone_desc.items()}

                query_feature = model(
                                    drone_img=drone_img, 
                                    drone_lidar_pts=drone_lidar_pts,
                                    drone_lidar_clr=drone_lidar_clr,
                                    drone_desc=drone_desc,
                                    drone_depth=drone_depth,
                                )
            
                # normalize is calculated in fp32
                if train_config.normalize_features:
                    query_feature = F.normalize(query_feature, dim=-1)
            
            # save features in fp32 for sim calculation
            query_feature_list.append(query_feature.to(torch.float32))

        # keep Features on GPU
        query_feature = torch.cat(query_feature_list, dim=0) 
        
    if train_config.verbose:
        bar.close()
    
    return query_feature


def evaluate(
        config,
        model,
        query_loader,
        gallery_loader,
        query_list,
        query_loc_xy_list,
        gallery_list,
        gallery_loc_xy_list,
        pairs_dict,
        ranks_list=[1, 5, 10],
        sdmk_list=[1, 3, 5],
        disk_list=[1, 3, 5],
        step_size=1000,
        cleanup=True,
        dis_threshold_list=[4*(i+1) for i in range(50)],
        plot_acc_threshold=False,
        top10_log=False,
    ):

    print("Extract Features and Compute Scores:")
    query_feature = predict(config, model, query_loader)
    # img_features_gallery = predict(config, model, gallery_loader)

    all_scores = []
    model.eval()
    with torch.no_grad():
        for gallery_batch in gallery_loader:
            with autocast():
                satellite_img = gallery_batch['satellite_img'].to(config.device)
                
                gallery_features_batch = model(
                    satellite_img=satellite_img,
                    satellite_desc=None,
                )

                if config.normalize_features:
                    gallery_features_batch = F.normalize(gallery_features_batch, dim=-1)

            scores_batch = query_feature @ gallery_features_batch.T
            all_scores.append(scores_batch.cpu())
    
    all_scores = torch.cat(all_scores, dim=1).numpy()
    # print('jyxjyxjyx', all_scores.shape)

    ap = 0.0

    gallery_idx = {}
    gallery_mapi_idx = {}
    for idx, gallery_img in enumerate(gallery_list):
        gallery_idx[gallery_img] = idx
        str_i = gallery_img.split('_')[0]
        gallery_mapi_idx.setdefault(str_i, []).append(idx)
    for k, v in gallery_mapi_idx.items():
        array = np.zeros(len(gallery_list), dtype=int)
        array[v] = 1
        gallery_mapi_idx[k] = array

    matches_list = []
    for query_i in query_list:
        pairs_list_i = pairs_dict[query_i]
        matches_i = []
        for pair in pairs_list_i:
            matches_i.append(gallery_idx[pair])
        matches_list.append(np.array(matches_i))

    matches_tensor = [torch.tensor(matches, dtype=torch.long) for matches in matches_list]

    query_num = query_feature.shape[0]

    all_ap = []
    cmc = np.zeros(len(gallery_list))
    sdm_list = []
    dis_list = []

    # for log
    top10_list = []
    loc1_list = []
    dis_ori_list = []
    dis_match_list = []
    acc_threshold = [0 for _ in range(len(dis_threshold_list))]

    for i in range(query_num):
        str_i = query_list[i].split('_')[0]
        score = all_scores[i] * gallery_mapi_idx[str_i]
        # predict index
        index = np.argsort(score)[::-1]

        sdm_list.append(sdm(query_loc_xy_list[i], sdmk_list, index, gallery_loc_xy_list))

        dis_list.append(get_dis(query_loc_xy_list[i], index, gallery_loc_xy_list, disk_list))

        top10_list.append(get_top10(index, gallery_list))
        loc1_lat, loc1_lon = gallery_loc_xy_list[index[0]]
        loc1_list.append((query_loc_xy_list[i][0], query_loc_xy_list[i][1], loc1_lat, loc1_lon))

        for j in range(len(dis_threshold_list)):
            if dis_list[i][0] < dis_threshold_list[j]:
                acc_threshold[j] += 1.
        
        good_index_i = np.isin(index, matches_tensor[i]) 
        
        # 计算 AP
        y_true = good_index_i.astype(int)
        y_scores = np.arange(len(y_true), 0, -1)  # 分数与排名相反
        if np.sum(y_true) > 0:  # 仅计算有正样本的情况
            ap = average_precision_score(y_true, y_scores)
            all_ap.append(ap)
        
        # 计算 CMC
        match_rank = np.where(good_index_i == 1)[0]
        if len(match_rank) > 0:
            cmc[match_rank[0]:] += 1
    
    mAP = np.mean(all_ap)
    cmc = cmc / query_num

    sdm_list = np.mean(np.array(sdm_list), axis=0)
    dis_list = np.mean(np.array(dis_list), axis=0)

    # top 1%
    top1 = round(len(gallery_list)*0.01)

    string = []

    for i in ranks_list:
        string.append('Recall@{}: {:.4f}'.format(i, cmc[i-1]*100))
        
    string.append('Recall@top1: {:.4f}'.format(cmc[top1]*100))
    string.append('AP: {:.4f}'.format(mAP*100))   
    
    for i in range(len(sdmk_list)):
        string.append('SDM@{}: {:.4f}'.format(sdmk_list[i], sdm_list[i]))
    for i in range(len(disk_list)):
        string.append('Dis@{}: {:.4f}'.format(disk_list[i], dis_list[i]))

    print(' - '.join(string)) 
    
    # cleanup and free memory on GPU
    if cleanup:
        del query_feature, gallery_features_batch, scores_batch
        gc.collect()
        #torch.cuda.empty_cache()

    if top10_log:
        for query_img, top10, loc, dis_ori, dis_match in zip(query_list, top10_list, loc1_list, dis_ori_list, dis_match_list):
            print('Query', query_img)
            print('Top10', top10)
            print('Query loc', loc[0], loc[1])
            print('Top1 loc', loc[2], loc[3])
            
            imgs_type = []
            for img_name in top10[:5]:
                if img_name in pairs_dict[query_img]:
                    imgs_type.append('Pos')
                else:
                    imgs_type.append('Null')
            print(imgs_type)

    if plot_acc_threshold:
        y = np.array(acc_threshold)
        x = np.array(dis_threshold_list)
        y = y / query_num * 100

        print(y.tolist())
    
    return cmc[0]
