import torch
import numpy as np
from tqdm import tqdm
import gc
import time
from torch.cuda.amp import autocast
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from PIL import Image
import pickle
import os


def sdm(query_loc, sdmk_list, index, gallery_loc_xy_list, s=0.001):
    query_x, query_y = query_loc

    sdm_list = []

    for k in sdmk_list:
        sdm_nom = 0.0
        sdm_den = 0.0
        for i in range(k):
            idx = index[i]
            gallery_x, gallery_y = gallery_loc_xy_list[idx]
            d = np.sqrt((query_x - gallery_x)**2 + (query_y - gallery_y)**2)
            sdm_nom += (k - i) / np.exp(s * d) 
            sdm_den += (k - i)
        sdm_list.append(sdm_nom/sdm_den)
    return sdm_list


def get_dis(query_loc, index, gallery_loc_xy_list, disk_list):
    query_x, query_y = query_loc
    dis_list = []
    for k in disk_list:
        dis_sum = 0.0
        for i in range(k):
            idx = index[i]
            gallery_x, gallery_y = gallery_loc_xy_list[idx]
            dis = np.sqrt((query_x - gallery_x)**2 + (query_y - gallery_y)**2)
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
        
    img_features_list = []
    
    with torch.no_grad():
        
        for img in bar:
                    
            with autocast():
            
                img = img.to(train_config.device)
                img_feature = model(img1=img)
            
                # normalize is calculated in fp32
                if train_config.normalize_features:
                    img_feature = F.normalize(img_feature, dim=-1)
            
            # save features in fp32 for sim calculation
            img_features_list.append(img_feature.to(torch.float32))

        # keep Features on GPU
        img_features = torch.cat(img_features_list, dim=0) 
        
    if train_config.verbose:
        bar.close()
    
    return img_features


def evaluate(config,
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
                top10_log=False):
    print("Extract Features and Compute Scores:")
    img_features_query = predict(config, model, query_loader)
    # img_features_gallery = predict(config, model, gallery_loader)

    all_scores = []
    model.eval()
    with torch.no_grad():
        for gallery_batch in gallery_loader:
            with autocast():
                gallery_batch = gallery_batch.to(device=config.device)
                gallery_features_batch = model(img2=gallery_batch)
                if config.normalize_features:
                    gallery_features_batch = F.normalize(gallery_features_batch, dim=-1)

            scores_batch = img_features_query @ gallery_features_batch.T
            all_scores.append(scores_batch.cpu())
    
    all_scores = torch.cat(all_scores, dim=1).numpy()
    # print('jyxjyxjyx', all_scores.shape)

    ap = 0.0

    gallery_idx = {}
    for idx, gallery_img in enumerate(gallery_list):
        gallery_idx[gallery_img] = idx

    matches_list = []
    for query_i in query_list:
        pairs_list_i = pairs_dict[query_i]
        matches_i = []
        for pair in pairs_list_i:
            matches_i.append(gallery_idx[pair])
        matches_list.append(np.array(matches_i))

    matches_tensor = [torch.tensor(matches, dtype=torch.long) for matches in matches_list]

    query_num = img_features_query.shape[0]

    all_ap = []
    cmc = np.zeros(len(gallery_list))
    sdm_list = []
    dis_list = []
    acc_threshold = [0 for _ in range(len(dis_threshold_list))]

    top10_list = []
    loc1_list = []

    for i in range(query_num):
        score = all_scores[i]    
        # predict index
        index = np.argsort(score)[::-1]

        sdm_list.append(sdm(query_loc_xy_list[i], sdmk_list, index, gallery_loc_xy_list))

        dis_list.append(get_dis(query_loc_xy_list[i], index, gallery_loc_xy_list, disk_list))

        top10_list.append(get_top10(index, gallery_list))
        loc1_x, loc1_y = gallery_loc_xy_list[index[0]]
        loc1_list.append((query_loc_xy_list[i][0], query_loc_xy_list[i][1], loc1_x, loc1_y))

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
        del img_features_query, gallery_features_batch, scores_batch
        gc.collect()
        #torch.cuda.empty_cache()

    if top10_log:
        satellite_dir = '/home/xmuairmud/data/GTA-UAV-data/randcam2_std0_stable_all/satellite_z41'
        pickle_path = '/home/xmuairmud/data/GTA-UAV-data/randcam2_std0_stable_all/same_h23456_z41_iou4_oc4/test_pair_meta.pkl'
        save_dir = '/home/xmuairmud/jyx/GTA-UAV/Game4Loc/visualization'
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
            test_semi_drone2sate_dict = data['pairs_semi_iou_drone2sate_dict']
            test_pos_drone2sate_dict = data['pairs_iou_drone2sate_dict']
        for query_img, top10, loc in zip(query_list, top10_list, loc1_list):
            print('Query', query_img)
            print('Top10', top10)
            print('Query loc', loc[0], loc[1])
            print('Top1 loc', loc[2], loc[3])
            
            imgs_path = []
            imgs_type = []
            for img_name in top10[:5]:
                imgs_path.append(os.path.join(satellite_dir, img_name))
                if img_name in test_pos_drone2sate_dict[query_img]:
                    imgs_type.append('Pos')
                elif img_name in test_semi_drone2sate_dict[query_img]:
                    imgs_type.append('Semi')
                else:
                    imgs_type.append('Null')
            print(imgs_type)

            images = [Image.open(x) for x in imgs_path]

            gap_width = 40
            total_width = sum(i.size[0] for i in images) + gap_width * (len(images) - 1)
            height = images[0].size[1]
            new_im = new_im = Image.new('RGBA', (total_width, height), (255, 255, 255, 0))

            x_offset = 0
            for im in images:
                new_im.paste(im, (x_offset, 0))
                x_offset += im.size[0] + gap_width  # 在每张图片后添加间隙
            new_im.save(f'{save_dir}/same_{query_img}')



    if plot_acc_threshold:
        y = np.array(acc_threshold)
        x = np.array(dis_threshold_list)
        y = y / query_num * 100

        x_new = np.linspace(x.min(), x.max(), 500)
        spl = make_interp_spline(x, y, k=3)  
        y_smooth = spl(x_new)

        plt.figure(figsize=(10, 6), dpi=300)
        plt.plot(x_new, y_smooth, label='Smooth Curve', color='red')
        plt.scatter(x, y, label='Discrete Points', color='blue')

        plt.xlabel('X Axis')
        plt.ylabel('Y Axis')
        plt.title('Smooth Curve with Discrete Points')
        plt.legend()

        # 调整边框
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        # 显示图表
        # plt.tight_layout()
        # plt.savefig('/home/xmuairmud/jyx/GTA-UAV-private/Game4Loc/images/plot_acc_threshold_samearea.png')
        print(y.tolist())
    
    return cmc[0]    

