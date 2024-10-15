import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn
import numpy as np
from pytorch_metric_learning import losses, miners


class BCEWithLogitsLossWithLabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super(BCEWithLogitsLossWithLabelSmoothing, self).__init__()
        self.smoothing = smoothing
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, logits, labels):
        # 计算平滑后的标签
        smoothed_labels = labels * (1 - self.smoothing) + 0.5 * self.smoothing
        loss = self.bce_with_logits(logits, smoothed_labels)
        return loss.mean()


class InfoNCE(nn.Module):

    def __init__(self, loss_function, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        
        self.loss_function = loss_function
        self.device = device

    def forward(self, image_features1, image_features2, logit_scale):
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)
        
        logits_per_image1 = logit_scale * image_features1 @ image_features2.T
        
        logits_per_image2 = logits_per_image1.T
        
        labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)
        
        loss = (self.loss_function(logits_per_image1, labels) + self.loss_function(logits_per_image2, labels))/2

        return loss  

def f(x):
    return 0.9 / (1 + np.exp(-5 * x))

class WeightedInfoNCE(nn.Module):
    def __init__(self, label_smoothing, k=-5, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.device = device
        self.k = k

    def loss(self, similarity_matrix, eps_all):
        n = similarity_matrix.shape[0]
        total_loss = 0.0
        for i in range(n):
            eps = eps_all[i]
            total_loss += (1 - eps) * (-1. * similarity_matrix[i, i] + torch.logsumexp(similarity_matrix[i, :], dim=0))
            total_loss += eps * (-1. / n * similarity_matrix[i, :].sum() + torch.logsumexp(similarity_matrix[i, :], dim=0))
        total_loss /= n
        return total_loss

    def forward(self, image_features1, image_features2, logit_scale, positive_weights=None):
        # Normalize the image features
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)
        
        # Compute similarity logits
        logits_per_image1 = logit_scale * image_features1 @ image_features2.T
        
        # Apply positive weights if provided
        if positive_weights is not None:
            eps = 1. - (1. - self.label_smoothing) / (1 + torch.exp(-self.k * positive_weights))
        else:
            eps = [self.label_smoothing for _ in range(image_features1.shape[0])]
        
        logits_per_image2 = logits_per_image1.T
        
        # Generate labels
        # labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)

        loss1 = self.loss(logits_per_image1, eps)
        loss2 = self.loss(logits_per_image2, eps)
        # # Compute loss
        # loss1 = self.loss_function(logits_per_image1, labels)
        # loss2 = self.loss_function(logits_per_image2, labels)
        loss = (loss1 + loss2) / 2

        return {"contrastive": loss}
    

class GroupInfoNCE(nn.Module):
    def __init__(self, group_len, label_smoothing, loss_type, k=3, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.group_len = group_len
        self.device = device
        self.loss_type = loss_type
        self.ce_loss = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        self.k = k

    def loss_contrastive_slice(self, similarity_matrix, G, N):
        total_loss = 0.0
        labels = torch.arange(G, dtype=torch.long, device=self.device)
        for i in range(N):
            for j in range(N):
                similarity_tmp = similarity_matrix[i::N, j::N]
                total_loss += self.ce_loss(similarity_tmp, labels)
        total_loss /= N * N
        return total_loss

    def loss_part_slice(self, similarity_matrix, G, N):
        total_loss = 0.0
        eps = self.label_smoothing

        I_g = torch.eye(G)
        # 创建一个 n x n 的单位矩阵
        I_n = torch.ones(N, N)
        # 使用 Kronecker 积生成目标矩阵
        labels = torch.kron(I_g, I_n).to(device=self.device)

        for i in range(G * N):
            all_matrix = similarity_matrix[i, :]
            y_i = labels[i]

            pos_matrix = all_matrix * y_i

            total_loss += (1 - eps) * ((-1. * pos_matrix).sum() + y_i.sum() * torch.logsumexp(all_matrix, dim=0))
            total_loss += eps * (-1. / G * all_matrix.sum() + N * torch.logsumexp(all_matrix, dim=0))
        return total_loss / G / N
    
    def loss_whole_slice(self, similarity_matrix, G, N):
        total_loss = 0.0
        eps = self.label_smoothing

        for i in range(G * N):
            all_matrix = similarity_matrix[i, :]

            g = i // N
            g_l = g * N
            g_r = g_l + N
            pos_matrix = all_matrix[g_l: g_r]
            all_logsumexp = torch.logsumexp(all_matrix, dim=0)

            total_loss += (1 - eps) * (-1. * torch.logsumexp(pos_matrix, dim=0) + all_logsumexp)

            for g in range(G):
                g_l = g * N
                g_r = g_l + N
                g_tmp_matrix = all_matrix[g_l: g_r]
                total_loss += eps / G * (-1. * torch.logsumexp(g_tmp_matrix, dim=0) + all_logsumexp)
        total_loss /= G * N
        return total_loss

    def loss_part_block(self, similarity_matrix, G, N):
        total_loss = 0.0
        eps = self.label_smoothing
        for g in range(G):
            g_l = g * N
            g_r = g_l + N

            g_pos_matrix = similarity_matrix[g_l: g_r, g_l: g_r]
            g_all_matrix = similarity_matrix[g_l: g_r, :].flatten()

            total_loss += (1 - eps) * (-1. / N / N * g_pos_matrix.sum() + torch.logsumexp(g_all_matrix, dim=0))
            total_loss += eps * (-1. / G / N / N * g_all_matrix.sum() + torch.logsumexp(g_all_matrix, dim=0))

        total_loss /= G
        return total_loss

    def loss_whole_block(self, similarity_matrix, G, N):
        total_loss = 0.0
        eps = self.label_smoothing
        for g in range(G):
            g_l = g * N
            g_r = g_l + N

            g_pos_matrix = similarity_matrix[g_l: g_r, g_l: g_r].flatten()
            g_all_matrix = similarity_matrix[g_l: g_r, :].flatten()

            pos_logsumexp = torch.logsumexp(g_pos_matrix, dim=0)
            all_logsumexp = torch.logsumexp(g_all_matrix, dim=0)

            total_loss += (1 - eps) * (-1.) * (pos_logsumexp - all_logsumexp)

            for g_2 in range(G):
                g_l_2 = g_2 * N
                g_r_2 = g_l_2 + N
                g_tmp_matrix = similarity_matrix[g_l: g_r, g_l_2: g_r_2].flatten()
                total_loss += eps / G * (-1.) * (torch.logsumexp(g_tmp_matrix, dim=0) - all_logsumexp)

        total_loss /= G
        return total_loss
        
    def forward(self, image_features1, image_features2, logit_scale, positive_weights=None):
        ## G*N, D
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)

        GN, D = image_features1.shape
        N = self.group_len
        G = GN // N

        if positive_weights is not None:
            eps = 1. - (1. - self.label_smoothing) / (1 + torch.exp(-self.k * positive_weights))
        else:
            eps = [self.label_smoothing for _ in range(image_features1.shape[0])]
        
        logits_per_image1 = logit_scale * image_features1 @ image_features2.T
        
        logits_per_image2 = logits_per_image1.T

        loss = {}

        for loss_type in self.loss_type:
            if loss_type == 'part_slice':
                loss[loss_type] = (self.loss_part_slice(logits_per_image1, G, N) + self.loss_part_slice(logits_per_image2, G, N))/2 
            elif loss_type == 'whole_slice':
                loss[loss_type] = (self.loss_whole_slice(logits_per_image1, G, N) + self.loss_whole_slice(logits_per_image2, G, N))/2 
            elif loss_type == 'part_block':
                loss[loss_type] = (self.loss_part_block(logits_per_image1, G, N) + self.loss_part_block(logits_per_image2, G, N))/2 
            elif loss_type == 'whole_block':
                loss[loss_type] = (self.loss_whole_block(logits_per_image1, G, N) + self.loss_whole_block(logits_per_image2, G, N))/2 
            elif loss_type == 'contrastive_slice':
                loss[loss_type] = (self.loss_contrastive_slice(logits_per_image1, G, N) + self.loss_contrastive_slice(logits_per_image2, G, N))/2 
        # loss = (self.loss_part(logits_per_image1, G, N) + self.loss_part(logits_per_image2, G, N))/2 
        
        # loss += (self.loss_whole_block(logits_per_image1, G, N) + self.loss_whole_block(logits_per_image2, G, N))/2
        # loss /= 2
        for k, v in loss.items():
            loss[k] /= 1. * len(self.loss_type)

        return loss


class ReconstructionLoss(nn.Module):
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device
        self.criterion = nn.MSELoss()

    def forward(self, recon_img, ori_img):
        x = self.criterion(recon_img, ori_img)
        return {"recon": x}


class OffsetLoss(nn.Module):
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device
        self.criterion = nn.MSELoss()

    def forward(self, offset_pred, query_loc_xy, ref_loc_xy):
        offset_label = query_loc_xy - ref_loc_xy
        x = self.criterion(offset_pred, offset_label)
        return {"recon": x}

class TripletLoss(nn.Module):
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device
        self.miner = miners.TripletMarginMiner(margin=0.2, type_of_triplets="semihard")
        self.criterion = losses.TripletMarginLoss(margin=0.2)

    def forward(self, image_features1, image_features2, logit_scale):
        N = image_features1.shape[0]
        labels = torch.arange(N)

        embeddings_all = torch.cat((image_features1, image_features2), dim=0)
        labels_all = torch.cat((labels, labels), dim=0)

        hard_pairs_all = self.miner(embeddings_all, labels_all)
        return {"triplet": self.criterion(embeddings_all, labels_all, hard_pairs_all)}


class MMWeightedInfoNCE(nn.Module):
    def __init__(self, label_smoothing, k=5, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.device = device
        self.k = k

    def loss(self, similarity_matrix, eps_all):
        n = similarity_matrix.shape[0]
        total_loss = 0.0
        for i in range(n):
            eps = eps_all[i]
            total_loss += (1 - eps) * (-1. * similarity_matrix[i, i] + torch.logsumexp(similarity_matrix[i, :], dim=0))
            total_loss += eps * (-1. / n * similarity_matrix[i, :].sum() + torch.logsumexp(similarity_matrix[i, :], dim=0))
        total_loss /= n
        return total_loss

    def forward(self, 
                drone_img_features, 
                drone_lidar_features, 
                satellite_img_features, 
                logit_scale, 
                positive_weights=None
                ):
        
        # print('jyxjyxjyx drone_lidar_features nan num', torch.isnan(drone_lidar_features).sum())
        # Normalize the features
        drone_img_features = F.normalize(drone_img_features, dim=-1)
        drone_lidar_features = F.normalize(drone_lidar_features, dim=-1)
        satellite_img_features = F.normalize(satellite_img_features, dim=-1)

        # Apply positive weights if provided
        if positive_weights is not None:
            eps = 1. - (1. - self.label_smoothing) / (1 + torch.exp(-self.k * positive_weights))
        else:
            eps = [self.label_smoothing for _ in range(drone_img_features.shape[0])]
        
        # Compute similarity logits
        logits_drone_img2satellite_img = logit_scale * drone_img_features @ satellite_img_features.T
        logits_satellite_img2drone_img = logits_drone_img2satellite_img.T

        logits_drone_img2drone_lidar = logit_scale * drone_img_features @ drone_lidar_features.T
        logits_drone_lidar2drone_img = logits_drone_img2drone_lidar.T

        logits_drone_lidar2satellite_img = logit_scale * drone_lidar_features @ satellite_img_features.T
        logits_satellite_img2drone_lidar = logits_drone_lidar2satellite_img.T

        loss_drone_img_drone_lidar = (self.loss(logits_drone_img2drone_lidar, eps) + self.loss(logits_drone_lidar2drone_img, eps)) / 2
        loss_drone_img_satellite_img = (self.loss(logits_drone_img2satellite_img, eps) + self.loss(logits_satellite_img2drone_img, eps)) / 2
        loss_drone_lidar_satellite_img = (self.loss(logits_drone_lidar2satellite_img, eps) + self.loss(logits_satellite_img2drone_lidar, eps)) / 2

        return {
            "contrastive_drone_img_drone_lidar": loss_drone_img_drone_lidar,
            "contrastive_drone_lidar_satellite_img": loss_drone_lidar_satellite_img,
            "contrastive_drone_img_satellite_img": loss_drone_img_satellite_img,
            }

if __name__ == '__main__':
    loss = GroupInfoNCE(group_len=2, label_smoothing=0.00)

    a = torch.rand(8, 8).cuda()
    loss1 = loss.loss_whole_block(a, 4, 2)
    loss2 = loss.loss_part_block(a, 4, 2)

    loss3 = loss.loss_whole_slice(a, 4, 2)
    loss4 = loss.loss_part_slice(a, 4, 2)

    print('loss_whole_block', loss1)
    print('loss_part_block', loss2)
    print('loss_whole_slice', loss3)
    print('loss_part_slice', loss4)   
