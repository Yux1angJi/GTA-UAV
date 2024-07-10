import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn


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


class ContrastiveLoss(nn.Module):
    def __init__(self, loss_function, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.loss_function = loss_function
        self.device = device

    def forward(self, image_features1, image_features2, logit_scale, positive_weights=None):
        # Normalize the image features
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)
        
        # Compute similarity logits
        logits_per_image1 = logit_scale * image_features1 @ image_features2.T
        
        # Apply positive weights if provided
        if positive_weights is not None:
            logits_per_image1 = logits_per_image1 * positive_weights.view(-1, 1)
        
        logits_per_image2 = logits_per_image1.T
        
        # Generate labels
        labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)
        
        # Compute loss
        loss1 = self.loss_function(logits_per_image1, labels)
        loss2 = self.loss_function(logits_per_image2, labels)
        loss = (loss1 + loss2) / 2

        return loss
    

class GroupInfoNCE(nn.Module):
    def __init__(self, group_len=2, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        
        self.group_len = group_len
        self.device = device


    def loss_part(self, similarity_matrix, G, N):
        total_loss = 0.0
        for g in range(G):
            g_l = g * N
            g_r = g_l + N

            g_pos_matrix = similarity_matrix[g_l: g_r, g_l: g_r]
            g_all_matrix = similarity_matrix[g_l: g_r, :].flatten()

            total_loss +=  -1. / N / N * g_pos_matrix.sum() + torch.logsumexp(g_all_matrix, dim=0)
        total_loss /= G
        return total_loss
    
    def loss_whole(self, similarity_matrix, G, N):
        total_loss = 0.0
        for g in range(G):
            g_l = g * N
            g_r = g_l + N

            g_pos_matrix = similarity_matrix[g_l: g_r, g_l: g_r].flatten()
            g_all_matrix = similarity_matrix[g_l: g_r, :].flatten()

            pos_logsumexp = torch.logsumexp(g_pos_matrix, dim=0)
            all_logsumexp = torch.logsumexp(g_all_matrix, dim=0)

            total_loss += -1. * (pos_logsumexp - all_logsumexp)
        total_loss /= G
        return total_loss
        

    def forward(self, image_features1, image_features2, logit_scale):
        ## G*N, D
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)

        GN, D = image_features1.shape
        N = self.group_len
        G = GN // N

        # I_g = torch.eye(G)
        # # 创建一个 n x n 的单位矩阵
        # I_n = torch.ones(N, N)
        # # 使用 Kronecker 积生成目标矩阵
        # labels = torch.kron(I_g, I_n).to(device=self.device)
        
        logits_per_image1 = logit_scale * image_features1 @ image_features2.T
        
        logits_per_image2 = logits_per_image1.T

        loss = (self.loss_part(logits_per_image1, G, N) + self.loss_part(logits_per_image2, G, N))/2 
        
        # loss += (self.loss_whole(logits_per_image1, G, N) + self.loss_whole(logits_per_image2, G, N))/2
        # loss /= 2

        return loss  
    

if __name__ == '__main__':
    loss = GroupInfoNCE()

    a = torch.rand(8, 8)
    loss1 = loss.loss_part(a, 4, 2)
    loss2 = loss.loss_whole(a, 4, 2)

    print(loss1, loss2)
