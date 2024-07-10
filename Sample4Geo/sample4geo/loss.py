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
    def __init__(self, group_len, label_smoothing, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        
        self.loss_function = BCEWithLogitsLossWithLabelSmoothing(label_smoothing)
        self.group_len = group_len
        self.device = device

    def forward(self, image_features1, image_features2, logit_scale):
        ## G*N, D
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)

        GN, D = image_features1.shape
        G = self.group_len
        N = GN // G

        I_g = torch.eye(G)
        # 创建一个 n x n 的单位矩阵
        I_n = torch.ones(N, N)
        # 使用 Kronecker 积生成目标矩阵
        labels = torch.kron(I_g, I_n).to(device=self.device)
        
        logits_per_image1 = logit_scale * image_features1 @ image_features2.T
        
        logits_per_image2 = logits_per_image1.T
        
        loss = (self.loss_function(logits_per_image1, labels) + self.loss_function(logits_per_image2, labels))/2

        return loss  
