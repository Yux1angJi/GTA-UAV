import torch
import timm
import numpy as np
import torch.nn as nn
from PIL import Image
from urllib.request import urlopen
from thop import profile


class SimpleDecoder(nn.Module):
    def __init__(self, input_dim=1024, output_channels=3, feature_size=8):
        super(SimpleDecoder, self).__init__()
        
        self.decoder = nn.Sequential(
            # First layer to transform 1024*feature_size*feature_size to 512*2*feature_size*2*feature_size
            nn.ConvTranspose2d(input_dim, 512, kernel_size=4, stride=2, padding=1),  # 512 x 2*feature_size x 2*feature_size
            nn.ReLU(True),
            
            # Second layer to transform 512*2*feature_size*2*feature_size to 256*4*feature_size*4*feature_size
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 256 x 4*feature_size x 4*feature_size
            nn.ReLU(True),
            
            # Third layer to transform 256*4*feature_size*4*feature_size to 128*8*feature_size*8*feature_size
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 128 x 8*feature_size x 8*feature_size
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 128 x 8*feature_size x 8*feature_size
            nn.ReLU(True),
            
            # Final layer to transform 128*8*feature_size*8*feature_size to 3*32*32 (if feature_size=8)
            nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1),  # 3 x 32 x 32
        )
    
    def forward(self, x):
        x = self.decoder(x)
        return x

class MLP(nn.Module):
    def __init__(self, input_size=2048, hidden_size=512, output_size=2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class TimmModel(nn.Module):

    def __init__(self, 
                 model_name,
                 pretrained=True,
                 img_size=384,
                 share_weights=True,
                 train_with_recon=False,
                 train_with_offset=False):
                 
        super(TimmModel, self).__init__()
        self.share_weights = share_weights
        self.model_name = model_name
        self.img_size = img_size
        if share_weights:
            if "vit" in model_name or "swin" in model_name:
                # automatically change interpolate pos-encoding to img_size
                self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size) 
            else:
                self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        else:
            if "vit" in model_name or "swin" in model_name:
                self.model1 = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size)
                self.model2 = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size) 
            else:
                self.model1 = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
                self.model2 = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        if train_with_recon:
            if share_weights:
                self.decoder = SimpleDecoder()
            else:
                self.decoder1 = SimpleDecoder()
                self.decoder2 = SimpleDecoder()
        if train_with_offset:
            self.MLP = MLP()
        
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        

    def get_config(self,):
        if self.share_weights:
            data_config = timm.data.resolve_model_data_config(self.model)
        else:
            data_config = timm.data.resolve_model_data_config(self.model1)
        return data_config
    
    
    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

    def freeze_layers(self, frozen_blocks=10, frozen_stages=[0,0,0,0]):
        if 'convnext' in self.model_name:
            if self.share_weights:
                # stem
                for param in self.model.stem.parameters():
                    param.requires_grad = False
                # stage
                for i in range(len(self.model.stages)):
                    for j in range(frozen_stages[i]):
                        for param in self.model.stages[i].blocks[j].parameters():
                            param.requires_grad = False
            else:
                # stem
                for param in self.model1.stem.parameters():
                    param.requires_grad = False
                for param in self.model2.stem.parameters():
                    param.requires_grad = False
                # stage
                for i in range(len(self.model1.stages)):
                    for j in range(frozen_stages[i]):
                        for param in self.model1.stages[i].blocks[j].parameters():
                            param.requires_grad = False
                for i in range(len(self.model2.stages)):
                    for j in range(frozen_stages[i]):
                        for param in self.model2.stages[i].blocks[j].parameters():
                            param.requires_grad = False

        elif 'vit' in self.model_name:
            if self.share_weights:
                # patch_embed
                for param in self.model.patch_embed.parameters():
                    param.requires_grad = False
                # blocks
                for i in range(frozen_blocks):
                    for param in self.model.blocks[i].parameters():
                        param.requires_grad = False
            else:
                # patch_embed
                for param in self.model1.patch_embed.parameters():
                    param.requires_grad = False
                for param in self.model2.patch_embed.parameters():
                    param.requires_grad = False
                # blocks
                for i in range(frozen_blocks):
                    for param in self.model1.blocks[i].parameters():
                        param.requires_grad = False
                    for param in self.model2.blocks[i].parameters():
                        param.requires_grad = False


    def forward(self, img1=None, img2=None, forward_features=False):

        if self.share_weights:
            if img1 is not None and img2 is not None:
                if forward_features:
                    x1, x1_feature = self.model.forward_with_feature(img1)
                    x2, x2_feature = self.model.forward_with_feature(img2)
                    return x1, x1_feature, x2, x2_feature
                else:
                    image_features1 = self.model(img1)     
                    image_features2 = self.model(img2)
                    return image_features1, image_features2            
            elif img1 is not None:
                if forward_features:
                    x1, x1_feature = self.model.forward_with_feature(img1)
                    return x1, x1_feature
                else:
                    image_features = self.model(img1)
                    return image_features
            else:
                if forward_features:
                    x2, x2_feature = self.model.forward_with_feature(img1)
                    return x2, x2_feature
                else:
                    image_features = self.model(img2)
                    return image_features
        else:
            if img1 is not None and img2 is not None:
                if forward_features:
                    x1, x1_feature = self.model1.forward_with_feature(img1)
                    x2, x2_feature = self.model2.forward_with_feature(img2)
                    return x1, x1_feature, x2, x2_feature
                else:
                    image_features1 = self.model1(img1)     
                    image_features2 = self.model2(img2)
                    return image_features1, image_features2            
            elif img1 is not None:
                if forward_features:
                    x1, x1_feature = self.model1.forward_with_feature(img1)
                    return x1, x1_feature
                else:
                    image_features = self.model1(img1)
                    return image_features
            else:
                if forward_features:
                    x2, x2_feature = self.model2.forward_with_feature(img1)
                    return x2, x2_feature
                else:
                    image_features = self.model2(img2)
                    return image_features

    def offset_pred(self, img_feature1, img_feature2):
        offset = self.MLP(torch.cat((img_feature1, img_feature2), dim=1))
        return offset
    
    def decode(self, img_feature1, img_feature2):
        if self.share_weights:
            x1 = self.decoder(img_feature1)
            x2 = self.decoder(img_feature2)
        else:
            x1 = self.decoder1(img_feature1)
            x2 = self.decoder2(img_feature2)
        return x1, x2


if __name__ == '__main__':
    # model = TimmModel(model_name='convnext_base.fb_in22k_ft_in1k_384', train_with_recon=True)
    # model = TimmModel(model_name='timm/vit_large_patch16_384.augreg_in21k_ft_in1k')
    # # model = TimmModel(model_name='timm/vit_base_patch16_224.augreg_in1k')
    # # from timm.models.vision_transformer import vit_base_patch16_224
    # # model = vit_base_patch16_224(img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, num_classes=0)

    # # model = TimmModel(model_name='timm/swin_base_patch4_window12_384.ms_in22k_ft_in1k', img_size=384)
    # # model = TimmModel(model_name='vit_base_patch16_rope_reg1_gap_256.sbb_in1k')
    # # model = TimmModel(model_name='timm/vit_medium_patch16_rope_reg1_gap_256.sbb_in1k')
    # # model = TimmModel(model_name='timm/vit_medium_patch16_gap_256.sw_in12k_ft_in1k')
    # # model = TimmModel(model_name='timm/resnet101.tv_in1k') 
    # # img = Image.open(urlopen(
    # # 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
    # # ))
    # x = torch.rand((1, 3, 384, 384))
    # flops, params = profile(model, inputs=(x,))
    # # print(img.size)
    # # img = transform(img)
    # # print(img.size)

    # # print(model1)
    # print('flops(G)', flops/1e9, 'params(M)', params/1e6)

    from transformers import CLIPProcessor, CLIPModel
    model = CLIPModel.from_pretrained("/home/xmuairmud/jyx/clip-vit-base-patch16")
    vision_model = model.vision_model
    # print(vision_model)

    x = torch.rand((1, 3, 224, 224))
    # print(vision_model(x).pooler_output.shape)
    flops, params = profile(vision_model, inputs=(x,))
    print('flops(G)', flops/1e9, 'params(M)', params/1e6)


