import torch
import timm
import numpy as np
import torch.nn as nn
from PIL import Image
from urllib.request import urlopen
from thop import profile

from .vit_adapter import ViTAdapter


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


class DesModelWithRGBD(nn.Module):

    def __init__(self, 
                 model_name='vit',
                 pretrained=True,
                 img_size=384,
                 share_weights=True,
                 diff_guidance=-1.0,
                 train_with_recon=False,
                 train_with_offset=False,
                 model_hub='timm',
                 global_pool='avg'):
                 
        super(DesModelWithRGBD, self).__init__()
        self.share_weights = share_weights
        self.model_name = model_name
        self.img_size = img_size
        if share_weights:
            if "vit" in model_name or "swin" in model_name:
                # automatically change interpolate pos-encoding to img_size
                # self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size) 
                self.model = ViTAdapter(global_pool=global_pool, pretrained=pretrained)
            else:
                self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        else:
            if "vit" in model_name or "swin" in model_name:
                model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size)
                patch_embed = model.patch_embed

                # 修改 patch_embed 以接受 4 通道输入
                new_proj = nn.Conv2d(4, patch_embed.proj.out_channels, kernel_size=patch_embed.proj.kernel_size, stride=patch_embed.proj.stride, padding=patch_embed.proj.padding)

                # 用原来的 3 通道权重初始化新的前 3 个通道
                with torch.no_grad():
                    new_proj.weight[:, :3] = patch_embed.proj.weight

                # 替换原来的 patch embedding 层
                patch_embed.proj = new_proj
                model.patch_embed = patch_embed

                self.model1 = model
                
                self.model2 = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size) 
            else:
                model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
                old_conv = model.stem[0]
                new_conv = torch.nn.Conv2d(4, old_conv.out_channels, 
                                        kernel_size=old_conv.kernel_size, 
                                        stride=old_conv.stride, 
                                        padding=old_conv.padding, 
                                        bias=old_conv.bias is not None)

                # 初始化新卷积层的前3个通道使用原始权重，第四通道使用0初始化
                new_conv.weight.data[:, :3, :, :] = old_conv.weight.data
                new_conv.weight.data[:, 3:, :, :] = 0

                # 替换模型中的第一个卷积层
                model.stem[0] = new_conv
                self.model1 = model

                self.model2 = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        if train_with_offset:
            self.MLP = MLP()

        self.diff_guidance = diff_guidance
        
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        

    def get_config(self,):
        if self.share_weights:
            data_config = timm.data.resolve_model_data_config(self.model.vit_model)
        else:
            data_config = timm.data.resolve_model_data_config(self.model1)
        return data_config
    
    
    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)


    def forward(self, img1=None, img2=None):

        if self.share_weights:
            if img1 is not None and img2 is not None:
                if self.diff_guidance > 0.0:
                    image_features1_rgb, _ = self.model(img1[:, :3, :, :])
                    image_features1_rgbd, lamda = self.model(img1)
                    # image_features1 = lamda * image_features1_rgbd + (1 - lamda) * image_features1_rgb
                    image_features1 = self.diff_guidance * (image_features1_rgbd - image_features1_rgb) + image_features1_rgb
                    image_features2, _ = self.model(img2)
                else:
                    image_features1, _ = self.model(img1)
                    image_features2, _ = self.model(img2)
                return image_features1, image_features2        
            elif img1 is not None:
                if self.diff_guidance > 0.0:
                    image_features1_rgb, _ = self.model(img1[:, :3, :, :])
                    image_features1_rgbd, lamda = self.model(img1)
                    # image_features = lamda * image_features1_rgbd + (1 - lamda) * image_features1_rgb
                    image_features = self.diff_guidance * (image_features1_rgbd - image_features1_rgb) + image_features1_rgb
                else:
                    image_features, _ = self.model(img1)
                return image_features
            else:
                image_features, _ = self.model(img2)
                return image_features
        else:
            if img1 is not None and img2 is not None:
                image_features1 = self.model1(img1)     
                image_features2 = self.model2(img2)
                return image_features1, image_features2            
            elif img1 is not None:
                image_features = self.model1(img1)
                return image_features
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
    # model = TimmModel(model_name='timm/vit_large_patch16_384.augreg_in21k_ft_in1k')
    # # model = TimmModel(model_name='timm/vit_base_patch16_224.augreg_in1k')
    # # from timm.models.vision_transformer import vit_base_patch16_224
    # # model = vit_base_patch16_224(img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, num_classes=0)


    # model = DesModel(model_name='timm/resnet101.tv_in1k', img_size=384)
    # model = DesModel(model_name='convnext_base.fb_in22k_ft_in1k_384', img_size=384)
    model = DesModelWithRGBD(model_name='vit_base_patch16_rope_reg1_gap_256.sbb_in1k', img_size=384)
    # # model = TimmModel(model_name='vit_base_patch16_rope_reg1_gap_256.sbb_in1k')
    # # model = TimmModel(model_name='timm/vit_medium_patch16_rope_reg1_gap_256.sbb_in1k')
    # # model = TimmModel(model_name='timm/vit_medium_patch16_gap_256.sw_in12k_ft_in1k')
    # # model = TimmModel(model_name='timm/resnet101.tv_in1k') 
    # # img = Image.open(urlopen(
    # # 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
    # # ))
    x = torch.rand((1, 4, 384, 384))
    # flops, params = profile(model, inputs=(x,))
    # # print(img.size)
    # # img = transform(img)
    # # print(img.size)

    print(model(x).shape)
    # print('flops(G)', flops/1e9, 'params(M)', params/1e6)

    # from transformers import CLIPProcessor, CLIPModel
    # model = CLIPModel.from_pretrained("/home/xmuairmud/jyx/clip-vit-base-patch16")
    # vision_model = model.vision_model
    # print(vision_model)

    # dinov2_vitb14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
    # print(dinov2_vitb14_reg.set_grad_checkpointing(True))

    # from transformers import ViTModel, ViTImageProcessor, AutoModelForImageClassification, AutoConfig
    # config = AutoConfig.from_pretrained('facebook/dino-vitb16')
    # config.image_size = 384
    # model = ViTModel.from_pretrained('facebook/dino-vitb16', config=config, ignore_mismatched_sizes=True)
    # model = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', pretrained=True, img_size=(384, 384))
    # data_config = timm.data.resolve_model_data_config(model)
    # print(data_config)
    # processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb16')


    # x = torch.rand((1, 3, 384, 384))
    # inputs = processor(images=x, return_tensors="pt")
    # print(inputs['pixel_values'].shape)
    # outputs = model(**inputs)
    # print(outputs.pooler_output.shape)
    # print(model(x).shape)
    # flops, params = profile(dinov2_vitb14_reg, inputs=(x,))
    # print('flops(G)', flops/1e9, 'params(M)', params/1e6)


