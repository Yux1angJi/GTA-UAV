import torch
import timm
import numpy as np
import torch.nn as nn
from PIL import Image
from urllib.request import urlopen
from thop import profile


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


class DesModel(nn.Module):

    def __init__(self, 
                 model_name='vit',
                 pretrained=True,
                 img_size=384,
                 share_weights=True,
                 train_with_recon=False,
                 train_with_offset=False,
                 model_hub='timm'):
                 
        super(DesModel, self).__init__()
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
        if self.share_weights:
            self.model.set_grad_checkpointing(enable)
        else:
            self.model1.set_grad_checkpointing(enable)
            self.model2.set_grad_checkpointing(enable)

    def freeze_layers(self, frozen_blocks=10, frozen_stages=[0,0,0,0]):
        pass

    def forward(self, img1=None, img2=None):

        if self.share_weights:
            if img1 is not None and img2 is not None:
                image_features1 = self.model(img1)     
                image_features2 = self.model(img2)
                return image_features1, image_features2            
            elif img1 is not None:
                image_features = self.model(img1)
                return image_features
            else:
                image_features = self.model(img2)
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


if __name__ == '__main__':
    # model = TimmModel(model_name='timm/vit_large_patch16_384.augreg_in21k_ft_in1k')
    # # model = TimmModel(model_name='timm/vit_base_patch16_224.augreg_in1k')
    # # from timm.models.vision_transformer import vit_base_patch16_224
    # # model = vit_base_patch16_224(img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, num_classes=0)


    # model = DesModel(model_name='timm/resnet101.tv_in1k', img_size=384)
    # model = DesModel(model_name='convnext_base.fb_in22k_ft_in1k_384', img_size=384)
    model = DesModel(model_name='timm/swin_base_patch4_window7_224.ms_in22k_ft_in1k', img_size=384)
    # # model = TimmModel(model_name='vit_base_patch16_rope_reg1_gap_256.sbb_in1k')
    # # model = TimmModel(model_name='timm/vit_medium_patch16_rope_reg1_gap_256.sbb_in1k')
    # # model = TimmModel(model_name='timm/vit_medium_patch16_gap_256.sw_in12k_ft_in1k')
    # # model = TimmModel(model_name='timm/resnet101.tv_in1k') 
    # # img = Image.open(urlopen(
    # # 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
    # # ))
    x = torch.rand((1, 3, 384, 384))
    x = x.cuda()
    model.cuda()
    x = model(x)
    print(x.shape)

    # flops, params = profile(model, inputs=(x,))
    # # print(img.size)
    # # img = transform(img)
    # # print(img.size)

    # # print(model1)
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


