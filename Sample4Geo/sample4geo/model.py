import torch
import timm
import numpy as np
import torch.nn as nn


class TimmModel(nn.Module):

    def __init__(self, 
                 model_name,
                 pretrained=True,
                 img_size=383):
                 
        super(TimmModel, self).__init__()
        
        self.img_size = img_size
        
        if "vit" in model_name:
            # automatically change interpolate pos-encoding to img_size
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size) 
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        

    def get_config(self,):
        data_config = timm.data.resolve_model_data_config(self.model)
        return data_config
    
    
    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

        
    def forward(self, img1, img2=None, forward_features=False):
        n = img1.shape[0]

        if img2 is not None:
            if forward_features:
                image_features1 = self.model.forward_features(img1).reshape(n, -1)
                image_features2 = self.model.forward_features(img2).reshape(n, -1)
            else:
                image_features1 = self.model(img1)     
                image_features2 = self.model(img2)
            return image_features1, image_features2            
              
        else:
            if forward_features:
                image_features = self.model.forward_features(img1).reshape(n, -1)
            else:
                image_features = self.model(img1)
            return image_features


if __name__ == '__main__':
    model = TimmModel(model_name='convnext_base.fb_in22k_ft_in1k_384')
    x = torch.rand((1, 3, 1024, 1024))
    print(model.model.forward_features(x).reshape(1, -1).shape)