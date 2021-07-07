import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from .connect import Corr_Up

class SiamFC(nn.Module):
    def __init__(self, config,  **kwargs):
        super(SiamFC, self).__init__()
        self.features = None
        self.connect_model = Corr_Up()
        self.zf = None  # for online tracking
        if kwargs['base'] is None:
            self.features = ResNet22W()
        else:
            self.features = kwargs['base'] 
        self.config = config
        self.model_alphaf = 0
        self.zf = None 
        self.features.eval()

    def feature_extractor(self, x):
        return self.features(x)

    def forward(self, x):
        xf = self.feature_extractor(x) * self.config.cos_window
        zf = self.zf 
        response = self.connect_model(zf, xf)
        return response
    
    def update(self, z, lr=0):
        zf = self.feature_extractor(z).detach() 
        _, _, ts, ts = zf.shape

        bg = ts//2-int(ts//(2*(self.config.padding+1)))
        ed = ts//2+int(ts//(2*(self.config.padding+1)))
        zf = zf[:,:,bg:ed, bg:ed]

        if self.zf is None:
            self.zf =  zf
        else:
            self.zf = (1 - lr) * self.zf + lr * zf



