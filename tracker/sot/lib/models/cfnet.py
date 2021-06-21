import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def complex_mul(x, z):
    out_real = x[..., 0] * z[..., 0] - x[..., 1] * z[..., 1]
    out_imag = x[..., 0] * z[..., 1] + x[..., 1] * z[..., 0]
    return torch.stack((out_real, out_imag), -1)


def complex_mulconj(x, z):
    out_real = x[..., 0] * z[..., 0] + x[..., 1] * z[..., 1]
    out_imag = x[..., 1] * z[..., 0] - x[..., 0] * z[..., 1]
    return torch.stack((out_real, out_imag), -1)

class CFNet(nn.Module):
    def __init__(self, config,  **kwargs):
        super(CFNet, self).__init__()
        self.features = None
        self.connect_model = None
        self.zf = None  # for online tracking
        if kwargs['base'] is None:
            self.features = ResNet22W()
        else:
            self.features = kwargs['base'] 
        self.config = config
        self.model_alphaf = 0
        self.model_zf = 0


    def feature_extractor(self, x):
        return self.features(x)

    def track(self, x):
        xf = self.feature_extractor(x)
        score = self.connector(self.zf, xf)
        return score

    def forward(self, x):
        x = self.feature_extractor(x) * self.config.cos_window
        xf = torch.rfft(x, signal_ndim=2)
        kxzf = torch.sum(complex_mulconj(xf, self.model_zf), dim=1, keepdim=True)
        response = torch.irfft(complex_mul(kxzf, self.model_alphaf), signal_ndim=2)
        # r_max = torch.max(response)
        # cv2.imshow('response', response[0, 0].data.cpu().numpy())
        # cv2.waitKey(0)
        return response
    
    def update(self, z, lr=0):
        z = self.feature_extractor(z) * self.config.inner_window
        zf = torch.rfft(z, signal_ndim=2)
        kzzf = torch.sum(torch.sum(zf ** 2, dim=4, keepdim=True), dim=1, keepdim=True)
        alphaf = self.config.yf / (kzzf + self.config.lambda0)
        self.model_alphaf = (1 - lr) * self.model_alphaf + lr * alphaf.data
        self.model_zf = (1 - lr) * self.model_zf + lr * zf.data



