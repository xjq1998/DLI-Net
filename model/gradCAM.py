import torch
from torch.nn import ReLU
import numpy as np


def cal_euclidean_dist(ph_fea, sk_fea):
    dist = torch.pow(ph_fea-sk_fea, 2)
    dist = torch.sum(dist)
    dist = torch.sqrt(dist)
    return dist


class GradCAM():
    """
       Produces gradients generated with guided back propagation from the given image
    """

    def __init__(self, ph_encoder, sk_encoder, self_interaction, match):
        self.ph_encoder = ph_encoder
        self.sk_encoder = sk_encoder
        self.self_interaction=self_interaction
        self.match = match
        self.ph_gradients = None
        self.sk_gradients = None
        # Put model in evaluation mode
        self.ph_encoder.eval()
        self.sk_encoder.eval()

    
    def get_cam(self, ph, sk):
        ph_fea = self.ph_encoder(ph)
        sk_fea = self.sk_encoder(sk)
        if len(ph_fea) == 2:
            ph_fea = ph_fea[0]
            sk_fea = sk_fea[0]
        ph_fea.retain_grad()
        sk_fea.retain_grad()
        ph_activation = ph_fea.reshape(1,16,16,1024).cpu().detach()
        sk_activation = sk_fea.reshape(1,16,16,1024).cpu().detach()

        ph_self_fea = self.self_interaction(ph_fea)
        sk_self_fea = self.self_interaction(sk_fea)
        dis = self.match(ph_self_fea, sk_self_fea)
        self.ph_encoder.zero_grad()
        self.sk_encoder.zero_grad()
        dis.backward()

        ph_grad = ph_fea.grad.cpu().detach()
        sk_grad = sk_fea.grad.cpu().detach()
        print(ph_grad[0][0][0])
        ph_weight = torch.mean(ph_grad,dim=1)
        sk_weight = torch.mean(sk_grad,dim=1)

        ph_cam = torch.sum(ph_weight[:,None,None,:] * ph_activation, dim=3)
        sk_cam = torch.sum(sk_weight[:,None,None,:] * sk_activation, dim=3)
        return ph_cam, sk_cam
