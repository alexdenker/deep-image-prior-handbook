import torch 
from .base_dip import BaseDeepImagePrior


class DeepImagePrior(BaseDeepImagePrior):
    def __init__(self, model, lr, num_steps, noise_std, L=1.0, callbacks=None):
        super().__init__(model, lr, num_steps, noise_std, L, callbacks)

    def compute_loss(self, x_pred, ray_trafo, y, **kwargs):
        
        loss = torch.sum((ray_trafo.trafo(x_pred) - y)**2/self.L**2) 
        mse_loss = loss 
        return loss, mse_loss 


