import torch 
from .base_dip import BaseDeepImagePrior


class DeepImagePrior(BaseDeepImagePrior):
    def __init__(self, model, lr, num_steps, noise_std, callbacks=None):
        super().__init__(model, lr, num_steps, noise_std, callbacks)

    def compute_loss(self, x_pred, ray_trafo, y, **kwargs):
        L = kwargs.get("alpha", 1.0)  
        # TODO: Is the scaling by the lipschitz constant neccessary?
        loss = torch.sum((ray_trafo.trafo(x_pred) - y)**2/L**2) 
        mse_loss = loss 
        return loss, mse_loss 


