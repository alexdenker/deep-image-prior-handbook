import torch 
import numpy as np 
from tqdm import tqdm 
from skimage.metrics import peak_signal_noise_ratio
import deepinv as dinv 
from collections import OrderedDict


from .utils import create_circular_mask, MaskedPSNR
from ..logging import FlexibleLogger
from .base_dip import BaseDeepImagePrior

class SequentialDeepImagePrior(BaseDeepImagePrior):
    def __init__(self, model, lr, num_steps, reg_strength, noise_std, L=1.0, callbacks=None):
        super().__init__(model, lr, num_steps, noise_std, L, callbacks)
        self.denoise_strength = denoise_strength
        self.L2_inv = 1./self.L**2 # todo add to base dip?

    def compute_loss(self, x, ray_trafo, y, z, **kwargs):
        mse_loss = ((ray_trafo.trafo(x) - y).pow(2)).sum() * self.L2_inv
        denoise_loss = (x - z).pow(2).sum()
        loss = mse_loss + self.denoise_strength * loss_scaling * denoise_loss
        return loss, mse_loss #TODO: report the reg loss if it exists?

    def train(self, ray_trafo, y, x_in, x_gt=None, return_metrics=True, **kwargs):
        logger = FlexibleLogger(use_wandb=False)
        im_size = x_in.shape[-1]
        PSNR = MaskedPSNR(im_size=im_size, mask_fn=create_circular_mask)
        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        psnr_list, loss_list = [], []
        z = x_in.clone().detach()
        self.loss_scaling = y.shape[-1]/np.prod(z.shape)

        self.model.train()
        for i in (pbar:=tqdm(range(num_steps), desc="Training SeqDIP", dynamic_ncols=True)):

            for j in range(num_inner_steps):
                global_step = i * num_inner_steps + j

                optim.zero_grad()
                x_pred = self.model(z)
                loss, mse_loss = self.compute_loss(x_pred, ray_trafo, y, z)

                log_data = OrderedDict([
                    ("loss", loss.item()),
                    ("mse_loss", mse_loss.item()),
                    ("denoise_loss", (loss - mse_loss).item()),
                ])

                desc = f"{i}:{inner_step:04d} | " + " | ".join(
                    f"{k}: {v:.4f}" for k, v in log_data.items()
                )

                pbar.set_description(desc)
                logger.log(log_data, step=global_step)
                loss.backward()
                optim.step()

            self.model.eval()
            with torch.no_grad():
                z = model(z).detach()
            if x_gt is not None:
                psnr_list.append(PSNR(x_gt, x_pred))
            loss_list.append(log_data["loss"])
            for cb in self.callbacks:
                cb(i, x_pred, loss, mse_loss, psnr_list[-1]) #TODO: add storage and printing callbacks?

        if logger.use_wandb:
            logger.finish()
        if return_metrics:
            return z, psnr_list, loss_list
        else:
            return z