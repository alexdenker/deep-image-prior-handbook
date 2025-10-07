import torch
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import matplotlib.pyplot as plt
from ..physics import power_iteration

from .utils import create_circular_mask, MaskedPSNR
from ..logging import FlexibleLogger
from .base_dip import BaseDeepImagePrior

# TODO imprt hit in __init__ dirsdo the


class SelfGuidanceDeepImagePrior(BaseDeepImagePrior):
    """
    This is an implementation of the Self-Guidance DIP
    Liang et al. "Robust Self-Guided Deep Image Prior" (TCI 2023)
        https://ieeexplore.ieee.org/document/10096631
    and its extension to inverse problems
    Liang et al. "Analysis of Deep Image Prior and Exploiting Self-Guidance for Image Reconstruction" (TCI 2025)
        https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10896571

    Optimisation Objective:

        .. math::

            \\hat{\\theta}, \\hat{z} = \\underset{\\theta, z}{\\arg\\min}
            \\left\\| A \\mathbb{E}_\\eta \\left[ f_\\theta(z + \\eta) \\right] - y \\right\\|_2^2
            + \\alpha \\left\\| \\mathbb{E}_\\eta \\left[ f_\\theta(z + \\eta) \\right] - z \\right\\|_2^2

        where:

        - :math:`f_\\theta` is a neatwork with parameters :math:`\\theta`
        - :math:`z` is a learnable latent variable (network input)
        - :math:`A` is the forward operator (Radon transform)
        - :math:`y` is the measured data (sinogram)
        - :math:`\\eta` is additive Gaussian noise
        - The first term enforces **data consistency**.
        - The second term is a **denoiser regularisation**.

        The final reconstruction is given by

        .. math::

            \\hat{x} = \\mathbb{E}_\\eta \\left[ f_{\\hat{\\theta}}(\\hat{z} + \\eta) \\right]

        or as an exponential moving average of the network outputs.

        .. math::
            \\hat{x}_t = \\beta \\hat{x}_{t-1} + (1 - \\beta) \\mathbb{E}_\\eta \\left[ f_{\\hat{\\theta}}(\\hat{z} + \\eta) \\right]

    """

    def __init__(
        self, model, lr, num_steps, denoise_strength, noise_std, rel_noise=0.01, L=1.0, callbacks=None
    ):
        super().__init__(model, lr, num_steps, noise_std, callbacks)
        self.denoise_strength = denoise_strength
        self.rel_noise = rel_noise
        self.name = "SelfGuidanceDIP"  # or self.__class__.__name__

    def compute_loss(self, x, x_pred_mean, ray_trafo, y, z, **kwargs):
        L2_inv = self.L2_inv if hasattr(self, "L2_inv") else kwargs.get("L2_inv", 1.0)
        num_noise_realisations = (
            self.num_noise_realisations
            if hasattr(self, "num_noise_realisations")
            else x.shape[0]
        )

        if x.shape[0] != z.shape[0] or x.shape[0] != num_noise_realisations:
            raise ValueError(
                f"Number of noise realisations in x ({x.shape[0]}) and z ({z.shape[0]}) do not match."
            )

        mse_loss = ((ray_trafo.trafo(x_pred_mean) - y).pow(2)).sum() * L2_inv
        denoise_loss = (x - z).pow(2).sum() * num_noise_realisations
        loss = mse_loss + self.denoise_strength * denoise_loss
        return loss, mse_loss  # TODO: report the reg loss if it exists?

    # def compute_loss(self, x, ray_trafo, y, z, **kwargs):
    #     L2_inv = self.L2_inv if hasattr(self, "L2_inv") else kwargs.get("L2_inv", 1.0)
    #     num_noise_realisations = (
    #         self.num_noise_realisations
    #         if hasattr(self, "num_noise_realisations")
    #         else x.shape[0]
    #     )

    #     mse_loss = ((ray_trafo.trafo(x) - y).pow(2)).sum() * L2_inv
    #     denoise_loss = (x - z).pow(2).sum() * num_noise_realisations
    #     loss = mse_loss + self.denoise_strength * denoise_loss
    #     return loss, mse_loss  # TODO: report the reg loss if it exists?


    def train(self, ray_trafo, y, x_in, x_gt=None, return_metrics=True, logger = None, **kwargs):
        if logger is None:
            from ..logging import NullLogger
            logger = NullLogger()
        num_steps = kwargs.get("num_steps", getattr(self, "num_steps", 1000))
        self.num_noise_realisations = kwargs.get(
            "num_noise_realisations", getattr(self, "num_noise_realisations", 4)
        )
        exp_weight = kwargs.get("exp_weight", getattr(self, "exp_weight", 0.99))
        lr_z = kwargs.get("lr_z", getattr(self, "lr_z", 1e-1))
        self.L = kwargs.get("L")
        if self.L is None:
            with torch.no_grad():
                self.L = power_iteration(ray_trafo, torch.rand_like(x_in).view(-1, 1))

        self.L2_inv = 1.0 / self.L ** 2
        self.loss_scaling = y.shape[-1] / np.prod(x_in.shape)

        im_size = x_in.shape[-1]
        PSNR = MaskedPSNR(im_size=im_size, mask_fn=create_circular_mask)

        psnr_list, loss_list = [], []

        # Make z learnable
        exp_average = torch.zeros_like(x_in)
        device = x_in.device
        z = torch.nn.Parameter(x_in.detach().clone().to(device))
        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        optim_z = torch.optim.Adam([z], lr=lr_z)
        # optim = torch.optim.Adam([
        #             {'params': self.model.parameters(), 'lr': self.lr},
        #             {'params': [z], 'lr': lr_z}
        #         ])
        
        self.model.train()
        
        for i in (
            pbar := tqdm(
                range(num_steps), desc="Training SelfGuidanceDIP", dynamic_ncols=True
            )
        ):
            optim.zero_grad()
            optim_z.zero_grad()
            
            noise_max = self.rel_noise *z.max().detach()

            # x_pred = torch.zeros_like(x_in, device=device)
            # for j in range(self.num_noise_realisations):
            #     noise = noise_max * torch.rand_like(z, device=device)
            #     x_pred += self.model(z+noise).squeeze()
            # x_pred /= self.num_noise_realisations
            # breakpoint()
            # loss, mse_loss = self.compute_loss(x_pred, ray_trafo, y, z)

            z_expand = z.expand(
                self.num_noise_realisations, -1, -1, -1
            )  # Expand z for N_noise
            eta = (
                noise_max * torch.rand_like(z_expand)
            )  # Random noise for self-guidance
            noisy_z = z_expand + eta
            x_pred = self.model(noisy_z)
            x_pred_mean = x_pred.mean(dim=0).unsqueeze(
                0
            )  # Average over noise realisations

            loss, mse_loss = self.compute_loss(
                x_pred, x_pred_mean, ray_trafo, y, z_expand
            )
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_([z], max_norm=1.0)
            optim.step()
            optim_z.step()
            # torch.nn.utils.clip_grad_norm_(list(self.model.parameters()) + [z], max_norm=1.0)
            # optim.step()

            with torch.no_grad():
                exp_average = (
                    exp_weight * exp_average + (1 - exp_weight) * x_pred.detach()
                )
            loss_list.append(loss.item())
            if x_gt is not None:
                psnr_list.append(PSNR(x_gt, exp_average))

            log_data = OrderedDict(
                [
                    ("loss", loss_list[-1]),
                    ("mse_loss", mse_loss.item()),
                    ("denoise_loss", (loss - mse_loss).item()),
                    ("psnr", psnr_list[-1] if psnr_list else 0),
                ]
            )

            desc = f"{i:04d} | " + " | ".join(
                f"{k}: {v:.4f}" for k, v in log_data.items()
            )
            pbar.set_description(desc)
            logger.log(log_data, step=i)
            logger.log_img(exp_average, step=i, title=f"Step {i:04d} | PSNR: {psnr_list[-1]:.2f}" if psnr_list else None)

            for cb in self.callbacks:
                cb(i, x_pred, loss, mse_loss, psnr_list[-1])

        if logger.use_wandb:
            logger.finish()

        if return_metrics:
            return exp_average, psnr_list, loss_list
        else:
            return exp_average
