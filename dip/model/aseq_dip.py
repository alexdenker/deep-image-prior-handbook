import torch
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from ..physics import power_iteration

from .utils import create_circular_mask, MaskedPSNR
from ..logging import FlexibleLogger
from .base_dip import BaseDeepImagePrior


class AutoEncodingSequentialDeepImagePrior(BaseDeepImagePrior):
    """
    This is an implementation of the Autoencoding Sequential Deep Image Prior (AutoeSeqDIP)

    Alkhouri et al. "Image Reconstruction Via Autoencoding Sequential Deep Image Prior" (Neurips 2024)
    (https://openreview.net/forum?id=K1EG2ABzNE&noteId=KLbtyQ08BC)


    The method consists of alternating optimisation of the network parameters and a latent variable :math:`z`, over iterations (k) as follows

    .. math::

        \\theta_k \\leftarrow \\arg\\min_{\\theta_k}
        \\left\\| A f_{\\theta_k}(z_{k-1}) - y \\right\\|_2^2
        + \\lambda \\left\\| f_{\\theta_k}(z_{k-1}) - z_{k-1} \\right\\|_2^2

    where
    - :math:`f_\\theta` is the neural network with parameters :math:`\\theta`
    - :math:`z_{k-1}` is a latent variable (network input) updated at each iteration
    - :math:`A` is the forward operator (Radon transform)
    - :math:`y` is the measured data (sinogram)
    - :math:`\\lambda` is a regularisation parameter
    - The first term enforces **data consistency**.
    - The second term is a **denoiser regularisation** encouraging the network to be close to an autoencoder

    Once above is optimised wrt parameters, the latent variable is updated via

    .. math::

        z_k \\leftarrow f_{\\theta_k}(z_{k-1})

    The final reconstruction after :math:`K` steps is given by

    .. math::

        \\hat{x} = z_K = f_{\\theta_K}(z_{K-1})

    """

    def __init__(
        self, model, lr, num_steps, denoise_strength, noise_std, L=1.0, callbacks=None
    ):
        super().__init__(model, lr, num_steps, noise_std, callbacks)
        self.denoise_strength = denoise_strength
        self.name = "AutoEncodingSequentialDIP"  # or self.__class__.__name__

    def compute_loss(self, x, ray_trafo, y, z, **kwargs):
        loss_scaling = self.loss_scaling if hasattr(self, "loss_scaling") else 1.0
        L2_inv = self.L2_inv if hasattr(self, "L2_inv") else kwargs.get("L2_inv", 1.0)

        mse_loss = ((ray_trafo.trafo(x) - y).pow(2)).sum() * L2_inv
        denoise_loss = (x - z).pow(2).sum()
        loss = mse_loss + self.denoise_strength * loss_scaling * denoise_loss
        return loss, mse_loss  # TODO: report the reg loss if it exists?

    def train(self, ray_trafo, y, x_in, x_gt=None, return_metrics=True, logger= None, **kwargs):
        if logger is None:
            from ..logging import NullLogger
            logger = NullLogger()

        num_steps = kwargs.get("num_steps", getattr(self, "num_steps", 1000))
        num_inner_steps = kwargs.get(
            "num_inner_steps", getattr(self, "num_inner_steps", 1000)
        )

        self.L = kwargs.get("L")
        if self.L is None:
            with torch.no_grad():
                self.L = power_iteration(ray_trafo, torch.rand_like(x_in).view(-1, 1))
        self.L2_inv = 1.0 / self.L ** 2
        self.loss_scaling = y.shape[-1] / np.prod(x_in.shape)

        im_size = x_in.shape[-1]
        PSNR = MaskedPSNR(im_size=im_size, mask_fn=create_circular_mask)

        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        psnr_list, loss_list = [], []
        z = x_in.clone().detach()

        self.model.train()
        for i in (
            pbar := tqdm(
                range(num_steps // num_inner_steps),
                desc="Training SeqDIP",
                dynamic_ncols=True,
            )
        ):
            self.model.train()
            for j in range(num_inner_steps):
                global_step = i * num_inner_steps + j

                optim.zero_grad()
                x_pred = self.model(z)
                loss, mse_loss = self.compute_loss(x_pred, ray_trafo, y, z)

                log_data = OrderedDict(
                    [
                        ("loss", loss.item()),
                        ("mse_loss", mse_loss.item()),
                        ("denoise_loss", (loss - mse_loss).item()),
                    ]
                )

                desc = f"{i}:{j:04d} | " + " | ".join(
                    f"{k}: {v:.4f}" for k, v in log_data.items()
                )

                pbar.set_description(desc)
                logger.log(log_data, step=global_step)
                logger.log_img(
                    x_pred, step=global_step
                )  
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optim.step()

                if x_gt is not None:
                    psnr_list.append(PSNR(x_gt, x_pred))
                loss_list.append(log_data["loss"])
                for cb in self.callbacks:
                    cb(global_step, x_pred, loss, mse_loss, psnr_list[-1])

                logger.log({"psnr": psnr_list[-1]}, step=global_step)

            self.model.eval()
            with torch.no_grad():
                z = self.model(z).detach()

            


        if logger.use_wandb:
            logger.finish()

        if return_metrics:
            return z, psnr_list, loss_list
        else:
            return z
