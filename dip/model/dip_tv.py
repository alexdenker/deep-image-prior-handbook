import torch
import numpy as np
from tqdm import tqdm
import deepinv as dinv
from typing_extensions import OrderedDict

from .utils import MaskedPSNR, create_circular_mask, tv_loss
from .base_dip import BaseDeepImagePrior
from ..physics import power_iteration


class DeepImagePriorTV(BaseDeepImagePrior):
    def __init__(
        self,
        model,
        lr,
        num_steps,
        tv_strength,
        noise_std=0.0,
        callbacks=None,
        save_dir=None,
    ):
        super().__init__(model, lr, num_steps, noise_std, callbacks, save_dir)

        self.tv_strength = tv_strength

    def compute_loss(self, x_pred, ray_trafo, y, **kwargs):
        L = kwargs.get("L", 1.0)
        mse_loss = torch.sum((ray_trafo.trafo(x_pred) - y) ** 2 / L ** 2)
        loss = mse_loss + self.tv_strength * tv_loss(x_pred)
        return loss, mse_loss

class DeepImagePriorHQS(BaseDeepImagePrior):
    def __init__(
        self,
        model,
        lr,
        num_steps,
        splitting_strength,
        tv_min,
        tv_max,
        inner_steps,
        noise_std=0.0,
        callbacks=None,
        save_dir=None,
    ):
        super().__init__(model, lr, num_steps, noise_std, callbacks, save_dir)

        self.splitting_strength = splitting_strength
        self.tv_min = tv_min
        self.tv_max = tv_max
        self.inner_steps = inner_steps
        self.name = "TV_HQS_DIP"

    def train(self, ray_trafo, y, x_in, x_gt=None, return_metrics=True, logger=None):
        """
        Training the DIP.

        y: measurements
        x_in: input to DIP

        """
        if logger is None:
            from ..logging import NullLogger

            logger = NullLogger()

        prior = dinv.optim.prior.TVPrior(n_it_max=100)

        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        psnr_list = []
        loss_list = []

        x_splitting = torch.zeros_like(x_in)

        # Decreasing sequence for tv prox
        # At the beginning we want a strong regularisation and gradually decrease it
        tv_reg = np.logspace(
            np.log10(self.tv_max),
            np.log10(self.tv_min),
            self.num_steps // self.inner_steps,
        )[::-1]

        with torch.no_grad():
            L = power_iteration(ray_trafo, torch.rand_like(x_in).view(-1, 1))

        # psnr_fun = MaskedPSNR(x_in.shape[2])
        im_size = x_in.shape[-1]
        psnr_fun = MaskedPSNR(im_size=im_size, mask_fn=create_circular_mask)

        self.model.train()
        for i in tqdm(range(self.num_steps // self.inner_steps)):
            beta = self.splitting_strength / tv_reg[i]

            for j in range(self.inner_steps):
                global_step = i * self.inner_steps + j
                optim.zero_grad()

                if self.noise_std > 0:
                    x_pred = self.model(
                        x_in + self.noise_std ** 2 * torch.randn_like(x_in)
                    )
                else:
                    x_pred = self.model(x_in)

                mse_loss = torch.sum((ray_trafo.trafo(x_pred) - y) ** 2 / L ** 2)
                reg_loss = torch.mean((x_pred - x_splitting) ** 2)
                loss = mse_loss + beta * reg_loss
                log_data = OrderedDict(
                    [
                        ("loss", loss.item()),
                        ("mse_loss", mse_loss.item()),
                        ("denoise_loss", reg_loss.item()),
                    ]
                )
                logger.log(log_data, step=global_step)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optim.step()
                loss_list.append(mse_loss.item())

            with torch.no_grad():
                x_pred = self.model(x_in)
                # arg min_x 1/2 || x - inp ||_2^2 + gamma * TV(x)
                x_splitting = prior.prox(x_pred, gamma=tv_reg[i])
            if x_gt is not None:
                psnr_list.append(psnr_fun(x_gt, x_pred))
            else:
                psnr_list.append(0)

            for cb in self.callbacks:
                cb(global_step, x_pred, loss, mse_loss, psnr_list[-1])
        self.model.eval()
        with torch.no_grad():
            x_out = self.model(x_in)

        if logger.use_wandb:
            logger.finish()

        if return_metrics:
            return x_out, psnr_list, loss_list
        else:
            return x_out
