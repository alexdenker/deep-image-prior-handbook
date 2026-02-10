from typing_extensions import OrderedDict
import torch
import numpy as np
from tqdm import tqdm
import deepinv as dinv

from .utils import MaskedPSNR, create_circular_mask
from .base_dip import BaseDeepImagePrior
from ..physics import power_iteration


class DeepImagePriorAPGDADenoiser(BaseDeepImagePrior):
    def __init__(
        self,
        model,
        lr,
        num_steps,
        denoise_strength,
        noise_std=0.0,
        denoiser=None,
        device=None,
        callbacks=None,
        save_dir=None,
    ):
        super().__init__(model, lr, num_steps, noise_std, callbacks, save_dir)

        self.denoise_strength = denoise_strength

        device = device if device is not None else "cpu"
        if denoiser is None or denoiser == "tv":
            reg_denoiser = dinv.optim.prior.TVPrior(n_it_max=100)
            self.denoiser = lambda x, y: reg_denoiser.prox(x, gamma=y)
        else:
            self.denoiser = denoiser
        # Note: A custom denoiser should have the same interface as the lambda function above, ie it should take an image and a strength parameter and return the denoised image
        # NB: rundip function does this
        self.name = "APGDA_Denoiser_DIP"

    def compute_loss(self, x_pred, ray_trafo, y, u, lagrangian, beta, **kwargs):
        L2_inv = self.L2_inv if hasattr(self, "L2_inv") else kwargs.get("L2_inv", 1.0)
        beta = self.beta if hasattr(self, "beta") else beta

        mse_loss = ((ray_trafo.trafo(x_pred) - y).pow(2)).sum() * L2_inv
        # denoise_loss = torch.mean((x_pred -u + lagrangian/beta) ** 2)
        denoise_loss = (x_pred - u + lagrangian / beta).pow(2).sum()

        loss = mse_loss + 0.5 * beta * denoise_loss
        return loss, mse_loss

    def train(
        self, ray_trafo, y, x_in, x_gt=None, return_metrics=True, logger=None, **kwargs
    ):
        """
        Training the DIP

        y: measurements
        x_in: input to DIP

        """
        if logger is None:
            from ..logging import NullLogger

            logger = NullLogger()

        num_steps = kwargs.get("num_steps", getattr(self, "num_steps", 1000))

        self.beta = kwargs.get("admm_weight", 10.0) / x_in.numel()

        self.L = kwargs.get("L")
        if self.L is None:
            with torch.no_grad():
                self.L = power_iteration(ray_trafo, torch.rand_like(x_in).view(-1, 1))
        self.L2_inv = 1.0 / self.L ** 2

        lagrangian = torch.zeros_like(x_in).detach()
        u = torch.zeros_like(x_in).detach()

        im_size = x_in.shape[-1]
        psnr_fun = MaskedPSNR(im_size=im_size, mask_fn=create_circular_mask)
        psnr_list, loss_list = [], []

        # if optimiser := kwargs.get("optimizer", None):
        #     optim = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0)
        # else:
        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()

        for i in (
            pbar := tqdm(
                range(num_steps),
                desc="APGDA-Denoiser",
                dynamic_ncols=True,
            )
        ):
            optim.zero_grad()

            x_pred = self.model(x_in)
            loss, mse_loss = self.compute_loss(
                x_pred, ray_trafo, y, u, lagrangian, beta=self.beta
            )
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )  # do i want this?
            optim.step()
            loss_list.append(mse_loss.item())

            with torch.no_grad():
                u = self.denoiser(
                    x_pred + lagrangian / self.beta,
                    np.sqrt(1.9 * self.denoise_strength / self.beta),
                )
                x_pred = self.model(x_in)
                lagrangian = lagrangian + (1.9 * self.beta) * (x_pred - u)

            log_data = OrderedDict(
                [
                    ("loss", loss.item()),
                    ("mse_loss", mse_loss.item()),
                    ("denoise_loss", (loss - mse_loss).item()),
                ]
            )

            if x_gt is not None:
                psnr_list.append(psnr_fun(x_gt, x_pred))
                log_data["psnr"] = psnr_list[-1]

            logger.log(log_data, step=i)
            logger.log_img(
                x_pred,
                step=i,
                title=f"Step {i:05d}"
                if x_gt is None
                else f"Step {i+1}, PSNR: {psnr_list[-1]:.2f}",
            )
            desc = f"{i:04d} | " + " | ".join(
                f"{k}: {v:.4f}" for k, v in log_data.items()
            )
            pbar.set_description(desc)

            for cb in self.callbacks:
                cb(i, x_pred, loss, mse_loss, psnr_list[-1])

        self.model.eval()
        with torch.no_grad():
            x_out = self.model(x_in)

        if logger.use_wandb:
            logger.finish()
        if return_metrics:
            return x_out, psnr_list, loss_list
        else:
            return x_out
