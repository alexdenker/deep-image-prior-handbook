from typing_extensions import OrderedDict
import torch
import numpy as np
from tqdm import tqdm
import deepinv as dinv

from .utils import MaskedPSNR, create_circular_mask
from .base_dip import BaseDeepImagePrior
from ..physics import power_iteration


class DeepImagePriorADMMDenoiser(BaseDeepImagePrior):
    def __init__(
        self,
        model,
        lr,
        num_steps,
        denoise_strength,
        num_inner_steps,
        noise_std=0.0,
        denoiser=None,
        device=None,
        callbacks=None,
        save_dir=None,
    ):
        super().__init__(model, lr, num_steps, noise_std, callbacks, save_dir)

        self.denoise_strength = denoise_strength
        self.num_inner_steps = num_inner_steps

        if denoiser is None or denoiser == "tv":
            reg_denoiser = dinv.optim.prior.TVPrior(n_it_max=100)
            self.denoiser = lambda x, y: reg_denoiser.prox(x, gamma=y)
        else:
            self.denoiser = denoiser
        # Note: A custom denoiser should have the same interface as the lambda function above, ie it should take an image and a strength parameter and return the denoised image
        # NB: rundip function does this
        # EGs:
        #     self.denoiser = dinv.models.DRUNet(in_channels=1, out_channels=1, device = device if device is not None else "cpu")
        #     self.denoiser = dinv.models.DnCNN(in_channels=1, out_channels=1, device = device if device is not None else "cpu")
        #     self.denoiser = dinv.models.BM3DDenoiser()
        #     self.denoiser = dinv.models.GSDRUNet(in_channels=1, out_channels=1, device = device if device is not None else "cpu")
        #     raise ValueError(f"Denoiser {denoiser} not recognized. Choose from None, 'tv', 'drunet', 'dncnn', 'bm3d', 'gsdrunet'.")
        
        self.name = "ADMM_Denoiser_DIP"

    def compute_loss(self, x_pred, ray_trafo, y, u, lagrangian, beta, **kwargs):
        L2_inv = self.L2_inv if hasattr(self, "L2_inv") else kwargs.get("L2_inv", 1.0)
        beta = self.beta if hasattr(self, "beta") else beta

        mse_loss = ((ray_trafo.trafo(x_pred) - y).pow(2)).sum() * L2_inv
        denoise_loss = torch.mean((x_pred - u + lagrangian) ** 2)

        loss = mse_loss + beta * 0.5 * denoise_loss
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
        num_inner_steps = kwargs.get(
            "num_inner_steps", getattr(self, "num_inner_steps", 10)
        )

        exp_weight = kwargs.get("exp_weight", getattr(self, "exp_weight", 0))

        self.beta = kwargs.get("admm_weight", 10.0)
        
        self.L = kwargs.get("L")
        if self.L is None:
            with torch.no_grad():
                self.L = power_iteration(ray_trafo, torch.rand_like(x_in).view(-1, 1))
        self.L2_inv = 1.0 / self.L ** 2

        lagrangian = torch.zeros_like(x_in).detach()
        u = torch.zeros_like(x_in).detach()
        output_img = x_in.clone().detach()

        im_size = x_in.shape[-1]
        psnr_fun = MaskedPSNR(im_size=im_size, mask_fn=create_circular_mask)
        psnr_list, psnrden_list, loss_list = [], [], []

        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()

        for i in (
            pbar := tqdm(
                range(num_steps // num_inner_steps),
                desc="ADMM-Denoiser",
                dynamic_ncols=True,
            )
        ):
            for j in range(num_inner_steps):
                global_step = i * num_inner_steps + j
                optim.zero_grad()

                x_pred = self.model(x_in)
                loss, mse_loss = self.compute_loss(
                    x_pred, ray_trafo, y, u, lagrangian, beta=self.beta
                )

                log_data = OrderedDict(
                    [
                        ("loss", loss.item()),
                        ("mse_loss", mse_loss.item()),
                        ("denoise_loss", (loss - mse_loss).item()),
                    ]
                )
                logger.log(log_data, step=global_step)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optim.step()
                loss_list.append(mse_loss.item())

            with torch.no_grad():
                x_pred = self.model(x_in)
                u = self.denoiser(x_pred + lagrangian, self.denoise_strength)
                lagrangian = lagrangian + (x_pred - u)
                output_img = (
                    exp_weight * output_img + (1 - exp_weight) * x_pred.detach()
                )

            if x_gt is not None:
                psnr_list.append(psnr_fun(x_gt, output_img))
                psnrden_list.append(psnr_fun(x_gt, u))

            logger.log(
                {"psnr": psnr_list[-1], "psnr_denoised": psnrden_list[-1]},
                step=global_step,
            )
            logger.log_img(
                output_img,
                step=global_step,
                title=f"Step {global_step:05d}"
                if x_gt is None
                else f"Step {global_step+1}, PSNR: {psnr_list[-1]:.2f}",
            )

            desc_parts = [f"{i:04d}"]
            desc_parts += [f"{k}: {v:.4f}" for k, v in log_data.items()]
            if psnr_list:
                desc_parts.append(f"PSNR: {psnr_list[-1]:.2f}")
            if psnrden_list:
                desc_parts.append(f"PSNR denoised: {psnrden_list[-1]:.2f}")
            desc = " | ".join(desc_parts)
            pbar.set_description(desc)

            for cb in self.callbacks:
                cb(i, output_img, loss, mse_loss, psnr_list[-1])

        self.model.eval()
        with torch.no_grad():
            x_out = self.model(x_in)
            x_out = exp_weight * output_img + (1 - exp_weight) * x_out.detach()
        if logger.use_wandb:
            logger.finish()
        if return_metrics:
            return x_out, psnr_list, loss_list
        else:
            return x_out
