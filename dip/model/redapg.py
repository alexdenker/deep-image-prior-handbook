from typing_extensions import OrderedDict
import torch
import numpy as np
from tqdm import tqdm
import deepinv as dinv

from .utils import MaskedPSNR, create_circular_mask
from .base_dip import BaseDeepImagePrior
from ..physics import power_iteration


class DeepImagePriorREDAPG(BaseDeepImagePrior):
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
        regularization_strength=None,
    ):
        super().__init__(model, lr, num_steps, noise_std, callbacks, save_dir)

        self.num_inner_steps = num_inner_steps
        self.denoise_strength = denoise_strength
        self.regularization_strength = regularization_strength

        if denoiser is None or denoiser == "tv":
            reg_denoiser = dinv.optim.prior.TVPrior(n_it_max=100)
            self.denoiser = lambda x, y: reg_denoiser.prox(x, gamma=y)
        else:
            self.denoiser = denoiser
        # Note: A custom denoiser should have the same interface as the lambda function above, ie it should take an image and a strength parameter and return the denoised image
        # NB: rundip function does this

        self.name = "REDAPG_DIP"

    def compute_loss(self, x_pred, ray_trafo, y, u, **kwargs):
        mixing_L = self.mixing_L if hasattr(self, "mixing_L") else 1.0
        L2_inv = self.L2_inv if hasattr(self, "L2_inv") else kwargs.get("L2_inv", 1.0)

        mse_loss = ((ray_trafo.trafo(x_pred) - y).pow(2)).sum() * L2_inv
        denoise_loss = 0.5 * (x_pred - u).pow(2).sum()

        loss = mse_loss + mixing_L * self.regularization_strength * denoise_loss
        return loss, mse_loss

    def train(
        self, ray_trafo, y, x_in, x_gt=None, return_metrics=True, logger=None, **kwargs
    ):
        """
        Training the DIP.

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
        if self.regularization_strength is None: self.regularization_strength = self.denoise_strength / x_in.numel()
        print(f"Using regularization strength {self.regularization_strength:.4e}, self.denoise_strength: {self.denoise_strength:.4e}, numel: {x_in.numel()}")
        exp_weight = kwargs.get("exp_weight", getattr(self, "exp_weight", 0.0))
        
        u = torch.zeros_like(x_in)
        previous_xpred = torch.zeros_like(x_in)
        output_img = x_in.clone().detach()
        self.mixing_L = kwargs.get("mixing_weight", 1.0)

        self.L = kwargs.get("L")
        if self.L is None:
            with torch.no_grad():
                self.L = power_iteration(ray_trafo, torch.rand_like(x_in).view(-1, 1))
        self.L2_inv = 1.0 / self.L ** 2
        t_old = 1.0

        # psnr_fun = MaskedPSNR(x_in.shape[2])
        im_size = x_in.shape[-1]
        psnr_fun = MaskedPSNR(im_size=im_size, mask_fn=create_circular_mask)
        psnr_list, loss_list = [], []

        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()

        for i in (
            pbar := tqdm(
                range(num_steps // num_inner_steps),
                desc="RED-APG DIP",
                dynamic_ncols=True,
            )
        ):

            for j in range(num_inner_steps):
                global_step = i * num_inner_steps + j
                optim.zero_grad()

                x_pred = self.model(x_in)

                loss, mse_loss = self.compute_loss(x_pred, ray_trafo, y, u)
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
                t_new = (
                    (1.0 + np.sqrt(1.0 + 4.0 * t_old ** 2)) / 2.0
                    if global_step > 0
                    else 1.0
                )

                z = x_pred + (t_old - 1.0) / t_new * (x_pred - previous_xpred)
                u = (
                    1.0 / self.mixing_L * self.denoiser(z, self.denoise_strength)
                    - (1.0 - 1.0 / self.mixing_L) * z
                )
                previous_xpred = x_pred.clone()
                t_old = t_new
                output_img = (
                    exp_weight * output_img + (1 - exp_weight) * x_pred.detach()
                )

            if x_gt is not None:
                psnr_list.append(psnr_fun(x_gt, output_img))
                logger.log({"psnr": psnr_list[-1] if psnr_list else None}, step=global_step)

            logger.log_img(
                x_pred,
                step=global_step,
                title=f"Step {global_step:05d}"
                if x_gt is None
                else f"Step {global_step+1}, PSNR: {psnr_list[-1]:.2f}",
            )

            desc_parts = [f"{i:04d}"]
            desc_parts += [f"{k}: {v:.4f}" for k, v in log_data.items()]
            if psnr_list:
                desc_parts.append(f"PSNR: {psnr_list[-1]:.2f}")
            desc = " | ".join(desc_parts)
            pbar.set_description(desc)

            for cb in self.callbacks:
                cb(i, x_pred, loss, mse_loss, psnr_list[-1] if psnr_list else None)

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
