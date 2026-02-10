import torch
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from ..physics import power_iteration
from plutils import plot_nimages
from .utils import create_circular_mask, MaskedPSNR
from .base_dip import BaseDeepImagePrior


class REDDeepImagePrior(BaseDeepImagePrior):
    """
    This is an implementation of the Regularization by Denoising Deep Image Prior (RED-DIP)

    Matev et al. "DeepRED: Deep Image Prior Powered by RED" (TIP ICCV)
    (https://openaccess.thecvf.com/content_ICCVW_2019/papers/LCI/Mataev_DeepRED_Deep_Image_Prior_Powered_by_RED_ICCVW_2019_paper.pdf)


    """

    def __init__(
        self,
        model,
        lr,
        num_steps,
        denoise_strength,
        noise_std,
        denoiser=None,
        callbacks=None,
    ):
        super().__init__(model, lr, num_steps, noise_std, callbacks)
        self.denoise_strength = denoise_strength
        self.name = "REDDIP"  # or self.__class__.__name__
        if denoiser is None:
            raise ValueError("Please provide a denoiser function/module")
        self.denoiser = denoiser

    def compute_loss(self, x, ray_trafo, y, x_aux, lagrangian, beta=0.1, **kwargs):
        L2_inv = self.L2_inv if hasattr(self, "L2_inv") else kwargs.get("L2_inv", 1.0)
        mse_loss = ((ray_trafo.trafo(x) - y).pow(2)).sum() * L2_inv
        denoise_loss = beta * (x_aux - x - lagrangian).pow(2).sum()
        loss = mse_loss + denoise_loss
        return loss, mse_loss

    def train(
        self, ray_trafo, y, x_in, x_gt=None, return_metrics=True, logger=None, **kwargs
    ):
        if logger is None:
            from ..logging import NullLogger

            logger = NullLogger()

        self.inner_strategy = kwargs.get("inner_strategy", "fixed_point")
        if self.inner_strategy not in ["fixed_point", "gradient_step"]:
            raise ValueError("inner_strategy must be 'fixed_point' or 'gradient_step'")
        num_steps = kwargs.get("num_steps", getattr(self, "num_steps", 1000))
        num_inner_steps = kwargs.get(
            "num_inner_steps", getattr(self, "num_inner_steps", 50)
        )
        num_lower_steps = kwargs.get(
            "num_lower_steps", getattr(self, "num_lower_steps", 5)
        )
        sigma = kwargs.get("sigma", 2.0)
        self.L = kwargs.get("L")
        if self.L is None:
            with torch.no_grad():
                self.L = power_iteration(ray_trafo, torch.rand_like(x_in).view(-1, 1))
        self.L2_inv = 1.0 / self.L ** 2

        im_size = x_in.shape[-1]
        PSNR = MaskedPSNR(im_size=im_size, mask_fn=create_circular_mask)

        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        psnr_list, loss_list = [], []
        z = torch.nn.Parameter(x_in.detach().clone().to(x_in.device))
        lagrangian = torch.zeros_like(z).to(x_in.device).detach()
        x_aux = torch.zeros_like(x_in).to(x_in.device).detach()
        beta = 5e-4

        self.model.train()
        for i in (
            pbar := tqdm(
                range(num_steps // num_inner_steps),
                desc="Training REDDIP",
                dynamic_ncols=True,
            )
        ):
            for j in range(num_inner_steps):
                global_step = i * num_inner_steps + j

                optim.zero_grad()
                x_pred = self.model(z)
                loss, mse_loss = self.compute_loss(
                    x_pred, ray_trafo, y, x_aux, lagrangian, beta=beta
                )
                log_data = OrderedDict(
                    [
                        ("top_loss", loss.item()),
                        ("top_mse_loss", mse_loss.item()),
                        ("top_denoise_loss", (loss - mse_loss).item()),
                    ]
                )
                desc = f"{i}:{j:04d} | " + " | ".join(
                    f"{k}: {v:.4f}" for k, v in log_data.items()
                )
                plot_nimages(
                    x_pred,
                    titles=[
                        f"{global_step} mse: {mse_loss.item():.4f} denoise: {(loss-mse_loss).item():.4f}"
                    ],
                    save_path=f"results/tmp_reddit/{global_step:05d}.png",
                )
                pbar.set_description(desc)
                logger.log(log_data, step=global_step)
                loss.backward()
                optim.step()
            # breakpoint()
            with torch.no_grad():
                self.model.eval()
                x_pred = self.model(z)
                for j in range(num_lower_steps):
                    if self.inner_strategy == "fixed_point":
                        x_aux = x_aux - 0.001 * (
                            self.denoise_strength
                            * (x_aux - self.denoiser(x_aux, sigma))
                            + beta * (x_aux - x_pred - lagrangian)
                        )
                    elif self.inner_strategy == "gradient_step":
                        x_aux = (
                            1.0
                            / (self.denoise_strength + beta)
                            * (
                                self.denoise_strength * self.denoiser(x_aux, sigma)
                                + beta * (x_pred + lagrangian)
                            )
                        )

            with torch.no_grad():
                lagrangian = lagrangian + (x_pred - x_aux)
                temp_mse_loss = (
                    (ray_trafo.trafo(x_pred) - y).pow(2)
                ).sum() * self.L2_inv
                temp_denoise_loss = (
                    self.denoise_strength
                    * (x_pred * (x_pred - self.denoiser(x_pred, sigma))).sum()
                )
                temp_loss = temp_mse_loss + temp_denoise_loss
                loss_list.append(temp_loss.item())
                if x_gt is not None:
                    psnr = PSNR(x_gt, x_pred)
                    psnr_list.append(psnr)
                    logger.log(
                        {
                            "PSNR": psnr,
                            "loss": temp_loss,
                            "mse_loss": temp_mse_loss,
                            "denoise_loss": temp_denoise_loss,
                        },
                        step=global_step,
                    )
            logger.log_img(
                x_pred,
                step=global_step,
                title=f"Step {global_step:05d}"
                if x_gt is None
                else f"Step {global_step+1}, PSNR: {psnr_list[-1]:.2f}",
            )
            # temp_red_loss
            # loss, mse_loss = self.compute_loss(x_pred, ray_trafo, y, z)
            # loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            # optim.step()
            # logger.log({"MSE": mse_loss.item()}, step=global_step)
            # loss_list.append(mse_loss.item())

            # if x_gt is not None:
            #     psnr = PSNR(x_gt, x_pred)
            #     psnr_list.append(psnr)
            #     logger.log({"PSNR": psnr}, step=global_step)
            # else:
            #     psnr_list.append(0)

            # for cb in self.callbacks:
            #     cb(global_step, x_pred, loss, mse_loss, psnr_list[-1])

        if logger.use_wandb:
            logger.finish()

        if return_metrics:
            return z, psnr_list, loss_list
        else:
            return z
