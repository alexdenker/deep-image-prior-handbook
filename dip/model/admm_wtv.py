import torch
from tqdm import tqdm
from collections import OrderedDict
from ..physics import power_iteration

from .utils import create_circular_mask, MaskedPSNR
from .base_dip import BaseDeepImagePrior

class WeightedTVDeepImagePrior(BaseDeepImagePrior):
    """
    This is an implementation of the ADMM Weighted TV Deep Image Prior (ADMM-WTV-DIP)

    Cascarano et al. "Combining Weighted Total Variation and Deep Image Prior for natural and medical image restoration via ADMM" (ICCSA 2021)
    (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9732379&tag=1)

    The method consists of alternating optimisation of the network parameters and a latent variable :math:`z`, over iterations (k) as follows

    .. math::

    where
    - :math:`f_\\theta` is the neural network with parameters :math:`\\theta`
    - :math:`A` is the forward operator (Radon transform)
    - :math:`y` is the measured data (sinogram)
    - :math:`\\lambda` is a regularisation parameter
    - The first term enforces **data consistency**.
    - The second term is a **weighted total variation regularisation**.

    """

    def __init__(
        self, model, lr, num_steps, tv_strength, noise_std, variable_regularisation=False, callbacks=None
    ):
        super().__init__(model, lr, num_steps, noise_std, callbacks)
        self.tv_strength = tv_strength
        self.variable_regularisation = variable_regularisation
        self.name = "ADMMWeightedTV_DIP"

    def D(self, x, pad=True):
        dh  = x[..., :, 1:] - x[..., :, :-1]
        dw  = x[..., 1:, :] - x[..., :-1, :]
        if pad:
            dw = torch.nn.functional.pad(dw, (0, 0, 0, 1))
            dh = torch.nn.functional.pad(dh, (0, 1, 0, 0))
        return torch.cat((dh, dw), dim=0)

    def compute_top_loss(self, x, ray_trafo, y, t, lagrangian, beta=10., **kwargs):
        L2_inv = self.L2_inv if hasattr(self, "L2_inv") else kwargs.get("L2_inv", 1.0)
        mse_loss = ((ray_trafo.trafo(x) - y).pow(2)).sum() * L2_inv
        denoise_loss = (0.5*beta)* (self.D(x, pad=True)- t + lagrangian/beta).pow(2).sum()
        loss = mse_loss + denoise_loss
        return loss, mse_loss

    def train(self, ray_trafo, y, x_in, x_gt=None, return_metrics=True, logger=None, **kwargs):
        if logger is None:
            from ..logging import NullLogger
            logger = NullLogger()

        num_steps = kwargs.get("num_steps", getattr(self, "num_steps", 1000))
        num_inner_steps = kwargs.get(
            "num_inner_steps", getattr(self, "num_inner_steps", 1000)
        )

        beta = kwargs.get("admm_weight", 10.0)
        self.L = kwargs.get("L")
        if self.L is None:
            with torch.no_grad():
                self.L = power_iteration(ray_trafo, torch.rand_like(x_in).view(-1, 1))
        self.L2_inv = 1.0 / self.L ** 2
        

        im_size = x_in.shape[-1]
        PSNR = MaskedPSNR(im_size=im_size, mask_fn=create_circular_mask)
        device = x_in.device

        psnr_list, loss_list = [], []
        # Initialize latent variable t and lagrangian
        #TODO: initalise properly at x_in?
        t = torch.zeros((2, *x_in.shape[1:]), device=device, dtype=x_in.dtype).detach()
        lagrangian = torch.zeros((2, *x_in.shape[1:]), device=device, dtype=x_in.dtype).detach()
        reg_weights = self.tv_strength * torch.ones_like(t)
        # t = self.D(x_in, pad=True)
        z = torch.nn.Parameter(x_in.detach().clone().to(device))

        self.model.train()
        for i in (
            pbar := tqdm(
                range(num_steps // num_inner_steps),
                desc="Training ADMM-WTV",
                dynamic_ncols=True,
            )
        ):
            
            # This is the top level
            if num_inner_steps > 1:
                optim_inner = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            for j in range(num_inner_steps):
                global_step = i*num_inner_steps + j
                optim_inner.zero_grad()
                x_pred = self.model(z)
                loss, mse_loss = self.compute_top_loss(x_pred, ray_trafo, y, t, lagrangian, beta=beta)
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

                pbar.set_description(desc)
                logger.log(log_data, step=global_step)
                loss.backward()
                optim_inner.step()

            self.model.eval()
            with torch.no_grad():
                x_pred = self.model(z)
                Dx = self.D(x_pred, pad=True)
                w = Dx + lagrangian/beta
                wnorm = w.norm(p=2, dim=0, keepdim=True)
                t = torch.clamp(1.0-reg_weights/(beta*(wnorm + 1e-8)), min=0.0) * w
                lagrangian += beta * (Dx - t)

                mse_loss = ((ray_trafo.trafo(x_pred) - y).pow(2)).sum() * self.L2_inv
                denoise_loss = (self.D(x_pred, pad=True)).pow(2)* self.tv_strength
                log_data = OrderedDict(
                    [
                        ("loss", mse_loss.item() + denoise_loss.sum().item()),
                        ("mse_loss", mse_loss.item()),
                        ("denoise_loss", denoise_loss.sum().item()),
                    ]
                )
                loss_list.append(log_data["loss"])

                if self.variable_regularisation:
                    den_loss = denoise_loss.sum(dim=0, keepdim=True)
                    reg_weights = torch.clamp(mse_loss / (den_loss+1e-8), min=0.01*self.tv_strength, max=100*self.tv_strength)
                if x_gt is not None:
                    psnr_list.append(PSNR(x_gt, x_pred))
                    log_data["psnr"] = psnr_list[-1]
                logger.log(log_data, step=global_step)

                for cb in self.callbacks:
                    cb(global_step+1, x_pred, loss, mse_loss, psnr_list[-1])


            logger.log_img(
                x_pred, step=global_step, title=f"Step {global_step:05d}" if x_gt is None else f"Step {global_step+1}, PSNR: {psnr_list[-1]:.2f}"
            )

        if logger.use_wandb:
            logger.finish()

        if return_metrics:
            return z, psnr_list, loss_list
        else:
            return z

            # TODO: update full loss? 