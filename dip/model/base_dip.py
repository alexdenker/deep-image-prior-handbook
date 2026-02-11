from typing_extensions import OrderedDict
import torch 
from tqdm import tqdm 

from .utils import MaskedPSNR, create_circular_mask
from ..physics import power_iteration

class BaseDeepImagePrior():
    def __init__(self, model, lr, num_steps, noise_std, callbacks=None, save_dir=None):
        self.model = model 

        self.lr = lr
        self.num_steps = num_steps
        self.noise_std = noise_std # add additional random noise to input 

        self.callbacks = callbacks if callbacks is not None else []
        self.save_dir = save_dir

    def compute_loss(self, x_pred, ray_trafo, y, **kwargs):
        """
        Override this method in subclasses to implement custom loss.
        """
        raise NotImplementedError

    def train(self, ray_trafo, y, x_in, x_gt=None, return_metrics=True, logger=None,**kwargs):
        if logger is None:
            from ..logging import NullLogger
            logger = NullLogger()
        
        num_steps = kwargs.get("num_steps", getattr(self, "num_steps", 1000))
        
        self.L = kwargs.get("L")
        if self.L is None:
            with torch.no_grad():
                L = power_iteration(ray_trafo, torch.rand_like(x_in).view(-1, 1))

        im_size = x_in.shape[-1]
        psnr_fun = MaskedPSNR(im_size=im_size, mask_fn=create_circular_mask)
        psnr_list, loss_list = [], []

        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        for i in (pbar:=tqdm(range(num_steps), desc = "DIP", dynamic_ncols=True)):
            optim.zero_grad()

            noise = self.noise_std**2 * torch.randn_like(x_in) if self.noise_std > 0 else 0
            x_pred = self.model(x_in + noise)
            
            loss, mse_loss = self.compute_loss(x_pred, ray_trafo, y, L=L)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optim.step()

            if x_gt is not None:
                psnr = psnr_fun(x_gt, x_pred)
                psnr_list.append(psnr)
            else:
                psnr_list.append(0)

            log_data = OrderedDict(
                [
                    ("loss", loss.item()),
                    ("mse_loss", mse_loss.item()),
                    ("psnr", psnr_list[-1]),
                ]
            )

            logger.log(log_data, step=i)
            loss_list.append(mse_loss.item())
            logger.log_img(
                x_pred,
                step=i,
                title=f"Step {i:05d}"
                if x_gt is None
                else f"Step {i+1}, PSNR: {psnr_list[-1]:.2f}",
            )
            desc_parts = [f"{i:04d}"]
            desc_parts += [f"{k}: {v:.4f}" for k, v in log_data.items()]
            if psnr_list:
                desc_parts.append(f"PSNR: {psnr_list[-1]:.2f}")
            desc = " | ".join(desc_parts)
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

