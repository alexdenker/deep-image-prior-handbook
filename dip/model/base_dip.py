import torch 
from tqdm import tqdm 

from .utils import MaskedPSNR
from ..logging import FlexibleLogger
from ..physics import power_iteration

class BaseDeepImagePrior():
    def __init__(self, model, lr, num_steps, noise_std, callbacks=None):
        self.model = model 

        self.lr = lr
        self.num_steps = num_steps
        self.noise_std = noise_std # add additional random noise to input 

        self.callbacks = callbacks if callbacks is not None else []

    def compute_loss(self, x_pred, ray_trafo, y, **kwargs):
        """
        Override this method in subclasses to implement custom loss.
        """
        raise NotImplementedError

    def train(self, ray_trafo, y, x_in, x_gt=None, return_metrics=True, **kwargs):
        logger = FlexibleLogger(use_wandb=False)
        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        psnr_list, loss_list = [], []
        with torch.no_grad():
            L = power_iteration(ray_trafo, torch.rand_like(x_in).view(-1, 1))

        psnr_fun = MaskedPSNR(x_in.shape[2])

        self.model.train()
        for i in tqdm(range(self.num_steps)):
            optim.zero_grad()
            noise = self.noise_std**2 * torch.randn_like(x_in) if self.noise_std > 0 else 0
            x_pred = self.model(x_in + noise)
            loss, mse_loss = self.compute_loss(x_pred, ray_trafo, y, L=L)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optim.step()
            logger.log({"MSE": mse_loss.item()}, step=i)
            loss_list.append(mse_loss.item())

            if x_gt is not None:
                psnr = psnr_fun(x_gt, x_pred)
                psnr_list.append(psnr)
            else:
                psnr_list.append(0)

            for cb in self.callbacks:
                cb(i, x_pred, loss, mse_loss, psnr_list[-1])

        self.model.eval()
        with torch.no_grad():
            x_out = self.model(x_in)
        logger.finish()

        if return_metrics:
            return x_out, psnr_list, loss_list
        else:
            return x_out
