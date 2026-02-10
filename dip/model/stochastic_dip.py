import torch 
from tqdm import tqdm
from .base_dip import BaseDeepImagePrior
from .utils import MaskedPSNR
from ..logging import FlexibleLogger
from ..physics import power_iteration


class StochasticDeepImagePrior(BaseDeepImagePrior):
    """
    Deep Image Prior with Stochastic Gradient Descent.
    
    Uses only a subset of rows from the forward operator (ray_trafo.trafo)
    and corresponding measurements in each iteration, reducing computational cost.
    """
    
    def __init__(self, model, lr, num_steps, noise_std, batch_size=None, 
                 callbacks=None, save_dir=None):
        """
        Args:
            model: Neural network model for DIP
            lr: Learning rate
            num_steps: Number of training steps
            noise_std: Standard deviation of noise added to input
            batch_size: Number of rows to sample from forward operator per iteration.
                       If None, all rows are used (equivalent to vanilla DIP).
            callbacks: List of callback functions
            save_dir: Directory to save results
        """
        super().__init__(model, lr, num_steps, noise_std, callbacks, save_dir)
        self.batch_size = batch_size

    def compute_loss(self, x_pred, ray_trafo, y, indices=None, **kwargs):
        """
        Compute stochastic loss using only specified rows of the forward operator.
        
        Args:
            x_pred: Predicted image
            ray_trafo: Forward operator with trafo attribute
            y: Measurement data
            indices: Indices of rows to use. If None, use all rows.
            **kwargs: Additional arguments (e.g., L for Lipschitz constant)
        
        Returns:
            loss: Stochastic loss (scaled to full loss)
            mse_loss: Mean squared error loss
        """
        L = kwargs.get("L", 1.0)
        
        # Apply forward operator
        x_pred_trafo = ray_trafo.trafo(x_pred)
        # Use subset of measurements if indices provided
        if indices is not None:
            x_pred_trafo = x_pred_trafo[..., indices]
            y_subset = y[..., indices]
        else:
            y_subset = y
        # Compute loss
        residual = x_pred_trafo - y_subset
        loss = torch.sum(residual**2 / L**2)
        
        # Scale loss to represent full data loss
        if indices is not None and len(indices) > 0:
            # Scale by ratio of full data to subset
            total_rows = y.shape[0] if len(y.shape) > 0 else 1
            subset_rows = len(indices)
            loss = loss * (total_rows / subset_rows)
        
        mse_loss = loss
        return loss, mse_loss

    def train(self, ray_trafo, y, x_in, x_gt=None, return_metrics=True, **kwargs):
        """
        Train the DIP model using stochastic gradient descent.
        
        Args:
            ray_trafo: Forward operator with trafo attribute
            y: Measurement data (1D array or 2D array)
            x_in: Initial input
            x_gt: Ground truth image (optional, for PSNR computation)
            return_metrics: Whether to return metrics
            **kwargs: Additional arguments
        
        Returns:
            x_out: Reconstructed image
            psnr_list: List of PSNR values per step
            loss_list: List of loss values per step
        """
        logger = FlexibleLogger(use_wandb=False)
        optim = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        psnr_list, loss_list = [], []
        
        with torch.no_grad():
            L = power_iteration(ray_trafo, torch.rand_like(x_in).view(-1, 1))

        psnr_fun = MaskedPSNR(x_in.shape[2])
        
        # Determine total number of rows
        total_rows = y.shape[-1] 
        batch_size =  self.batch_size if self.batch_size is not None else total_rows
        self.model.train()
        for i in tqdm(range(self.num_steps)):
            optim.zero_grad()
            noise = self.noise_std**2 * torch.randn_like(x_in) if self.noise_std > 0 else 0
            x_pred = self.model(x_in + noise)
            
            # Sample random indices for stochastic update
            indices = torch.randperm(total_rows)[:batch_size]
            # TODO: The loss still evaluated the full forward operator, 
            # and only subsamples according to the indices. This means that the computational cost 
            # is not reduced at the moment, but we have the same convergence behavior as SGD.
            loss, mse_loss = self.compute_loss(x_pred, ray_trafo, y, 
                                               indices=indices, L=L)
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
