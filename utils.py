import torch
from types import SimpleNamespace
import logging
from typing import Optional
from pathlib import Path
from datetime import datetime
try:
    import wandb
except ImportError:
    wandb = None

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    else:
        return d
    

def create_circular_mask(size):
    """
    The output of this function is a torch tensor of size (size, size) with binary values:
        1: point is inside a circle of radius size/2
        0: point is outside a circle of radius size/2

    This method is used to only calculate the quality metrics inside of the circle.  
    
    """

    H, W = size
    Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    center = (H // 2, W // 2)
    radius = min(center[0], center[1])

    dist = (X - center[1])**2 + (Y - center[0])**2
    mask = (dist <= radius**2)
    return mask  # shape: (501, 501), values: 0 or 1

def power_iteration(ray_trafo, x0, max_iter=100,verbose=True, tol=1e-6):
    """
    Estimate the Lipschitz constant of the ray_trafo
    
    """
    x = torch.randn_like(x0)
    x /= torch.norm(x)
    zold = torch.zeros_like(x)
    for it in range(max_iter):
        y = ray_trafo.trafo_flat(x)
        y = ray_trafo.trafo_adjoint_flat(y)
        z = torch.matmul(x.conj().reshape(-1), y.reshape(-1)) / torch.norm(x) ** 2

        rel_var = torch.norm(z - zold)
        if rel_var < tol and verbose:
            print(
                f"Power iteration converged at iteration {it}, value={z.item():.2f}"
            )
            break
        zold = z
        x = y / torch.norm(y)
    return z.real


def make_serializable(obj):
    if isinstance(obj, torch.nn.Parameter):
        return obj.detach().cpu().item() if obj.numel() == 1 else obj.detach().cpu().tolist()

    elif isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()

    elif isinstance(obj, SimpleNamespace):
        return {"__namespace__": {k: make_serializable(v) for k, v in vars(obj).items()}}

    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}

    elif isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]

    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj

    else:
        return str(obj)


class FlexibleLogger:
    def __init__(self, use_wandb: bool = False, project: Optional[str] = None, config: Optional[dict] = None, console_printing: bool=True, log_file: Optional[Path]=None):
        self.use_wandb = use_wandb and wandb is not None
        if self.use_wandb:
            wandb.init(project=project, config=config)
        # else:
        if log_file is None:
            Path("logs").mkdir(exist_ok=True)
            log_file = Path("logs") / f"log_{datetime.now():%Y-%m-%d_%H-%M-%S}.log"

        self._init_logging(log_file=log_file, console_printing=console_printing)

    def _init_logging(self, log_file, console_printing=False):
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%d-%m-%Y %H:%M:%S",
            force=True
        )
        self.logger = logging.getLogger()
        if console_printing and not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%d-%m-%Y %H:%M:%S"
            ))
            self.logger.addHandler(console_handler)

    def log(self, data: dict, step: Optional[int] = None):
        if self.use_wandb:
            wandb.log(data, step=step)

        # Fixed width for each field
        field_width = 25

        # Start with step info
        msg = f"Step {step:04d} | " if step is not None else ""

        # Format each key-value pair
        msg += " | ".join(
            f"{k}: {v:.4f}".ljust(field_width) if isinstance(v, float) else f"{k}: {v}".ljust(field_width)
            for k, v in data.items()
        )

        self.logger.info(msg)
    def log_img(self, img, step):
        if not self.use_wandb:
            return
        if isinstance(img, torch.Tensor):
           tensor = img.detach().cpu()
        else:
           tensor = torch.from_numpy(img).cpu()

        if tensor.ndim == 4:
            tensor = tensor[0]

        if tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        else:
            tensor = tensor.permute(1, 2, 0)
        wandb.log({"reconstruction": wandb.Image(tensor.numpy())}, step=step)
       
    def log_dict(self, data:dict, message:str=None):
        if message is not None:
            self.logger.info(f"{"="*30}{message}{"="*30}")
        max_key_len = max(len(str(k)) for k in data.keys())
        for key, value in data.items():
            if isinstance(value, torch.nn.Parameter) or isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    val_str = f"{value.item():.6f}"
                else:
                    val_str = str(value.detach().cpu().numpy())
            else:
                val_str = str(value).replace("\n", " ").replace("  ", " ")
            self.logger.info(f"    {key:<{max_key_len}} : {val_str}")
        if message is not None:
            self.logger.info(f"{"="*30}{"="*len(message)}{"="*30}")

    def info(self, message: str):
        self.logger.info(message)

    def finish(self):
        if self.use_wandb:
            wandb.finish()

from skimage.metrics import peak_signal_noise_ratio

class MaskedPSNR:
    def __init__(self, im_size, mask_fn=None):
        """
        Args:
            im_size (int): The image size (assumes square images).
            mask_fn (callable): A function that returns a boolean mask array.
        """
        self.im_size = im_size
        if mask_fn is None:
            mask_fn = create_circular_mask
        self.mask = mask_fn((im_size, im_size))

    def __call__(self, x, x_pred):
        """
        Args:
            x (torch.Tensor): Ground truth image tensor, shape (B, C, H, W)
            x_pred (torch.Tensor): Predicted image tensor, same shape as x

        Returns:
            float: PSNR computed only within the masked region
        """
        # Use the first image and first channel (assumes grayscale)
        x_masked = x[0, 0, self.mask].detach().cpu().numpy()
        x_pred_masked = x_pred[0, 0, self.mask].detach().cpu().numpy()

        data_range = x_masked.max()  # Can also use x_masked.max() - x_masked.min() if needed
        return peak_signal_noise_ratio(x_masked, x_pred_masked, data_range=data_range)
