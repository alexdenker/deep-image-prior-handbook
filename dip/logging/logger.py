import torch
import logging
from typing import Optional
from pathlib import Path
import os
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import numpy as np
try:
    import wandb
except ImportError:
    wandb = None
import matplotlib.pyplot as plt

class NullLogger:
    def __init__(self, console_printing=False, image_path: Optional[Path]=None):
        self.console_printing = console_printing
        self.image_logging = 200
        self.image_path = image_path

    def log(self, data, step=None):
        if not self.console_printing:
            return

        formatted_data = []
        for k, v in data.items():
            if isinstance(v, (float, np.floating)):
                formatted_data.append(f"{k}: {v:.4f}")
            else:
                formatted_data.append(f"{k}: {v}")                
        desc = f"Step {step} | " + " | ".join(formatted_data)
        print(desc)

    def log_img(self, img, step=None, title=None):
        if self.image_logging is None or (step+1) % self.image_logging != 0:
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

        np_img = tensor.numpy()
        if np.issubdtype(np_img.dtype, np.floating):
            np_img = np.clip(np_img, 0.0, 1.0)
            np_img = (np_img * 255).astype(np.uint8)
        else:
            np_img = np.clip(np_img, 0, 255).astype(np.uint8)
            
        img_pil = Image.fromarray(np_img)
        if self.image_path is None:
            plt.imshow(img_pil, cmap="gray" if tensor.ndim == 2 else None)
            plt.title(title if title is not None else f"Step {step}")
            plt.axis("off")
            plt.show()
        else:
            save_dir = Path(self.image_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"reconstruction_{step:04d}.png"
            img_pil.save(save_path)

class FlexibleLogger:
    def __init__(self, use_wandb: bool = False, project: Optional[str] = None, wandb_config: Optional[dict] = None, console_printing: bool=True, log_file: Optional[Path]=None, image_logging: Optional[int]=None, image_path: Optional[Path]=None):
        self.use_wandb = use_wandb and wandb is not None
        if self.use_wandb:
            wandb.init(project=project, config=wandb_config)
            wandb.run.name = wandb_config["name"] if "name" in wandb_config else f"run_{datetime.now():%Y-%m-%d_%H-%M-%S}"
        if log_file is None:
            Path("logs").mkdir(exist_ok=True)
            log_file = Path("logs") / f"log_{datetime.now():%Y-%m-%d_%H-%M-%S}.log"
        self.image_logging = image_logging
        self.image_path = image_path
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

        field_width = 25
        msg = f"Step {step:04d} | " if step is not None else ""
        msg += " | ".join(
            f"{k}: {v:.4f}".ljust(field_width) if isinstance(v, float) else f"{k}: {v}".ljust(field_width)
            for k, v in data.items()
        )

        self.logger.info(msg)
    def log_img(self, img, step, title: Optional[str]=None):
        if self.image_logging is None or step % self.image_logging != 0:
            return
        if not self.use_wandb and self.image_path is None:
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
        np_img = tensor.numpy()
        if self.use_wandb:
            wandb.log({"reconstruction": wandb.Image(np_img)}, step=step)
        if self.image_path is not None:
            save_path = os.path.join(self.image_path, f"reconstruction_{step:04d}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            img_pil = Image.fromarray((np_img * 255).astype("uint8") if np_img.max() <= 1 else np_img.astype("uint8"))
            if title is not None:
                width, height = img_pil.size
                padding_height = 50  # space for title + separator line
                new_height = height + padding_height

                new_img = Image.new("RGB", (width, new_height), color=(0, 0, 0))
                draw = ImageDraw.Draw(new_img)
                line_y = padding_height - 5  # 5 pixels above where image starts
                draw.line([(0, line_y), (width, line_y)], fill=(255, 255, 255), width=2)
                new_img.paste(img_pil, (0, padding_height))
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=30)
                except IOError:
                    font = ImageFont.load_default()

                draw.text((10, 10), title, font=font, fill=(255, 255, 255))

                img_pil = new_img
            img_pil.save(save_path)

    def log_dict(self, data:dict, message:str=None):
        if message is not None:
            self.logger.info(f"{'='*30}{message}{'='*30}")
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
            self.logger.info(f"{'='*30}{'='*len(message)}{'='*30}")

    def info(self, message: str):
        self.logger.info(message)

    def finish(self):
        if self.use_wandb:
            wandb.finish()