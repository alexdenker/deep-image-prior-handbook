"""
This is an implementation of the aSeqDIP 

Alkhouri et al. "Image Reconstruction Via Autoencoding Sequential Deep Image Prior" (Neurips 2024)
(https://openreview.net/forum?id=K1EG2ABzNE&noteId=KLbtyQ08BC)

"""
import os 
import torch
import random
import yaml 
import argparse
import matplotlib 
import wandb 

from tqdm import tqdm
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
import matplotlib.pyplot as plt 
from pathlib import Path
from datetime import datetime

from PIL import Image
from pprint import pprint

from dataset.walnut import get_walnut_data
from dataset.walnut_2d_ray_trafo import get_walnut_2d_ray_trafo
from model.unet import get_unet_model
from utils import create_circular_mask, dict_to_namespace, power_iteration, FlexibleLogger

p = argparse.ArgumentParser("Sequential DIP")
p.add_argument("--model_input", choices=["adjoint","fbp"], default="adjoint")
p.add_argument("--model_type", default="unet")
p.add_argument("--channels", nargs="+", type=int, default=[128,128,128,128,128,128],
help="space-separated ints, e.g. --channels 64 64 64")
p.add_argument("--scales", type=int, default=6)
p.add_argument("--skip", type=int, default=32)
p.add_argument("--activation", choices=["relu","silu","leakyrelu"], default="relu")
p.add_argument("--padding_mode", choices=["zeros","circular"], default="zeros")
p.add_argument("--upsample_mode", choices=["nearest","bilinear"], default="nearest")
p.add_argument("--use_norm", action="store_true")
p.add_argument("--no-norm", dest="use_norm", action="store_false")
p.set_defaults(use_norm=True)
p.add_argument("--use_sigmoid", action="store_true")
p.add_argument("--no-sigmoid", dest="use_sigmoid", action="store_false")
p.set_defaults(use_sigmoid=True)
p.add_argument("--random_seed", type=int, default=1)
p.add_argument("--lambda_denoise", type=float, default=0.01)
p.add_argument("--num_steps", type=int, default=2000)
p.add_argument("--inner_iters", type=int, default=10)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--psnr_tv", type=float, default=26.88)
p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

args = p.parse_args()
cfg_dip = vars(args)
cfg_dip["channels"] = tuple(cfg_dip["channels"])
cfg_dip["optimiser"] = "adam"

device = cfg_dip['device']
torch.manual_seed(cfg_dip["random_seed"])
random.seed(cfg_dip["random_seed"])
np.random.seed(cfg_dip["random_seed"])
torch.cuda.manual_seed_all(cfg_dip["random_seed"])

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

if cfg_dip["model_type"] == "unet":
    model = get_unet_model(in_ch=1, out_ch=1, scales=cfg_dip["scales"],
                           skip=cfg_dip["skip"],
                           channels=cfg_dip["channels"],
                           use_sigmoid=cfg_dip["use_sigmoid"],
                           use_norm=cfg_dip["use_norm"],
                           activation=cfg_dip["activation"],
                           padding_mode=cfg_dip["padding_mode"],
                           upsample_mode=cfg_dip["upsample_mode"])        
    model.to(device)
    model.train()
else:
    raise NotImplementedError

    
cfg_dict = {}
with open('configs/walnut_config.yaml', 'r') as f:
    data = yaml.safe_load(f)
    cfg_dict["data"] = data

cfg_dip["forward_operator"] = data

# pprint(cfg_dip)

path_name = f"aseqdip_{cfg_dip["inner_iters"]}inner_{cfg_dip["lambda_denoise"]}lmbd_{cfg_dip["lr"]}lr"
base_path = Path(f"dip_results/{path_name}")
paths = {
    "weights": base_path / "weights",
    "imgs": base_path / "imgs"
}
for p in paths.values():
    p.mkdir(parents=True, exist_ok=True)

wandb_project = path_name
# if cfg_dip["optimiser"] == "gd" and cfg_dip["momentum"] > 0:
#     wandb_project = wandb_project + "+momentum"

# wandb_project = wandb_project + f"_tv={cfg_dip["tv_reg"]}"

wandb_cfg = {
        "project": "deep-image-prior-handbook",
        "entity": "zkereta",
        "config": cfg_dip,
        "name": wandb_project,
        "mode": "online", #"disabled", #"online" ,
        "settings": wandb.Settings(code_dir="wandb"),
        "dir": "wandb",
    }


logger = FlexibleLogger(use_wandb=True, project=wandb_project, config=wandb_cfg, log_file = base_path / f"log_{datetime.now():%Y-%m-%d_%H-%M-%S}.log", console_printing=False)
with open(base_path/"cfg.yaml", "w") as f:
    yaml.dump(cfg_dip, f)

cfg = dict_to_namespace(cfg_dict)
cfg.device = device

ray_trafo = get_walnut_2d_ray_trafo(
    data_path=cfg.data.data_path,
    matrix_path=cfg.data.data_path,
    walnut_id=cfg.data.walnut_id,
    orbit_id=cfg.data.orbit_id,
    angular_sub_sampling=cfg.data.angular_sub_sampling,
    proj_col_sub_sampling=cfg.data.proj_col_sub_sampling)

ray_trafo.to(device)
data = get_walnut_data(cfg, ray_trafo=ray_trafo)

y, x, x_fbp = data[0]
#y = y[0,0,0,:].unsqueeze(-1)
im_size = x.shape[-1]
print(y.shape, x.shape, x_fbp.shape)

img = x[0,0].cpu().numpy() * 255
img = img.astype(np.uint8)
Image.fromarray(img).save(paths['imgs'] /"groundtruth.png")

x_test = torch.rand_like(x).view(-1, 1)
L = power_iteration(ray_trafo, x_test, max_iter=100, verbose=True, tol=1e-6)

logger.info(f"L: {L}. number of parameters: {sum([p.numel() for p in model.parameters()])}")
if cfg_dip["model_input"] == "fbp":
    z = x_fbp.clone().detach()
elif cfg_dip["model_input"] == "adjoint":  
    z = ray_trafo.trafo_adjoint(y).detach()
    z = z/torch.max(z)
else:
    raise NotImplementedError

# print("z_init shape: ", z.shape)
# Make z a trainable parameter

y_noise = y.to(device)

#torch.save(z, "dip_inp.pt")

with torch.no_grad():
    x_pred = model(z)

# Add z to optimizer with a different learning rate
optim_model = torch.optim.Adam(model.parameters(), lr=cfg_dip["lr"])
#img_size = int(cfg_dip['forward_operator']['im_size'])
from utils import MaskedPSNR
PSNR = MaskedPSNR(im_size=im_size, mask_fn=create_circular_mask)
psnr_fbp = PSNR(x, x_fbp)
logger.info(f"Step 0 PSNR:{PSNR(x, x_pred):.4f}, FBP :{psnr_fbp:.4f}")

mask = create_circular_mask((im_size, im_size))
#psnr = peak_signal_noise_ratio(x[0,0,mask].cpu().numpy(), x_pred[0,0,mask].detach().cpu().numpy(), data_range=x[0,0,mask].cpu().numpy().max())
#psnr_fbp = peak_signal_noise_ratio(x[0,0,mask].cpu().numpy(), x_fbp[0,0,mask].detach().cpu().numpy(), data_range=x[0,0,mask].cpu().numpy().max())

#print(f"Get PSNR = {psnr:.4f}dB at step {0}")
#print(f"FBP PSNR = {psnr_fbp:.4f}dB")


psnr_tv = cfg_dip["psnr_tv"]

psnr_list = [] 
loss_list = [] 

best_psnr = 0 
best_psnr_idx = 0 
best_psnr_image = None 

plot_storage_epoch = 25
img_storage_epoch = 100
loss_scaling = y_noise.shape[-1]/np.prod(z.shape)

for i in (pbar:=tqdm(range(cfg_dip["num_steps"]), desc="Training", dynamic_ncols=True)):
    #tqdm(range(cfg_dip["num_steps"])):
    
    model.train()
    for inner_step in range(cfg_dip["inner_iters"]):
        global_step = i * cfg_dip["inner_iters"] + inner_step

        optim_model.zero_grad()

        x_pred = model(z)
        data_loss = ((ray_trafo.trafo(x_pred) - y_noise).pow(2)).sum() / (L * L)
        denoise_loss = (x_pred - z).pow(2).sum()
        #data_loss = torch.sum((ray_trafo.trafo(x_pred) - y_noise)**2/L**2)
        #denoise_loss = torch.sum((x_pred - z) ** 2)
        loss = data_loss + cfg_dip["lambda_denoise"] * loss_scaling * denoise_loss
        log_data = {
            "data_loss": data_loss.item(),
            "denoise_loss": denoise_loss.item(),
            "total_loss": loss.item()
        }

        pbar.set_description(
            f"{i}:{inner_step:04d} | data: {data_loss.item():.4f} | denoise: {denoise_loss.item():.4f} | loss: {loss.item():.4f}"
        )
        logger.log(log_data, step=global_step)

        #message = f"Step {i:04d} | data: {data_loss.item():.4f}, denoise: {denoise_loss.item():.4f}, total loss: {loss.item():.4f}"
        #pbar.set_description(message)
        #logger.info(message)
        #print("data_loss: ", data_loss.item(), "denoise_loss: ", denoise_loss.item(), "loss: ", loss.item())
        loss.backward()

        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim_model.step() 

    model.eval()
    with torch.no_grad():
        z_pred = model(z)
        z = torch.clone(z_pred)

    loss_list.append(data_loss.item())
    with torch.no_grad():
        psnr_value = PSNR(x, z)
        psnr_list.append(psnr_value)
        logger.log({"psnr": psnr_value}, step=global_step)

        #psnr_list.append(peak_signal_noise_ratio(x[0,0,mask].cpu().numpy(), z[0,0,mask].detach().cpu().numpy(), data_range=x[0,0,mask].cpu().numpy().max()))
    
        if psnr_list[-1] > best_psnr:
            best_psnr = psnr_list[-1]
            best_psnr_idx = i 
            best_psnr_image = torch.clone(z.detach().cpu()).numpy()
        
        if (i + 1)%img_storage_epoch:
            psnr_at = psnr_list[-1]
            reco_at = torch.clone(z.detach().cpu()).numpy()

            img = reco_at[0,0] * 255
            img = img.astype(np.uint8)
            Image.fromarray(img).save(paths["imgs"]/f"recon_{i+1}.png")
            logger.log_img(reco_at, step=i+1)

        if (i+1) % plot_storage_epoch == 0:
            print(f"Get PSNR = {psnr_list[-1]:.4f}dB at step {i+1}")
        
            psnr = psnr_list[-1]
            
            
            fig, axes = plt.subplots(2,2, figsize=(13,6))

            axes[0,0].semilogx(np.arange(1, len(loss_list)+1), psnr_list, label="DIP PSNR")
            axes[0,0].set_title("PSNR")
            axes[0,0].hlines(psnr_fbp, 1, len(psnr_list), colors="red", label="FBP PSNR")
            axes[0,0].hlines(psnr_tv, 1, len(psnr_list), colors="green", label="TV PSNR")
            axes[0,0].legend() 

            axes[0,1].loglog(np.arange(1, len(loss_list)+1), loss_list)
            axes[0, 1].set_title(r"$\|Ax - y\|^2 + \lambda \|x - z\|^2$")
            #axes[0,1].set_title("Data Consistency ||Ax - y ||^2 + Denoise Loss")

            im = axes[1,0].imshow(z[0,0,:,:].detach().cpu().numpy(), cmap="gray")
            fig.colorbar(im, ax=axes[1,0], fraction=0.08)
            axes[1,0].set_title(f"Reconstruction iteration {i}")
            axes[1,0].axis("off")

            im = axes[1,1].imshow(x[0,0,:,:].detach().cpu().numpy(), cmap="gray")
            fig.colorbar(im, ax=axes[1,1], fraction=0.08)
            axes[1,1].set_title("Ground Truth")
            axes[1,1].axis("off")
 
            plt.savefig(paths['imgs']/f"figs_at_{i+1}.png")
            plt.close()


if logger.use_wandb:
    logger.finish()
img = z.detach().cpu().numpy()[0,0] * 255
img = img.astype(np.uint8)
Image.fromarray(img).save(paths["imgs"]/"final_recon.png")

img = best_psnr_image[0,0] * 255
img = img.astype(np.uint8)
Image.fromarray(img).save(paths["imgs"]/"best_recon.png")

results = {} 
results["best_psnr"] = float(best_psnr)
results["best_psnr_idx"] = int(best_psnr_idx)
results["psnr_tv"] = cfg_dip["psnr_tv"]
results["psnr_fbp"] = float(psnr_fbp)

with open(base_path/ "results.yaml", "w") as f:
    yaml.dump(results, f)

loss_list = np.asarray(loss_list)
np.save(base_path/ "loss.npy", loss_list)


psnr_list = np.asarray(psnr_list)
np.save(base_path/ "psnrs.npy", psnr_list)

