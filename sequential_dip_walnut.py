"""
This is an implementation of the aSeqDIP 

Alkhouri et al. "Image Reconstruction Via Autoencoding Sequential Deep Image Prior" (Neurips 2024)
(https://openreview.net/forum?id=K1EG2ABzNE&noteId=KLbtyQ08BC)

"""

import os 

import torch
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio
import matplotlib 
#matplotlib.use("Agg")
import matplotlib.pyplot as plt 
import numpy as np
import random

from tqdm import tqdm

from skimage.metrics import peak_signal_noise_ratio
from PIL import Image
import yaml 
from types import SimpleNamespace


from dataset.walnut import get_walnut_data
from dataset.walnut_2d_ray_trafo import get_walnut_2d_ray_trafo

from model.unet import get_unet_model

cfg_dip = {"model_inp": "adjoint", # "fbp", "adjoint" 
       "model_type": "unet", 
       "channels": (128,128,128,128,128,128), # 6 scales
       "scales": 6,
       "skip": 32,
       "activation": "relu", # "relu", "silu", "leakyrelu
       "padding_mode": "zeros", # "circular", "zeros"
       "upsample_mode": "nearest", # "nearest", "bilinear"
       "use_norm": True,
       "use_sigmoid": True, 
       "random_seed": 1, 
       "lambda_denoise" : 0.1,  # You can tune this value
       "num_steps": 2000,
       "inner_iters": 4,  # Number of update steps before the network input gets updated
       "lr": 1e-4,
       "psnr_tv": 26.88} 

save_dir = "dip_results/sequential_dip"
os.makedirs(save_dir, exist_ok=True)

save_dir_img = "dip_results/sequential_dip/imgs"
os.makedirs(save_dir_img, exist_ok=True)

def create_circular_mask(size):
    H, W = size
    Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    center = (H // 2, W // 2)
    radius = min(center[0], center[1])

    dist = (X - center[1])**2 + (Y - center[0])**2
    mask = (dist <= radius**2)
    return mask  # shape: (501, 501), values: 0 or 1

mask = create_circular_mask((501, 501))


device = "cuda"
torch.manual_seed(cfg_dip["random_seed"])
random.seed(cfg_dip["random_seed"])
np.random.seed(cfg_dip["random_seed"])
torch.cuda.manual_seed_all(cfg_dip["random_seed"])

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
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

device = "cuda"

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    else:
        return d
    
cfg_dict = {}
with open('configs/walnut_config.yaml', 'r') as f:
    data = yaml.safe_load(f)
    cfg_dict["data"] = data

cfg_dip["forward_op"] = data

print(cfg_dip)

with open(os.path.join(save_dir, "cfg.yaml"), "w") as f:
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
Image.fromarray(img).save(os.path.join(save_dir, "groundtruth.png"))


x_test = torch.rand_like(x).view(-1, 1)
print("x_test: ", x_test.shape)
def power_iteration(x0, max_iter=100,verbose=True, tol=1e-6):
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

L = power_iteration(x_test)
print("L: ", L)





print("Number of parameters: ", sum([p.numel() for p in model.parameters()]))
if cfg_dip["model_inp"] == "fbp":
    z = x_fbp.clone().detach()
elif cfg_dip["model_inp"] == "adjoint":  
    z = ray_trafo.trafo_adjoint(y).detach()
    z = z/torch.max(z)
    print(z.min(), z.max())
else:
    raise NotImplementedError

#plt.figure()
#plt.imshow(z[0,0].cpu().numpy())
#plt.show()


print("z_init shape: ", z.shape)
# Make z a trainable parameter

y_noise = y.to(device)

#torch.save(z, "dip_inp.pt")

with torch.no_grad():
    x_pred = model(z)

# Add z to optimizer with a different learning rate
optim_model = torch.optim.Adam(model.parameters(), lr=cfg_dip["lr"])

psnr = peak_signal_noise_ratio(x[0,0,mask].cpu().numpy(), x_pred[0,0,mask].detach().cpu().numpy(), data_range=x[0,0,mask].cpu().numpy().max())
psnr_fbp = peak_signal_noise_ratio(x[0,0,mask].cpu().numpy(), x_fbp[0,0,mask].detach().cpu().numpy(), data_range=x[0,0,mask].cpu().numpy().max())

print(f"Get PSNR = {psnr:.4f}dB at step {0}")
print(f"FBP PSNR = {psnr_fbp:.4f}dB")


psnr_tv = cfg_dip["psnr_tv"]

psnr_list = [] 
loss_list = [] 

best_psnr = 0 
best_psnr_idx = 0 
best_psnr_image = None 

loss_scaling = y_noise.shape[-1]/np.prod(z.shape)
for i in tqdm(range(cfg_dip["num_steps"])):
    
    model.train()
    for _ in range(cfg_dip["inner_iters"]):
        optim_model.zero_grad()

        x_pred = model(z)
        data_loss = torch.sum((ray_trafo.trafo(x_pred) - y_noise)**2/L**2)
        denoise_loss = torch.sum((x_pred - z) ** 2)
        loss = data_loss + cfg_dip["lambda_denoise"] * loss_scaling * denoise_loss
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
        psnr_list.append(peak_signal_noise_ratio(x[0,0,mask].cpu().numpy(), z[0,0,mask].detach().cpu().numpy(), data_range=x[0,0,mask].cpu().numpy().max()))
    
        if psnr_list[-1] > best_psnr:
            best_psnr = psnr_list[-1]
            best_psnr_idx = i 
            best_psnr_image = torch.clone(z.detach().cpu()).numpy()
        
        if (i + 1) == 200:
            psnr_at_1000 = psnr_list[-1]
            reco_at_1000 = torch.clone(z.detach().cpu()).numpy()

            img = reco_at_1000[0,0] * 255
            img = img.astype(np.uint8)
            Image.fromarray(img).save(os.path.join(save_dir, "reco_at_200.png"))

        if (i+1) % 25 == 0:
            print(f"Get PSNR = {psnr_list[-1]:.4f}dB at step {i+1}")
        
            psnr = psnr_list[-1]
            
            
            fig, axes = plt.subplots(2,2, figsize=(13,6))

            axes[0,0].semilogx(np.arange(1, len(loss_list)+1), psnr_list, label="DIP PSNR")
            axes[0,0].set_title("PSNR")
            axes[0,0].hlines(psnr_fbp, 1, len(psnr_list), colors="red", label="FBP PSNR")
            axes[0,0].hlines(psnr_tv, 1, len(psnr_list), colors="green", label="TV PSNR")
            axes[0,0].legend() 

            axes[0,1].loglog(np.arange(1, len(loss_list)+1), loss_list)
            axes[0,1].set_title("Data Consistency ||Ax - y ||^2 + Denoise Loss")

            im = axes[1,0].imshow(z[0,0,:,:].detach().cpu().numpy(), cmap="gray")
            fig.colorbar(im, ax=axes[1,0], fraction=0.08)
            axes[1,0].set_title("Reco")
            axes[1,0].axis("off")

            im = axes[1,1].imshow(x[0,0,:,:].detach().cpu().numpy(), cmap="gray")
            fig.colorbar(im, ax=axes[1,1], fraction=0.08)
            axes[1,1].set_title("GT")
            axes[1,1].axis("off")
 
            plt.savefig(f"{save_dir_img}/DIP_reconstruction_at_step={i+1}.png")
            plt.close()


img = z.detach().cpu().numpy()[0,0] * 255
img = img.astype(np.uint8)
Image.fromarray(img).save(os.path.join(save_dir, "final_reco.png"))

img = best_psnr_image[0,0] * 255
img = img.astype(np.uint8)
Image.fromarray(img).save(os.path.join(save_dir, "best_reco.png"))

results = {} 
results["best_psnr"] = float(best_psnr)
results["best_psnr_idx"] = int(best_psnr_idx)
results["psnr_at_200"] = float(psnr_at_1000)
results["psnr_tv"] = cfg_dip["psnr_tv"]
results["psnr_fbp"] = float(psnr_fbp)

with open(os.path.join(save_dir, "results.yaml"), "w") as f:
    yaml.dump(results, f)

loss_list = np.asarray(loss_list)
np.save(os.path.join(save_dir, "loss.npy"), loss_list)


psnr_list = np.asarray(psnr_list)
np.save(os.path.join(save_dir, "psnrs.npy"), psnr_list)