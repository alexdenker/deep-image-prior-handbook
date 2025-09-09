import yaml 


import os 

import torch
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio
import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt 
import numpy as np
import random

from tqdm import tqdm

from PIL import Image
import yaml 

from dataset.walnut import get_walnut_data
from dataset.walnut_2d_ray_trafo import get_walnut_2d_ray_trafo

from model.unet import get_unet_model
from utils import dict_to_namespace, create_circular_mask, power_iteration
from dip import DeepImagePrior

import argparse 

parser = argparse.ArgumentParser(description="Run DIP")

parser.add_argument("--method",
                    type=str,
                    default="vanilla")

parser.add_argument("--model_inp", 
                    type=str, 
                    default="fbp", 
                    choices=["fbp", "random"],
                    help="Input to the DIP")

parser.add_argument("--num_steps", 
                    type=int,
                    default=10000)

parser.add_argument("--lr", 
                    type=float,
                    default=2e-4)

parser.add_argument("--noise_std", 
                    type=float,
                    default=0.0,
                    help="adding additional noise to the DIP input")

parser.add_argument("--random_seed", 
                    type=int, 
                    default=1)

parser.add_argument("--random_seed_noise", 
                    type=int, 
                    default=2,
                    help="Random seed for the initial input of the DIP (only used if model_inp = random)")

parser.add_argument("--device",
                    type=str,
                    default="cuda")

parser.add_argument("--only_fitting", type=str, default=False)
args = parser.parse_args()



save_dir = f"dip_results/{args.method}/{args.model_inp}"
os.makedirs(save_dir, exist_ok=True)

save_dir_img = f"dip_results/{args.method}/{args.model_inp}/imgs"
os.makedirs(save_dir_img, exist_ok=True)


device = args.device
torch.manual_seed(args.random_seed)
random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


model_dict = {}
with open('configs/dip_architecture.yaml', 'r') as f:
    model_dict = yaml.safe_load(f)

print(model_dict)

model = get_unet_model(in_ch=1, out_ch=1, scales=model_dict.scales,
                        skip=model_dict.skip,
                        channels=model_dict.channels,
                        use_sigmoid=model_dict.use_sigmoid,
                        use_norm=model_dict.use_norm,
                        activation=model_dict.activation,
                        padding_mode=model_dict.padding_mode,
                        upsample_mode=model_dict.upsample_mode)   
model.to(device)
model.train()

    
cfg_dict = {}
with open('configs/walnut_config.yaml', 'r') as f:
    data = yaml.safe_load(f)
    cfg_dict["data"] = data

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

img = x[0,0].cpu().numpy() * 255
img = img.astype(np.uint8)
Image.fromarray(img).save(os.path.join(save_dir, "groundtruth.png"))


x_test = torch.rand_like(x).view(-1, 1)
L = power_iteration(ray_trafo, x_test)
print("L: ", L)


print("Number of parameters: ", sum([p.numel() for p in model.parameters()]))
if args.model_inp == "fbp":
    z = x_fbp
else:
    g = torch.Generator()
    g.manual_seed(args.random_seed_noise)
    z = 0.1*torch.randn(x.shape, generator=g)

noise_std = 0.0

z = z.to(device)
y_noise = y.to(device)

if args.method == "vanilla":
    dip = DeepImagePrior(model=model, 
                         lr=args.lr, 
                         num_steps=args.num_steps, 
                         noise_std=args.noise_std, 
                         L=L)
    
    x_pred, psnr_list, loss_list, best_psnr_image, best_psnr_idx = dip.train(ray_trafo, y, z)
else:
    raise NotImplementedError


best_psnr = max(psnr_list)

img = x_pred.detach().cpu().numpy()[0,0] * 255
img = img.astype(np.uint8)
Image.fromarray(img).save(os.path.join(save_dir, "final_reco.png"))

img = best_psnr_image[0,0] * 255
img = img.astype(np.uint8)
Image.fromarray(img).save(os.path.join(save_dir, "best_reco.png"))

results = {} 
results["best_psnr"] = float(best_psnr)
results["best_psnr_idx"] = int(best_psnr_idx)

with open(os.path.join(save_dir, "results.yaml"), "w") as f:
    yaml.dump(results, f)

loss_list = np.asarray(loss_list)
np.save(os.path.join(save_dir, "loss.npy"), loss_list)


psnr_list = np.asarray(psnr_list)
np.save(os.path.join(save_dir, "psnrs.npy"), psnr_list)