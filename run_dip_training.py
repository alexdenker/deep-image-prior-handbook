import argparse 
import numpy as np
import os 
import random
import torch
import yaml 

from datetime import datetime
from dip import get_unet_model, get_walnut_2d_ray_trafo, dict_to_namespace
from dip.dataset import get_disk_dist_ellipses_dataset
from dip.trainer import ParameterSampler

parser = argparse.ArgumentParser(description="Run DIP")

parser.add_argument("--method",
                    type=str,
                    default="vanilla", 
                    choices=["vanilla", "tv_hqs", "tv", "aseq", "selfguided"],
                    help="DIP method to use")

parser.add_argument("--model_inp", 
                    type=str, 
                    default="fbp", 
                    choices=["fbp", "random", "adjoint"],
                    help="Input to the DIP")

parser.add_argument("--phantom",
                    type=str,
                    default="walnut",
                    choices=["walnut", "shepplogan"],
                    help="phantom to test the DIP")

parser.add_argument("--num_steps", 
                    type=int,
                    default=10000)

parser.add_argument("--lr", 
                    type=float,
                    default=1e-4)

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

parser.add_argument("--use_wandb",
                    type=bool,
                    default=False,
                    help="Whether to use wandb logging or not")

base_args, remaining = parser.parse_known_args()

if base_args.method == "tv_hqs":
    parser.add_argument("--splitting_strength", 
                        type=float, 
                        default=60)
    parser.add_argument("--tv_min", 
                        type=float, 
                        default=1e-2)
    parser.add_argument("--tv_max", 
                        type=float,
                        default=0.5)
    parser.add_argument("--inner_steps", 
                        type=int, 
                        default=20)

elif base_args.method == "tv":
    parser.add_argument("--tv_strength", 
                        type=float,
                        default=1e-5)
elif base_args.method == "aseq":
    parser.add_argument("--denoise_strength",
                        type=float,
                        default=0.01,
                        help="Denoising strength for the denoising prior")
    parser.add_argument("--num_inner_steps",
                        type=int,
                        default=5,
                        help="Number of inner optimisation steps for the aseq DIP")
elif base_args.method == "selfguided":
    parser.add_argument("--denoise_strength",
                        type=float,
                        default=0.01,
                        help="Denoising strength for the denoising prior")
    parser.add_argument("--num_noise_realisations",
                        type=int,
                        default=4,
                        help="Number of noise realisations for the self-guided DIP")
    parser.add_argument("--exp_weight",
                        type=float,
                        default=0.99,
                        help="Weight for the exponential averaging of the output")
else:
    pass 

args = parser.parse_args()


time = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = f"dip_results/{args.phantom}/{args.method}/{args.model_inp}/{time}"
os.makedirs(save_dir, exist_ok=True)

save_dir_img = f"dip_results/{args.phantom}/{args.method}/{args.model_inp}/{time}/imgs"
os.makedirs(save_dir_img, exist_ok=True)

device = args.device
torch.manual_seed(args.random_seed)
random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed)

model_dict = {}
with open('configs/dip_architecture.yaml', 'r') as f:
    model_dict = yaml.safe_load(f)
    model_dict = dict_to_namespace(model_dict)
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

dataset = {
    "train": get_disk_dist_ellipses_dataset(
        fold="train",
        ray_trafo=ray_trafo,
        im_size=cfg.data.im_size,
        device=device,
    ),
    "validation": get_disk_dist_ellipses_dataset(
        fold="validation",
        ray_trafo=ray_trafo,
        im_size=cfg.data.im_size,
        device=device,
    )
}

sampler = ParameterSampler(model=model, dataset=dataset)
sampler.sample(optim_kwargs={
    "torch_manual_seed": 42,
    "batch_size": 4,
    "epochs": 50,
    "num_samples": 100,
    "sampling_strategy": "linear",
    "burn_in": 100,
    "optimizer": {
        "lr": 1e-3,
        "weight_decay": 0
    }, 
    "scheduler":{
        "name": "cosine",
        "lr_min": 1e-5,       
    }, 
    "save_best_learned_params_path": "./parameters/", 
    "save_best_learned_params_per_epoch": True,
}, 
save_samples=True,
)