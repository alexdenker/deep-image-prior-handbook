import yaml 
import os 
import torch
import numpy as np
import random
from PIL import Image
import argparse 
from datetime import datetime
import wandb
import time 
from dip import (DeepImagePrior, DeepImagePriorHQS, DeepImagePriorTV, AutoEncodingSequentialDeepImagePrior, 
                 SelfGuidanceDeepImagePrior,
                get_unet_model, get_walnut_data, get_walnut_2d_ray_trafo, 
                dict_to_namespace,
                track_best_psnr_output, save_images, early_stopping)
from configs.wandb_config import WANDB_PROJECT, WANDB_ENTITY


parser = argparse.ArgumentParser(description="Run DIP")

parser.add_argument("--method",
                    type=str,
                    default="vanilla", 
                    choices=["vanilla", "tv_hqs", "tv", "aseq", "selfguided", "edip", "edip_tv"],
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
                        default=1.0)
    parser.add_argument("--tv_min", 
                        type=float, 
                        default=0.5)
    parser.add_argument("--tv_max", 
                        type=float,
                        default=1e-2)
    parser.add_argument("--inner_steps", 
                        type=int, 
                        default=10)

elif base_args.method == "edip":
    parser.add_argument("--pretrained_path", 
                        type=str,
                        default="pretrained_model/epoch_8_nn_learned_params.pt")
    

elif base_args.method == "edip_tv":
    parser.add_argument("--pretrained_path", 
                        type=str,
                        default="pretrained_model/epoch_8_nn_learned_params.pt")
    parser.add_argument("--tv_strength", 
                        type=float,
                        default=1e-5)
    
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
    parser.add_argument("--relative_noise", 
                        type=float,
                        default=0.01, 
                        help="relative level of noise added to input")
else:
    pass 

args = parser.parse_args()


time_save = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = f"dip_results/{args.phantom}/{args.method}/{args.model_inp}/{time_save}"
os.makedirs(save_dir, exist_ok=True)

save_dir_img = f"dip_results/{args.phantom}/{args.method}/{args.model_inp}/{time_save}/imgs"
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

if args.phantom == "walnut":
    data = get_walnut_data(cfg, ray_trafo=ray_trafo)

    y, x, x_fbp = data[0]
elif args.phantom == "shepplogan":
    from skimage.data import shepp_logan_phantom
    sl = shepp_logan_phantom() 
    from skimage.transform import resize

    #x = resize(sl, (501, 501))
    pad_top = 50
    pad_bottom = 51
    pad_left = 50
    pad_right = 51

    x = np.pad(sl, 
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant', 
            constant_values=0)

    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float().to(cfg.device)
    y = ray_trafo.trafo(x) 
    g = torch.Generator(device=y.device).manual_seed(1234)
    y = y + 0.01 * torch.mean(y.abs()) * torch.randn(y.shape, generator=g, device=y.device)
    x_fbp = ray_trafo.fbp(y)
else:
    raise NotImplementedError

#y = y[0,0,0,:].unsqueeze(-1)
im_size = x.shape[-1]

img = x[0,0].cpu().numpy() * 255
img = img.astype(np.uint8)
Image.fromarray(img).save(os.path.join(save_dir, "groundtruth.png"))

img_fbp = torch.clamp(x_fbp[0,0], 0,1).cpu().numpy() * 255
img_fbp = img_fbp.astype(np.uint8)
Image.fromarray(img_fbp).save(os.path.join(save_dir, "fbp.png"))

print("Number of parameters: ", sum([p.numel() for p in model.parameters()]))
if args.model_inp == "fbp":
    z = x_fbp
elif args.model_inp == "adjoint":
    z = ray_trafo.trafo_adjoint(y).detach()
    z = z/torch.max(z)
else:
    g = torch.Generator()
    g.manual_seed(args.random_seed_noise)
    z = 0.1*torch.randn(x.shape, generator=g)

z = z.to(device)
y_noise = y.to(device)

best_psnr = {'value': 0, 'idx': 0, 'reco': None}
best_psnr_early_stopping = {'value': 0, 'index': 0, 'reco': None}

# defaults from subspace DIP paper for early stopping
patience = 1000
delta = 0.99
w = 100
variance_list = []
callbacks = [track_best_psnr_output(best_psnr), 
             save_images(save_dir_img, skip=100), 
             early_stopping(patience=patience, delta=delta, w=w,variance_list=variance_list, best_psnr=best_psnr_early_stopping)]

logger_kwargs = {
    "use_wandb": args.use_wandb,
    "project": WANDB_PROJECT,
    "log_file": os.path.join(save_dir, f"log_{datetime.now():%Y-%m-%d_%H-%M-%S}.log"),
    "console_printing": True,
    "image_logging": 25,
    "wandb_config": {
                     "project": WANDB_PROJECT,
                     "entity": WANDB_ENTITY, 
                     "name": f"DIP_{args.method}_{args.model_inp}_{args.random_seed}",
                     "mode": "online" if args.use_wandb else "disabled",
                     "settings": wandb.Settings(code_dir="wandb"),
                     "dir": "wandb_logs",
                     "config": vars(args),},
}

if args.method == "vanilla":

    dip = DeepImagePrior(model=model, 
                         lr=args.lr, 
                         num_steps=args.num_steps, 
                         noise_std=args.noise_std, 
                         callbacks=callbacks,
                         save_dir=save_dir)
    
    x_pred, psnr_list, loss_list = dip.train(ray_trafo, y, z, x_gt=x)
elif args.method == "tv_hqs":
    dip = DeepImagePriorHQS(model=model, 
                         lr=args.lr, 
                         num_steps=args.num_steps, 
                         noise_std=args.noise_std, 
                         splitting_strength=args.splitting_strength, 
                         tv_min=args.tv_min, 
                         tv_max=args.tv_max, 
                         inner_steps=args.inner_steps,
                         callbacks=callbacks,
                         save_dir=save_dir)

    x_pred, psnr_list, loss_list = dip.train(ray_trafo, y, z, x_gt=x)
elif args.method == "tv":
    dip = DeepImagePriorTV(model=model, 
                         lr=args.lr, 
                         num_steps=args.num_steps, 
                         noise_std=args.noise_std, 
                         tv_strength=args.tv_strength,
                         callbacks=callbacks,
                         save_dir=save_dir)

    x_pred, psnr_list, loss_list = dip.train(ray_trafo, y, z, x_gt=x)

elif args.method == "edip_tv":
    model.load_state_dict(torch.load(args.pretrained_path))
    dip = DeepImagePriorTV(model=model, 
                         lr=args.lr, 
                         num_steps=args.num_steps, 
                         noise_std=args.noise_std, 
                         tv_strength=args.tv_strength,
                         callbacks=callbacks,
                         save_dir=save_dir)

    x_pred, psnr_list, loss_list = dip.train(ray_trafo, y, z, x_gt=x)
elif args.method == "aseq":
    dip = AutoEncodingSequentialDeepImagePrior(model=model, 
                         lr=args.lr, 
                         num_steps=args.num_steps, 
                         noise_std=args.noise_std, 
                         denoise_strength=args.denoise_strength,
                         callbacks=callbacks)
    x_pred, psnr_list, loss_list = dip.train(ray_trafo, y, z, x_gt=x, num_inner_steps=args.num_inner_steps,logger_kwargs=logger_kwargs)
elif args.method == "selfguided":
    dip = SelfGuidanceDeepImagePrior(model=model, 
                         lr=args.lr, 
                         num_steps=args.num_steps, 
                         noise_std=args.noise_std, 
                         denoise_strength=args.denoise_strength,
                         rel_noise=args.relative_noise,
                         callbacks=callbacks)
    x_pred, psnr_list, loss_list = dip.train(ray_trafo, y, z, x_gt=x, logger_kwargs=logger_kwargs,
                         num_noise_realisations=args.num_noise_realisations,
                         exp_weight=args.exp_weight)    
elif args.method == "edip":
    model.load_state_dict(torch.load(args.pretrained_path))
    dip = DeepImagePrior(model=model, 
                         lr=args.lr, 
                         num_steps=args.num_steps, 
                         noise_std=args.noise_std, 
                         callbacks=callbacks,
                         save_dir=save_dir)
    
    x_pred, psnr_list, loss_list = dip.train(ray_trafo, y, z, x_gt=x)
    
    x_pred, psnr_list, loss_list = dip.train(ray_trafo, y, z, x_gt=x)
else:
    raise NotImplementedError


print(best_psnr['value'], best_psnr['index'])


img = x_pred.detach().cpu().numpy()[0,0] * 255
img = img.astype(np.uint8)
Image.fromarray(img).save(os.path.join(save_dir, "final_reco.png"))

img = best_psnr['reco'][0,0].numpy() * 255
img = img.astype(np.uint8)
Image.fromarray(img).save(os.path.join(save_dir, "best_reco.png"))

results = {} 
results["best_psnr"] = float(best_psnr['value'])
results["best_psnr_idx"] = int(best_psnr['index'])

if best_psnr_early_stopping['reco'] is not None:
    img = best_psnr_early_stopping['reco'][0,0].numpy() * 255
    img = img.astype(np.uint8)
    Image.fromarray(img).save(os.path.join(save_dir, "best_reco_early_stopping.png"))
    results["best_psnr_early_stopping"] = float(best_psnr_early_stopping['value'])
    results["best_psnr_early_stopping_idx"] = int(best_psnr_early_stopping['index'])

    np.save(os.path.join(save_dir, "variance.npy"), np.asarray(variance_list))

with open(os.path.join(save_dir, "results.yaml"), "w") as f:
    yaml.dump(results, f)

args = vars(args)
with open(os.path.join(save_dir, "config.yaml"), "w") as f:
    yaml.dump(args, f)

loss_list = np.asarray(loss_list)
np.save(os.path.join(save_dir, "loss.npy"), loss_list)


psnr_list = np.asarray(psnr_list)
np.save(os.path.join(save_dir, "psnrs.npy"), psnr_list)