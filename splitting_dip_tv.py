
import os 
import numpy as np
import torch
import random
import matplotlib
#matplotlib.use("Agg")

import matplotlib.pyplot as plt 
from skimage.metrics import peak_signal_noise_ratio
from tqdm import tqdm
import wandb
from deepinv.physics import Tomography, Downsampling
import deepinv as dinv 


from operator_module import OperatorModule

from model import get_unet_model
device = "cuda"

# If I set torch.manuel_seed it is still random 
torch.manual_seed(1)
random.seed(1)
np.random.seed(1)
torch.cuda.manual_seed_all(1)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

prior = dinv.optim.prior.TVPrior(n_it_max=100)

cfg = {
    "forward_operator": "radon",  #"downsampling", # "radon"
    "lr": 1e-4,
    "num_angles": 60,
    "rel_noise": 0.05,
    "num_dip_updates": 100,
    "num_iterations": 100,
    "img_log_freq": 100,
    "model_params": {
        "use_norm": True,
        "scales": 5,
        "use_sigmoid": False,
        "skip": 16,
        "channels": (32, 64, 128, 128, 256, 256),
        "activation" : "relu" # "silu"
    },
    "model_inp": "fbp", # "random" "fbp"
    "inp_noise": 0.05,
    "adam_betas": (0.9, 0.999),
    "tv_reg_max": 0.5,
    "tv_reg_min": 1e-3, 
    "lam": 25.0, 
}

wandb_name = f"splitting_dip_{cfg["forward_operator"]}_device={device}"


wandb_kwargs = {
        "project": "deep-image-prior-handbook",
        "entity": "alexanderdenker",
        "config": cfg,
        "name": wandb_name,
        "mode": "online", #"disabled", #"online" ,
        "settings": wandb.Settings(code_dir="wandb"),
        "dir": "wandb",
    }
with wandb.init(**wandb_kwargs) as run:
    
    x = torch.load("walnut.pt")

    x = x.float().to(device)

    print("x: ", x.shape)

    if cfg["forward_operator"] == "radon":
        A = Tomography(angles=cfg["num_angles"], img_width=256, device=device) 
    elif cfg["forward_operator"] == "downsampling":
        A = Downsampling(img_size=(1,128,128), filter="gaussian", factor=4, device=device, padding="circular")
    else:
        raise NotImplementedError

    A = OperatorModule(A)

    y = A(x)
    print("noise std: ", cfg["rel_noise"]*torch.mean(y.abs()))
    y_noise = y + cfg["rel_noise"]*torch.mean(y.abs())*torch.randn_like(y)
    x_fbp = A.A_dagger(y_noise) 
    print(y.shape, y_noise.shape)
    model = get_unet_model(use_norm=cfg["model_params"]["use_norm"], 
                            scales=cfg["model_params"]["scales"],
                            use_sigmoid=cfg["model_params"]["use_sigmoid"], 
                            skip=cfg["model_params"]["skip"],
                            channels=cfg["model_params"]["channels"])
    model.train()
    model.to(device)
    print("Number of parameters: ", sum([p.numel() for p in model.parameters()]))
    if cfg["model_inp"] == "fbp":
        z = x_fbp
    else:
        z = torch.randn(x.shape)

    z = z.to(device)
    y_noise = y_noise.to(device)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(18,7))

    ax1.imshow(y[0,0,:,:].cpu().numpy().T)
    ax1.set_title("clean sinogram")
    ax1.axis("off")

    ax2.imshow(y_noise[0,0,:,:].cpu().numpy().T)
    ax2.set_title("noisy sinogram")
    ax2.axis("off")

    ax3.imshow(x_fbp[0,0,:,:].cpu().numpy(), cmap="gray")
    ax3.set_title("FBP")
    ax3.axis("off")
    
    ax4.imshow(z[0,0,:,:].cpu().numpy(), cmap="gray")
    ax4.set_title("Input to Model")
    ax4.axis("off")

    wandb.log({"data": wandb.Image(plt)})
    plt.close()


    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], eps=1e-8, betas=cfg["adam_betas"])

    func_evals = 0

    best_psnr = 0.
    best_psnr_iter = 0
    best_psnr_img = torch.clone(z)
    
    x_splitting = torch.zeros_like(z)
    tv_reg = np.logspace(np.log10(cfg["tv_reg_min"]), np.log10(cfg["tv_reg_max"]), cfg["num_iterations"])[::-1]

    for step in range(cfg["num_iterations"]):
        print(f"Step {step+1} out of {cfg["num_iterations"]}")
        wandb_dict = {} 
        
        mse_loss_list = [] 
        reg_loss_list = [] 
        total_loss_list = [] 
        
        beta = cfg["lam"] / tv_reg[step] 
        for i in tqdm(range(cfg["num_iterations"])):
            optimizer.zero_grad()

            if cfg["inp_noise"] > 0:
                z_inp = z + cfg["inp_noise"] * torch.randn_like(z)
            else:
                z_inp = z
            x_pred = model(z_inp)
            mse_loss = torch.mean((A(x_pred) - y_noise)**2) 
            reg_loss = torch.mean((x_pred - x_splitting)**2)
            loss = mse_loss + beta * reg_loss

            func_evals +=1
            loss.backward() 

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            mse_loss_list.append(mse_loss.item())
            reg_loss_list.append(reg_loss.item())
            total_loss_list.append(loss.item())

        wandb_dict["train/mse_loss"] = np.mean(mse_loss_list)
        wandb_dict["train/reg_loss"] = np.mean(reg_loss_list)
        wandb_dict["train/total_loss"] = np.mean(total_loss_list)
        wandb_dict["train/beta"] = beta.item()

        wandb_dict["step"] = step
        wandb_dict["func_evals"] = func_evals

        psnr = peak_signal_noise_ratio(x[0,0,:,:].cpu().numpy(), x_pred[0,0,:,:].detach().cpu().numpy())
        if psnr > best_psnr:
            best_psnr = psnr 
            best_psnr_iter = step 
            best_psnr_img = torch.clone(x_pred.detach().cpu())

        wandb_dict["train/psnr"] = psnr

        with torch.no_grad():
            x_splitting = prior.prox(x_pred, gamma=tv_reg[step])


        fig, axes = plt.subplots(2,3, figsize=(16,7))

        axes[0,0].imshow(x[0,0,:,:].cpu().numpy(), cmap="gray")
        axes[0,0].set_title("ground truth")
        axes[0,0].axis("off")

        axes[0,1].imshow(x_pred[0,0,:,:].detach().cpu().numpy(), cmap="gray")
        axes[0,1].set_title("prediction")
        axes[0,1].axis("off")

        axes[0,2].imshow(x_splitting[0,0,:,:].cpu().numpy(), cmap="gray")
        axes[0,2].set_title("Splitting variable")
        axes[0,2].axis("off")

        axes[1,0].imshow(x_fbp[0,0,:,:].detach().cpu().numpy(), cmap="gray")
        axes[1,0].set_title("FBP")
        axes[1,0].axis("off")

        axes[1,1].imshow(best_psnr_img[0,0,:,:].numpy(), cmap="gray")
        axes[1,1].set_title(f"Best prediction (iter {best_psnr_iter})")
        axes[1,1].axis("off")

        axes[1,2].axis("off")

        fig.suptitle(f"DIP at iter {step}")
        wandb_dict["reconstruction"] = wandb.Image(plt)
        plt.close()
    


        wandb.log(wandb_dict)


    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(16,7))

    ax1.imshow(x[0,0,:,:].cpu().numpy(), cmap="gray")
    ax1.set_title("ground truth")
    ax1.axis("off")
    ax2.imshow(x_pred[0,0,:,:].detach().cpu().numpy(), cmap="gray")
    ax2.set_title("prediction")
    ax2.axis("off")
    ax3.imshow(x_fbp[0,0,:,:].detach().cpu().numpy(), cmap="gray")
    ax3.set_title("FBP")
    ax3.axis("off")
    wandb.log({"final reconstruction": wandb.Image(plt)})
    plt.close()
