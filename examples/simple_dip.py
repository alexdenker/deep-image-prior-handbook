
import os 
import numpy as np
import torch
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 
import skimage 
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio
from tqdm import tqdm
import wandb
from deepinv.physics import Tomography, Downsampling


from operator_module import OperatorModule

from model.unet import get_unet_model
from model.utils import tv_loss, isotropic_tv_loss


device = "cuda"

# If I set torch.manuel_seed it is still random 
torch.manual_seed(1)
random.seed(1)
np.random.seed(1)
torch.cuda.manual_seed_all(1)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)



cfg = {
    "forward_operator": "radon",  #"downsampling", # "radon"
    "lr": 1e-4,
    "num_angles": 90,
    "rel_noise": 0.05,
    "num_epochs": 10000,
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
    "optimiser": "adam", # "lbfgs" "adam" "gd" #amsgrad # "rmsprop"
    "betas": (0.9, 0.999),
    "momentum" : 0.0,
    "weight_decay": 0.0,
    "tv_reg": 4e-4
    ,
    "tv_type": "anisotropic"
}

wandb_name = f"dip_{cfg["forward_operator"]}_device={device}_{cfg["optimiser"]}"
if cfg["optimiser"] == "gd" and cfg["momentum"] > 0:
    wandb_name = wandb_name + "+momentum"

wandb_name = wandb_name + f"_tv={cfg["tv_reg"]}"

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
    
    if cfg["tv_type"] == "isotropic":
        tv_loss = isotropic_tv_loss

    x = torch.load("walnut.pt")
    
    #x = np.array(skimage.data.shepp_logan_phantom())
    #x = resize(x, (128,128), anti_aliasing=True)
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

    if cfg["optimiser"] == "lbfgs":
        optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=1, history_size=10, line_search_fn="strong_wolfe")
    elif cfg["optimiser"] == "adam":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], eps=1e-8, betas=cfg["betas"], weight_decay=cfg["weight_decay"])
    elif cfg["optimiser"] == "amsgrad":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], eps=1e-8, amsgrad=True, weight_decay=cfg["weight_decay"])
    elif cfg["optimiser"] == "gd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
    elif cfg["optimiser"] == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    else:
        raise NotImplementedError
    #print("NUMBER OF PARAMS: ", sum([p.numel() for p in model.parameters()]))
    func_evals = 0

    best_psnr = 0.
    best_psnr_iter = 0
    best_psnr_img = torch.clone(z)
    for i in tqdm(range(cfg["num_epochs"])):
        wandb_dict = {} 
        if cfg["optimiser"] == "lbfgs":
            optimizer.zero_grad()
            # I dont know how to get x_pred from the closure
            # it is also not saved as part of the state_dict of the optimiser 
            # maybe we have to slightly rewrite the LBFGS optimiser to also output x_pred 
            # else we always have one additional network evaluation for visualisation
            with torch.no_grad():
                x_pred = model(z)

            def closure():
                if torch.is_grad_enabled():
                    optimizer.zero_grad()
                x_pred = model(z)
                loss = torch.mean((A(x_pred) - y_noise)**2) + cfg["tv_reg"] * tv_loss(x_pred)
                if loss.requires_grad:
                    loss.backward()
                return loss

            loss = optimizer.step(closure)

            #print(optimizer.state_dict())
            #print(optimizer.state_dict()["state"][0]["d"].shape, optimizer.state_dict()["state"][0]["t"])
            d = optimizer.state_dict()["state"][0]["d"]
            t = optimizer.state_dict()["state"][0]["t"]
            wandb_dict["lr"] = t
            scaled_step = d.mul(t).abs().max()
            if scaled_step < 1e-9:
                break
            #print("max gradient: ", d.mul(t).abs().max())
            #print("step size: ", t)
            #print(optimizer.state_dict()["state"][0].keys())
            #print(optimizer.state_dict()["d"].shape, optimizer.state_dict()["t"].shape)
            func_evals = optimizer.state_dict()["state"][0]["func_evals"]

            wandb_dict["train/total_loss"] = loss.item()
        else:
            optimizer.zero_grad()

            if cfg["inp_noise"] > 0:
                z_inp = z + cfg["inp_noise"] * torch.randn_like(z)
            else:
                z_inp = z
            x_pred = model(z_inp)
            mse_loss = torch.mean((A(x_pred) - y_noise)**2) 
            reg_loss =  tv_loss(x_pred)
            loss = mse_loss + cfg["tv_reg"]* reg_loss

            func_evals +=1
            loss.backward() 

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            wandb_dict["train/mse_loss"] = mse_loss.item()
            wandb_dict["train/reg_loss"] = reg_loss.item()
            wandb_dict["train/total_loss"] = loss.item()

        wandb_dict["step"] = i 
        wandb_dict["func_evals"] = func_evals

        psnr = peak_signal_noise_ratio(x[0,0,:,:].cpu().numpy(), x_pred[0,0,:,:].detach().cpu().numpy())
        if psnr > best_psnr:
            best_psnr = psnr 
            best_psnr_iter = i 
            best_psnr_img = torch.clone(x_pred.detach().cpu())
        #print("PSNR: ", psnr, " Loss: ", loss.item(), "")

        wandb_dict["train/psnr"] = psnr
        # Compute gradient norm
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)  # L2 norm of gradients
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        wandb_dict["train/gradient_norm"] = total_norm
        
        if i % cfg["img_log_freq"] == 0:       

            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(16,7))

            ax1.imshow(x[0,0,:,:].cpu().numpy(), cmap="gray")
            ax1.set_title("ground truth")
            ax1.axis("off")
            ax2.imshow(x_pred[0,0,:,:].detach().cpu().numpy(), cmap="gray")
            ax2.set_title("prediction")
            ax2.axis("off")
            ax3.imshow(x_fbp[0,0,:,:].detach().cpu().numpy(), cmap="gray")
            ax3.set_title("FBP")
            ax3.axis("off")

            ax4.imshow(best_psnr_img[0,0,:,:].numpy(), cmap="gray")
            ax4.set_title(f"Best prediction (iter {best_psnr_iter})")
            ax4.axis("off")
            fig.suptitle(f"DIP at iter {i}")
            wandb_dict["reconstruction"] = wandb.Image(plt)
            #wandb.log({"reconstruction": wandb.Image(plt)})
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
