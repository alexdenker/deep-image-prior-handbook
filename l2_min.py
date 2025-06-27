
import os 
import numpy as np
import torch
import random

device = "cpu"

# If I set torch.manuel_seed it is still random 
torch.manual_seed(1)
random.seed(1)
np.random.seed(1)
torch.cuda.manual_seed_all(1)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

#os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import matplotlib.pyplot as plt 
import skimage 
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio

from tqdm import tqdm

import wandb

from deepinv.physics import Tomography, Downsampling
from deepinv.optim.utils import conjugate_gradient

cfg = {
    "lr": 0.1,
    "num_angles": 40,
    "rel_noise": 0.05,
    "num_epochs": 6000,
    "img_log_freq": 50,
    "optimiser": "adam", # "lbfgs" "adam" "gd" 
    "momentum" : 0.0
}

wandb_name = "optimise_image"

wandb_kwargs = {
        "project": "deep-image-prior-handbook",
        "entity": "alexanderdenker",
        "config": cfg,
        "name": wandb_name,
        "mode": "disabled", #"disabled", #"online" ,
        "settings": wandb.Settings(code_dir="wandb"),
        "dir": "wandb",
    }

with wandb.init(**wandb_kwargs) as run:
    
    x = np.array(skimage.data.shepp_logan_phantom())
    x = resize(x, (128,128), anti_aliasing=True)
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float()

    #A = Tomography(angles=cfg["num_angles"], img_width=128) 
    A = Downsampling(filter="gaussian", img_size=(1,128,128), factor=4)

    y = A(x)
    y_noise = y + cfg["rel_noise"]*torch.mean(y.abs())*torch.randn(y.shape)
    x_adj = A.A_adjoint(y_noise) #A.A_dagger(y_noise) 
    print(y.shape, y_noise.shape)
    
    x_lsq = conjugate_gradient(lambda x: A.A_adjoint(A(x)), A.A_adjoint(y_noise), verbose=True, max_iter=20)
    
    print("Error of LSQ: ", torch.mean((A(x_lsq) - y_noise)**2))

    x_init = torch.ones_like(x)
    xk = torch.nn.Parameter(x_init, requires_grad=True)

    y_noise = y_noise.to(device)

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5, figsize=(18,7))

    ax1.imshow(y[0,0,:,:].cpu().numpy(), cmap="gray")
    ax1.set_title("clean data")
    ax1.axis("off")

    ax2.imshow(y_noise[0,0,:,:].cpu().numpy(), cmap="gray")
    ax2.set_title("noisy data")
    ax2.axis("off")

    ax3.imshow(x_adj[0,0,:,:].cpu().numpy(), cmap="gray")
    ax3.set_title("Adjoint")
    ax3.axis("off")
    
    ax4.imshow(xk[0,0,:,:].detach().cpu().numpy(), cmap="gray")
    ax4.set_title("Initialisation")
    ax4.axis("off")
    
    ax5.imshow(x_lsq[0,0,:,:].detach().cpu().numpy(), cmap="gray")
    ax5.set_title("least squares sol")
    ax5.axis("off")
    wandb.log({"data": wandb.Image(plt)})
    plt.show()

    if cfg["optimiser"] == "lbfgs":
        optimizer = torch.optim.LBFGS([xk], lr=1.0, max_iter=1, history_size=10, line_search_fn="strong_wolfe")
    elif cfg["optimiser"] == "adam":
        beta1 = 0.1
        beta2 = 0.999
        optimizer = torch.optim.Adam([xk], lr=cfg["lr"], betas=(beta1, beta2))
    elif cfg["optimiser"] == "gd":
        optimizer = torch.optim.SGD([xk], lr=cfg["lr"], momentum=cfg["momentum"])
    else:
        raise NotImplementedError
    #print("NUMBER OF PARAMS: ", sum([p.numel() for p in model.parameters()]))
    func_evals = 0

    loss_list = []
    for i in tqdm(range(cfg["num_epochs"])):
        wandb_dict = {} 
        if cfg["optimiser"] == "lbfgs":
            
            def closure():
                if torch.is_grad_enabled():
                    optimizer.zero_grad()
                loss = torch.mean((A(xk) - y_noise)**2) #+ torch.nn.functional.relu(-xk).sum() + 0.1* torch.mean(xk**2)
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
        else:
            optimizer.zero_grad()

            loss = torch.mean((A(xk) - y_noise)**2)
            func_evals +=1
            loss.backward() 
            optimizer.step()

            print(optimizer.state_dict()["state"][0]["exp_avg_sq"].shape)
        
        print("Norm of gradient: ", torch.mean(xk.grad**2))
        loss_list.append(loss.item())
        wandb_dict["train/mse_loss"] = loss.item()
        wandb_dict["step"] = i 
        wandb_dict["func_evals"] = func_evals

        psnr = peak_signal_noise_ratio(x[0,0,:,:].cpu().numpy(), xk[0,0,:,:].detach().cpu().numpy())
        print("PSNR: ", psnr, " Loss: ", loss.item(), "")

        wandb_dict["train/psnr"] = psnr.item() 

        if i % cfg["img_log_freq"] == 0:       
            
            if cfg["optimiser"] == "adam":
                fig, axes = plt.subplots(2,3, figsize=(16,7))

                im = axes[0,0].imshow(x[0,0,:,:], cmap="gray")
                axes[0,0].set_title("ground truth")
                axes[0,0].axis("off")
                fig.colorbar(im, ax=axes[0,0])
                im = axes[0,1].imshow(xk[0,0,:,:].detach().cpu().numpy(), cmap="gray")
                axes[0,1].set_title("prediction")
                axes[0,1].axis("off")
                fig.colorbar(im, ax=axes[0,1])
                im = axes[0,2].imshow(x_lsq[0,0,:,:].detach().cpu().numpy(), cmap="gray")
                axes[0,2].set_title("LSQ solution")
                axes[0,2].axis("off")
                fig.colorbar(im, ax=axes[0,2])

                im = axes[1,0].imshow(optimizer.state_dict()["state"][0]["exp_avg"][0,0,:,:].detach().cpu().numpy(), cmap="gray")

                eps = 1e-8
                exp_avg = optimizer.state_dict()["state"][0]["exp_avg"][0,0,:,:].detach().cpu()
                exp_avg_sq = optimizer.state_dict()["state"][0]["exp_avg_sq"][0,0,:,:].detach().cpu()
                denom = exp_avg_sq.sqrt() / (1 - beta2**i) + eps
                bias_correction1 = 1 - beta1**i

                axes[1,0].set_title("exp_avg")
                axes[1,0].axis("off")
                fig.colorbar(im, ax=axes[1,0])

                im = axes[1,1].imshow(exp_avg/bias_correction1*cfg["lr"]/denom.numpy(), cmap="gray")
                axes[1,1].set_title("effective gradient step")
                axes[1,1].axis("off")
                fig.colorbar(im, ax=axes[1,1])
                  
                axes[1,2].semilogy(loss_list)
                axes[1,2].set_title("||Ax - y ||^2")
                
                wandb_dict["reconstruction"] = wandb.Image(plt)
                #wandb.log({"reconstruction": wandb.Image(plt)})
                plt.close()

                #plt.show()

            else:
                fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(16,7))

                ax1.imshow(x[0,0,:,:], cmap="gray")
                ax1.set_title("ground truth")
                ax1.axis("off")
                ax2.imshow(xk[0,0,:,:].detach().cpu().numpy(), cmap="gray")
                ax2.set_title("prediction")
                ax2.axis("off")
                ax3.imshow(x_adj[0,0,:,:].detach().cpu().numpy(), cmap="gray")
                ax3.set_title("adjoint")
                ax3.axis("off")
                wandb_dict["reconstruction"] = wandb.Image(plt)
                #wandb.log({"reconstruction": wandb.Image(plt)})
                #plt.close()
                plt.show()
        wandb.log(wandb_dict)

    if cfg["optimiser"] == "adam":
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5, figsize=(16,7))

        im = ax1.imshow(x[0,0,:,:], cmap="gray")
        ax1.set_title("ground truth")
        ax1.axis("off")
        fig.colorbar(im, ax=ax1)
        im = ax2.imshow(xk[0,0,:,:].detach().cpu().numpy(), cmap="gray")
        ax2.set_title("prediction")
        ax2.axis("off")
        fig.colorbar(im, ax=ax2)
        im = ax3.imshow(x_lsq[0,0,:,:].detach().cpu().numpy(), cmap="gray")
        ax3.set_title("LSQ solution")
        ax3.axis("off")
        fig.colorbar(im, ax=ax3)

        im = ax4.imshow(optimizer.state_dict()["state"][0]["exp_avg"][0,0,:,:].detach().cpu().numpy(), cmap="gray")

        eps = 1e-8
        exp_avg = optimizer.state_dict()["state"][0]["exp_avg"][0,0,:,:].detach().cpu()
        exp_avg_sq = optimizer.state_dict()["state"][0]["exp_avg_sq"][0,0,:,:].detach().cpu()
        denom = exp_avg_sq.sqrt() / (1 - beta2**cfg["num_epochs"]) + eps
        bias_correction1 = 1 - beta1**cfg["num_epochs"]
        ax4.set_title("exp_avg")
        ax4.axis("off")
        fig.colorbar(im, ax=ax4)

        im = ax5.imshow(exp_avg/bias_correction1*cfg["lr"]/denom.numpy(), cmap="gray")
        ax5.set_title("effective gradient step")
        ax5.axis("off")
        fig.colorbar(im, ax=ax5)
        wandb.log({"final reconstruction": wandb.Image(plt)})
        #wandb.log({"reconstruction": wandb.Image(plt)})
        #plt.close()
        plt.show()

    else:
        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(16,7))

        ax1.imshow(x[0,0,:,:], cmap="gray")
        ax1.set_title("ground truth")
        ax1.axis("off")
        ax2.imshow(xk[0,0,:,:].detach().cpu().numpy(), cmap="gray")
        ax2.set_title("prediction")
        ax2.axis("off")
        ax3.imshow(x_adj[0,0,:,:].detach().cpu().numpy(), cmap="gray")
        ax3.set_title("Adjoint")
        ax3.axis("off")
        wandb.log({"final reconstruction": wandb.Image(plt)})
        #wandb.log({"reconstruction": wandb.Image(plt)})
        #plt.close()
        plt.show()
