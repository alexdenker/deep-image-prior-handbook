
import os 
import numpy as np
import torch
import random


device = "cuda"

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
from unet import get_unet_model
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio

from tqdm import tqdm

import wandb

from deepinv.physics import Tomography, Downsampling
from operator_module import OperatorModule
from utils import tv_loss


cfg = {
    "forward_operator": "radon",  #"downsampling", # "radon"
    "lr": 1e-5,
    "num_angles": 60,
    "rel_noise": 0.05,
    "num_epochs": 3000,
    "img_log_freq": 50,
    "model_params": {
        "use_norm": False,
        "scales": 5,
        "use_sigmoid": False,
        "skip": 8,
        "channels": (128, 256, 512, 512, 256, 256)
    },
    "model_inp": "fbp", # "random"
    "tv_reg": 8e-4
}

wandb_name = f"dip_{cfg["forward_operator"]}_device={device}_linearised"

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
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float().to(device)

    if cfg["forward_operator"] == "radon":
        A = Tomography(angles=cfg["num_angles"], img_width=128, device=device) 
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

    from torch.func import jacrev, functional_call

    theta_0 = dict(model.named_parameters())
    #forward_pass = functional_call(model, theta_0, z)

    x_pred = model(z)
    
    def model_forward(*theta):
        return functional_call(model, theta, z)
    (output, jvp_out) = torch.func.jvp(model_forward, (theta_0,), (theta_0,))

    print(output.shape, jvp_out.shape)

    print(torch.sum((x_pred - output)**2))

    #jacobians = jacrev(functional_call, argnums=0)(model, theta_0, (z, ))

    #print(jacobians.shape)

    #x_pred = model(z)
    #loss = torch.mean((A(x_pred) - y_noise)**2) 
    #loss.backward() 
    