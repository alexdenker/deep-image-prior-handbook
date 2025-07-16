
import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import random
import matplotlib.pyplot as plt 

from deepinv.physics import Tomography

from torch.func import functional_call

from physics.operator_module import OperatorModule
from model.unet import get_unet_model


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
        "use_norm": False,
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
    "tv_reg": 4e-4,
    "tv_type": "anisotropic"
}

x = torch.load("walnut.pt")
x = x.float().to(device)

print("x: ", x.shape)

A = Tomography(angles=cfg["num_angles"], img_width=256, device=device) 
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

plt.show()

 

theta_0 = dict(model.named_parameters())
#forward_pass = functional_call(model, theta_0, z)

x_pred = model(z)

def model_forward(*theta):
    return functional_call(model, theta, z)

#(output, jvp_out) = torch.func.jvp(model_forward, (theta_0,), (theta_0,))

(_, vjpfunc) = torch.func.vjp(model_forward, theta_0)


print(vjpfunc)
#print(output.shape, jvp_out.shape)


#jacobians = jacrev(functional_call, argnums=0)(model, theta_0, (z, ))

#print(jacobians.shape)

#x_pred = model(z)
#loss = torch.mean((A(x_pred) - y_noise)**2) 
#loss.backward() 