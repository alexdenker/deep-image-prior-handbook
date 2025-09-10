import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt 
from skimage.metrics import peak_signal_noise_ratio
from tqdm import tqdm
from deepinv.physics import Tomography

from torch.func import jvp, functional_call, vjp

from dip import get_unet_model, OperatorModule

cfg = {
    "forward_operator": "radon",  #"downsampling", # "radon"
    "num_angles": 90,
    "rel_noise": 0.05,
    "model_inp": "fbp", # "random" "fbp"
}



device = "cuda"

saved_steps = [0,2,10,1000, 4000, 8000]#,4000, 8000]#, 12000, 20000,80000,200000]


x = torch.load("walnut.pt")
x = x.float().to(device)

x = torch.nn.functional.interpolate(x, size=(128,128), mode="bilinear")


A = Tomography(angles=cfg["num_angles"], img_width=128, device=device) 
A = OperatorModule(A)

g = torch.Generator()
g.manual_seed(1)
y = A(x)
y_noise = y + cfg["rel_noise"]*torch.mean(y.abs())*torch.randn(y.shape, generator=g).to(device)

x_fbp = A.A_dagger(y_noise) 

if cfg["model_inp"] == "fbp":
    z = x_fbp
else:
    g = torch.Generator()
    g.manual_seed(42)
    z = torch.randn(x.shape, generator=g)

z = z.to(device)
y_noise = y_noise.to(device)

num_rows = 100 #int(0.05 * np.prod(x.shape))



def get_jacobian_col(col_index, theta):
    
    _, vjp_fun = vjp(model_forward, theta)

    #v = torch.zeros_like(x).ravel()
    #v[col_index] = 1.0 
    #v = v.reshape(*x.shape)
    v = torch.randn_like(x)
    jac_col = vjp_fun(v)[0] # this should get me J.T v 

    _, jvp_out = jvp(model_forward, (theta,), (jac_col,)) # this should get me J J.T v
    
    return jvp_out.detach()

def get_K_vec(v, theta):
    
    _, vjp_fun = vjp(model_forward, theta)

    v = v.reshape(*x.shape)
    jac_col = vjp_fun(v)[0] # this should get me J.T v 

    _, jvp_out = jvp(model_forward, (theta,), (jac_col,)) # this should get me J J.T v
    
    return jvp_out.detach()

def align_signs(U_approx, U_true):

    U_aligned = np.copy(U_approx)
    for i in range(U_approx.shape[1]):
        sign = np.sign(np.dot(U_true[:, i], U_approx[:, i]))
        U_aligned[:, i] *= sign
    return U_aligned

x_recos = []
singular_values = [] 
singular_vectors = [] 

for save_step in saved_steps:
    print("Do step: ", save_step)
    #model = UNet(mean=torch.mean(x).item(), std=torch.std(x).item()) 
    #model = Autoencoder(channels=512, padding_mode="circular", use_norm=True)
    model = get_unet_model(in_ch=1, out_ch=1, scales=6,
                            skip=64, channels=(64,64,64,64,64,64), use_norm=False, use_sigmoid=False)
    #model.load_state_dict(torch.load(f"UNet_GD_steps={save_step}.pt"))
    model.to(device)   
    model.eval() 

    with torch.no_grad():
        x_pred = model(z)
    x_recos.append(x_pred.cpu().numpy())

    def model_forward(theta):
        return functional_call(model, theta, z)


    theta_model = dict(model.named_parameters())

    rows_idx = np.random.choice(np.prod(x.shape), num_rows, replace=False)
    rows_idx = np.sort(rows_idx)

    jac_approx = [] 

    for row_idx in tqdm(rows_idx):
        jac_row = get_jacobian_col(row_idx, theta_model)
        jac_approx.append(jac_row.ravel())    

    jac_approx = torch.stack(jac_approx, dim=-1)
    #print("jac approx: ", jac_approx.shape)

    Q, R = torch.linalg.qr(jac_approx)

    KQ_ = [] 

    for i in tqdm(range(Q.shape[-1])):
        jac_row = get_K_vec(Q[:,i], theta_model)
        KQ_.append(jac_row.ravel())    

    KQ = torch.stack(KQ_, dim=-1)

    B = Q.T @ KQ

    U_B, S_B, Vh_B = torch.linalg.svd(B)
    print(U_B.shape)
    U_approx = Q @ U_B

    singular_values.append(S_B.detach().cpu().numpy())
    if len(singular_vectors) > 0:
        singular_vectors.append(align_signs(U_approx.detach().cpu().numpy(), singular_vectors[-1]))
    else:
        singular_vectors.append(U_approx.detach().cpu().numpy())

    del model, theta_model, jac_approx, B, KQ

fig, ax = plt.subplots(1,1, figsize=(7,4))

for idx, step in enumerate(saved_steps):
    if idx == 0:
        ax.loglog(np.arange(1, len(singular_values[0])+1), singular_values[idx], label="init model")
    else:
        ax.loglog(np.arange(1, len(singular_values[0])+1), singular_values[idx], label=f"step {step}")
    

ax.legend() 
ax.set_title("Singular values")
ax.set_xlabel("index")

fig, axes = plt.subplots(5, len(saved_steps), figsize=(18,12))

for i in range(len(saved_steps)):
    for j in range(4):
        axes[j,i].imshow(singular_vectors[i][:,j].reshape(128,128), cmap="gray")
        axes[j,i].set_title(f"model {saved_steps[i]} | sv {j}")
        axes[j,i].axis("off")
    axes[4, i].imshow(x_recos[i][0,0], cmap="gray")
    axes[4, i].set_title(f"reco of model {saved_steps[i]}")
    axes[4, i].axis("off")
fig.tight_layout() 
plt.show()