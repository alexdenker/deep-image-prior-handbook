import numpy as np
import torch

import matplotlib.pyplot as plt 
from tqdm import tqdm
from deepinv.physics import Tomography

from torch.func import jvp, functional_call, vjp



from dip import OperatorModule, get_unet_model

cfg = {
    "forward_operator": "radon",  #"downsampling", # "radon"
    "num_angles": 90,
    "rel_noise": 0.05,
    "model_inp": "fbp", # "random" "fbp"
}



device = "cuda"


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

num_rows = 50 #int(0.05 * np.prod(x.shape))



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

model = get_unet_model(in_ch=1, out_ch=1, scales=6,
                            skip=64, channels=(64,64,64,64,64,64), use_norm=False, use_sigmoid=False)
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
ax.loglog(np.arange(1, len(singular_values[0])+1), singular_values[0], label="init model")
ax.legend() 
ax.set_title("Singular values")
ax.set_xlabel("index")

plt.savefig(f"singular_values_{cfg["model_inp"]}.png")

fig, axes = plt.subplots(1,5, figsize=(12,4))
for j in range(4):
    axes[j].imshow(singular_vectors[0][:,j].reshape(128,128), cmap="gray")
    axes[j].set_title(f"Eigenvector {j}")
    axes[j].axis("off")
axes[4].imshow(x_recos[0][0,0], cmap="gray")
axes[4].set_title("Initial reconstruction")
axes[4].axis("off")
fig.tight_layout() 
plt.savefig(f"jacobian_singular_values_{cfg["model_inp"]}.png",bbox_inches='tight')
plt.show()