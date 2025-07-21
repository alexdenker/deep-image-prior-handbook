import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt 
from skimage.metrics import peak_signal_noise_ratio
from tqdm import tqdm
from deepinv.physics import Tomography

from torch.func import jvp, functional_call, vjp

from physics.operator_module import OperatorModule
from model.autoencoder import Autoencoder

cfg = {
    "forward_operator": "radon",  #"downsampling", # "radon"
    "num_angles": 90,
    "rel_noise": 0.05,
    "model_inp": "fbp", # "random" "fbp"
}


class UNet(nn.Module):
    def __init__(self):#, mean, std):
        super(UNet, self).__init__()
        
        # Encoder
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv4_b = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # Decoder
        self.upconv1 = nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1)  # concat with conv3 output
        self.upconv2 = nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1)  # concat with conv2 output
        self.upconv3 = nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1)  # concat with conv1 output
        self.final_conv = nn.Conv2d(128, 1, kernel_size=1) 
        self.relu = nn.ReLU(inplace=True)
        #self.sigmoid = nn.Sigmoid() 

        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Encoder
        x1 = self.relu(self.conv1(x))  # convd1 + relu1
        x1_down = F.interpolate(x1, scale_factor=0.5, mode='bilinear', align_corners=True)  # down1

        x2 = self.relu(self.conv2(x1_down))  # convd2 + relu2
        x2_down = F.interpolate(x2, scale_factor=0.5, mode='bilinear', align_corners=True)  # down2

        x3 = self.relu(self.conv3(x2_down))  # convd3 + relu3
        x3_down = F.interpolate(x3, scale_factor=0.5, mode='bilinear', align_corners=True)  # down3

        x4 = self.relu(self.conv4(x3_down))  # convd4 + relu4
        x4 = self.relu(self.conv4_b(x4))     # conv4

        # Decoder
        x_up1 = F.interpolate(x4, scale_factor=2.0, mode='bilinear', align_corners=True)  # up1
        x_up1 = self.relu(self.upconv1(torch.cat([x_up1, x3], dim=1)))  # skip connection from x3

        x_up2 = F.interpolate(x_up1, scale_factor=2.0, mode='bilinear', align_corners=True)  # up2
        x_up2 = self.relu(self.upconv2(torch.cat([x_up2, x2], dim=1)))  # skip connection from x2

        x_up3 = F.interpolate(x_up2, scale_factor=2.0, mode='bilinear', align_corners=True)  # up3
        x_up3 = self.relu(self.upconv3(torch.cat([x_up3, x1], dim=1)))  # skip connection from x1

        out = self.final_conv(x_up3)  # convu4
        return out #self.sigmoid(out)


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


#model = UNet(mean=torch.mean(x).item(), std=torch.std(x).item()) 
model = UNet()
#model = Autoencoder(channels=512, padding_mode="circular", use_norm=True)

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