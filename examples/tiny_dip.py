

import torch
import torch.nn as nn 
import torch.nn.functional as F
from tqdm import tqdm
from deepinv.physics import Tomography
from skimage.metrics import peak_signal_noise_ratio
import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt 

from operator_module import OperatorModule
from model.utils import get_scale_modules
from model.autoencoder import Autoencoder

cfg = {
    "forward_operator": "radon",  #"downsampling", # "radon"
    "num_angles": 90,
    "rel_noise": 0.05,
    "model_inp": "fbp", # "random" "fbp"
}


"""
class UNet(nn.Module):
    def __init__(self, mean, std):
        super(UNet, self).__init__()
        self.scale_in, self.scale_out = get_scale_modules(ch_in=1, ch_out=1, mean_in=mean, mean_out=mean, std_in=std, std_out=std)        
        
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
        self.final_conv = nn.Conv2d(128, 1, kernel_size=1)  # final RGB output

        self.relu = nn.ReLU(inplace=True)
        #self.sigmoid = nn.Sigmoid() 
128
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    
    def forward(self, x):
        # Encoder
        x = self.scale_in(x)
        x1 = self.relu(self.conv1(x))  # convd1 + relu1
        x1_down = F.interpolate(x1, scale_factor=0.5, mode='bilinear', align_corners=False)  # down1

        x2 = self.relu(self.conv2(x1_down))  # convd2 + relu2
        x2_down = F.interpolate(x2, scale_factor=0.5, mode='bilinear', align_corners=False)  # down2

        x3 = self.relu(self.conv3(x2_down))  # convd3 + relu3
        x3_down = F.interpolate(x3, scale_factor=0.5, mode='bilinear', align_corners=False)  # down3

        x4 = self.relu(self.conv4(x3_down))  # convd4 + relu4
        x4 = self.relu(self.conv4_b(x4))     # conv4

        # Decoder
        x_up1 = F.interpolate(x4, scale_factor=2.0, mode='bilinear', align_corners=False)  # up1
        x_up1 = self.relu(self.upconv1(torch.cat([x_up1, x3], dim=1)))  # skip connection from x3

        x_up2 = F.interpolate(x_up1, scale_factor=2.0, mode='bilinear', align_corners=False)  # up2
        x_up2 = self.relu(self.upconv2(torch.cat([x_up2, x2], dim=1)))  # skip connection from x2

        x_up3 = F.interpolate(x_up2, scale_factor=2.0, mode='bilinear', align_corners=False)  # up3
        x_up3 = self.relu(self.upconv3(torch.cat([x_up3, x1], dim=1)))  # skip connection from x1

        out = self.final_conv(x_up3)  # convu4
        out = self.scale_out(out)
        return out #self.sigmoid(out)
"""


#device = "cuda:1"
device = "cuda"
# If I set torch.manuel_seed it is still random 
torch.manual_seed(1)

x = torch.load("walnut.pt")
x = x.float().to(device)

x = torch.nn.functional.interpolate(x, size=(128,128), mode="bilinear")

print("x: ", x.shape)
print("mean: ", torch.mean(x))
print("std: ", torch.std(x))


#model = get_unet_model(use_norm=cfg["model_params"]["use_norm"], 
#                        scales=cfg["model_params"]["scales"],
#                        use_sigmoid=cfg["model_params"]["use_sigmoid"], 
#                        skip=cfg["model_params"]["skip"],
#                        channels=cfg["model_params"]["channels"])
#model = UNet(mean=torch.mean(x).item(), std=torch.std(x).item()) #Autoencoder()
model = Autoencoder(channels=128, padding_mode="circular", use_norm=True, use_sigmoid=True)
model.to(device)
model.train()

A = Tomography(angles=cfg["num_angles"], img_width=128, device=device) 

L = A.compute_norm(torch.rand_like(x)).item()

print("L:", L)

A = OperatorModule(A)

y = A(x)
print("noise std: ", cfg["rel_noise"]*torch.mean(y.abs()))
g = torch.Generator()
g.manual_seed(1)
y_noise = y + cfg["rel_noise"]*torch.mean(y.abs())*torch.randn(y.shape, generator=g).to(device)
x_fbp = A.A_dagger(y_noise) 
print(y.shape, y_noise.shape)

print("Number of parameters: ", sum([p.numel() for p in model.parameters()]))
if cfg["model_inp"] == "fbp":
    z = x_fbp
else:
    g = torch.Generator()
    g.manual_seed(42)
    z = torch.randn(x.shape, generator=g)


z = z.to(device)
y_noise = y_noise.to(device)

torch.save(z, "dip_inp.pt")

with torch.no_grad():
    x_pred = model(z)


num_steps = 200000
save_at = [1,2,4,8, 10, 50, 100, 500, 1000, 2000, 4000, 6000, 8000, 12000, 20000, 80000,100000, num_steps]
optim = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-5, amsgrad = True, betas=(0.9,0.99))
#optim = torch.optim.SGD(model.parameters(), lr=1./512)

torch.save(model.state_dict(), f"UNet_GD_steps={0}.pt")

psnr = peak_signal_noise_ratio(x[0,0,:,:].cpu().numpy(), x_pred[0,0,:,:].detach().cpu().numpy())
psnr_fbp = peak_signal_noise_ratio(x[0,0,:,:].cpu().numpy(), x_fbp[0,0,:,:].detach().cpu().numpy())

print(f"Get PSNR = {psnr:.4f}dB at step {0}")

psnr_list = [] 
for i in tqdm(range(num_steps)):
    optim.zero_grad()

    x_pred = model(z)
    loss = torch.sum((A(x_pred) - y_noise)**2/L**2)
    loss.backward()

    #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optim.step() 

    psnr_list.append(peak_signal_noise_ratio(x[0,0,:,:].cpu().numpy(), x_pred[0,0,:,:].detach().cpu().numpy()))
    if (i+1) % 500 == 0:
        print(f"Get PSNR = {psnr_list[-1]:.4f}dB at step {i+1}")
    if (i+1) in save_at:
        torch.save(model.state_dict(), f"UNet_GD_steps={i+1}.pt")

        psnr = peak_signal_noise_ratio(x[0,0,:,:].cpu().numpy(), x_pred[0,0,:,:].detach().cpu().numpy())
        print(f"Get PSNR = {psnr:.4f}dB at step {i+1}")

        #plt.figure()
        #plt.imshow(x_pred[0,0,:,:].detach().cpu().numpy(), cmap="gray")
        #plt.show()
        
        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(13,6))

        ax1.plot(psnr_list, label="DIP PSNR")
        ax1.set_title("PSNR")
        ax1.hlines(psnr_fbp, 0, len(psnr_list), colors="red", label="FBP PSNR")
        ax1.legend() 
        im = ax2.imshow(x_pred[0,0,:,:].detach().cpu().numpy(), cmap="gray")
        fig.colorbar(im, ax=ax2)
        ax2.set_title("Reco")
        ax2.axis("off")

        im = ax3.imshow(x[0,0,:,:].detach().cpu().numpy(), cmap="gray")
        fig.colorbar(im, ax=ax3)
        ax3.set_title("GT")
        ax3.axis("off")

        plt.savefig(f"DIP_reconstruction_at_step={i+1}.png")
        plt.close()

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(13,6))

ax1.plot(psnr_list)
ax1.set_title("PSNR")

ax2.imshow(x_pred[0,0,:,:].detach().cpu().numpy(), cmap="gray")
ax2.set_title("Reco")

ax3.imshow(x[0,0,:,:].detach().cpu().numpy(), cmap="gray")
ax3.set_title("GT")

plt.savefig("DIP_reconstruction.png")
plt.show()