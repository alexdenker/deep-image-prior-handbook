
import torch

import matplotlib.pyplot as plt 
from tqdm import tqdm

import deepinv as dinv 
from deepinv.physics import Tomography
from skimage.metrics import peak_signal_noise_ratio


device = "cuda"

num_angles = 60 
rel_noise = 0.1

# From random search:
# alpha = 3.99
# step size = 1.335 
# PSNR = 30.34 dB 

x = torch.load("walnut.pt")
x = x.float().to(device)

print("x: ", x.shape)

physics = Tomography(angles=num_angles, img_width=256, device=device) 
y = physics.A(x)
g = torch.Generator()
g.manual_seed(1)
y_noise = y + rel_noise*torch.mean(y.abs())*torch.randn(y.shape, generator=g).to(device)
L = physics.compute_norm(torch.rand_like(x))
print("L: ", L.item())

x_fbp = physics.A_dagger(y_noise)

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(18,7))

ax1.imshow(y[0,0,:,:].cpu().numpy())
ax1.set_title("clean sinogram")
ax1.axis("off")

ax2.imshow(y_noise[0,0,:,:].cpu().numpy())
ax2.set_title("noisy sinogram")
ax2.axis("off")

ax3.imshow(x_fbp[0,0,:,:].cpu().numpy(), cmap="gray")
ax3.set_title("FBP")
ax3.axis("off")

plt.show()



max_iter = 200 
step_size =  1.335 
tol = 1e-6

def TV_rec(y, physics, L, x_init, step_size, alpha, max_iter=2000, tol=1e-6):
    xk = torch.clone(x_init)
    x_prev = torch.clone(xk)

    prior =  dinv.optim.prior.TVPrior(n_it_max=100)

    # FISTA
    tk = torch.tensor(1.0)
    for _ in tqdm(range(max_iter)):
        tk_new = (1 + torch.sqrt(1 + 4 * tk**2))/2
        ak = (tk -1) / tk_new

        xk_tilde = xk + ak * (xk - x_prev)

        res = physics.A(xk_tilde)  - y_noise 
        x_next = xk_tilde - step_size/L * physics.A_adjoint(res)
        x_next = prior.prox(x_next, gamma=step_size/L*alpha)
        x_next[x_next < 0] = 0.
        x_next[x_next > 1] = 1.

        x_prev = xk 
        xk = x_next
        tk = tk_new

        if torch.sum((xk - x_prev)**2).item() < tol:
            break
        
    return xk 




alpha = 15.0 
x_init = torch.zeros_like(x)
x_rec = TV_rec(y_noise, physics, L, x_init, step_size, alpha, max_iter=max_iter, tol=tol)

psnr = peak_signal_noise_ratio(x[0,0,:,:].cpu().numpy(), x_rec[0,0,:,:].detach().cpu().numpy())
print(f"Get PSNR = {psnr:.4f}dB")



fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(12,6))
ax1.imshow(x[0,0].cpu().numpy(), cmap="gray")
ax2.imshow(x_rec[0,0].cpu().numpy(), cmap="gray")
#ax3.semilogy(res_list)
#ax3.set_title("Mean Squared Error")
#ax4.plot(psnr_list)
#ax4.set_title("PSNR")
plt.show()