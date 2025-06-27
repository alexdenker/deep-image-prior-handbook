
import torch

import matplotlib.pyplot as plt 
from tqdm import tqdm

import deepinv as dinv 
from deepinv.physics import Tomography
from skimage.metrics import peak_signal_noise_ratio


device = "cuda"

num_angles = 90 
rel_noise = 0.05 

# From random search:
# alpha = 3.99
# step size = 1.335 
# PSNR = 30.34 dB 

x = torch.load("walnut.pt")
x = x.float().to(device)

print("x: ", x.shape)

physics = Tomography(angles=num_angles, img_width=256, device=device) 
y = physics.A(x)
y_noise = y + rel_noise*torch.mean(y.abs())*torch.randn_like(y)

L = physics.compute_norm(torch.rand_like(x))
print("L: ", L.item())


max_iter = 2000 
step_size = 0.5 
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


# random search 
alpha_min = 1.
alpha_max = 15. 
step_size_min = 0.1 
step_size_max = 2.0 

n_tries = 10

best_psnr = 0 
best_alpha = None 
best_step_size = None 

for i in range(n_tries):

    alpha = torch.rand(1, device=device) * (alpha_max - alpha_min) + alpha_min
    step_size = torch.rand(1, device=device) * (step_size_max - step_size_min) + step_size_min

    print(f"Try alpha = {alpha.item():.3f} and step size = {step_size.item():.3f}")

    x_init = torch.zeros_like(x)
    x_rec = TV_rec(y_noise, physics, L, x_init, step_size, alpha)

    psnr = peak_signal_noise_ratio(x[0,0,:,:].cpu().numpy(), x_rec[0,0,:,:].detach().cpu().numpy())
    print(f"Get PSNR = {psnr:.4f}dB")

    if psnr > best_psnr:
        best_psnr = psnr
        best_alpha = alpha.item()
        best_step_size = step_size.item()

print("Finished random search. Best: ")
print("\t PSNR: ", best_psnr)
print("\t alpha: ", best_alpha)
print("\t step size: ", best_step_size) 

#print(f"Final PSNR = {psnr_list[-1]:4f} dB")
#fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(12,6))
#ax1.imshow(x[0,0].cpu().numpy(), cmap="gray")
#ax2.imshow(xk[0,0].cpu().numpy(), cmap="gray")
#ax3.semilogy(res_list)
#ax3.set_title("Mean Squared Error")
#ax4.plot(psnr_list)
#ax4.set_title("PSNR")
#plt.show()