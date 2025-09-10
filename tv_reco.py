import yaml 
from types import SimpleNamespace

import torch
import matplotlib.pyplot as plt 
from tqdm import tqdm

import deepinv as dinv 
from skimage.metrics import peak_signal_noise_ratio

from dip import get_walnut_data, get_walnut_2d_ray_trafo, create_circular_mask, dict_to_namespace, power_iteration

device = "cuda"


mask = create_circular_mask((501, 501))

cfg_dict = {}
with open('configs/walnut_config.yaml', 'r') as f:
    data = yaml.safe_load(f)
    cfg_dict["data"] = data

cfg = dict_to_namespace(cfg_dict)
cfg.device = device

ray_trafo = get_walnut_2d_ray_trafo(
    data_path=cfg.data.data_path,
    matrix_path=cfg.data.data_path,
    walnut_id=cfg.data.walnut_id,
    orbit_id=cfg.data.orbit_id,
    angular_sub_sampling=cfg.data.angular_sub_sampling,
    proj_col_sub_sampling=cfg.data.proj_col_sub_sampling)
ray_trafo.to(device)
data = get_walnut_data(cfg, ray_trafo=ray_trafo)

y, x, xfbp = data[0]
#y = y[0,0,0,:].unsqueeze(-1)
im_size = x.shape[-1]
print(y.shape, x.shape, xfbp.shape)

x_test = torch.rand_like(x).view(-1, 1)
print("x_test: ", x_test.shape)

L = power_iteration(ray_trafo, x_test)
print("L: ", L)


max_iter = 1000 
step_size =  1.0
tol = 1e-6

def TV_rec(y, L, x_init, step_size, alpha, max_iter=2000, tol=1e-6):
    xk = torch.clone(x_init)
    x_prev = torch.clone(xk)

    prior =  dinv.optim.prior.TVPrior(n_it_max=100)

    # FISTA
    tk = torch.tensor(1.0)
    for _ in tqdm(range(max_iter)):
        tk_new = (1 + torch.sqrt(1 + 4 * tk**2))/2
        ak = (tk -1) / tk_new

        xk_tilde = xk + ak * (xk - x_prev)
        
        Ax = ray_trafo.trafo(xk_tilde)
        res =  Ax - y 
        x_next = xk_tilde - step_size/L * ray_trafo.trafo_adjoint(res)
        x_next = prior.prox(x_next, gamma=step_size/L*alpha)
        x_next[x_next < 0] = 0.
        x_next[x_next > 1] = 1.

        x_prev = xk 
        xk = x_next
        tk = tk_new

        if torch.sum((xk - x_prev)**2).item() < tol:
            break
        
    return xk 




alpha = 0.1
x_init = torch.zeros_like(x)
x_rec = TV_rec(y, L, x_init, step_size, alpha, max_iter=max_iter, tol=tol)

psnr = peak_signal_noise_ratio(x[0,0,mask].cpu().numpy(), x_rec[0,0,mask].detach().cpu().numpy(), data_range=x[0,0,mask].cpu().numpy().max())
print(f"Get PSNR = {psnr:.4f}dB")



fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(12,6))
ax1.imshow(x[0,0].cpu().numpy(), cmap="gray", interpolation=None)
ax2.imshow(x_rec[0,0].cpu().numpy(), cmap="gray", interpolation=None)
ax3.imshow(xfbp[0,0].cpu().numpy(),cmap="gray", interpolation=None)

plt.show()