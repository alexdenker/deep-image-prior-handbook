import yaml 
from types import SimpleNamespace

import numpy as np 
import torch
import matplotlib.pyplot as plt 
from tqdm import tqdm

import deepinv as dinv 
from skimage.metrics import peak_signal_noise_ratio

from dip import get_walnut_data, get_walnut_2d_ray_trafo, create_circular_mask, dict_to_namespace, power_iteration

device = "cuda"
phantom = "shepplogan" # "shepplogan" or "walnut"
print("Running TV reconstruction for phantom: ", phantom)
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


if phantom == "walnut":
    data = get_walnut_data(cfg, ray_trafo=ray_trafo)

    y, x, x_fbp = data[0]

elif phantom == "shepplogan":
    from skimage.data import shepp_logan_phantom
    sl = shepp_logan_phantom() 
    from skimage.transform import resize

    #x = resize(sl, (501, 501))
    pad_top = 50
    pad_bottom = 51
    pad_left = 50
    pad_right = 51

    x = np.pad(sl, 
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant', 
            constant_values=0)

    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float().to(cfg.device)
    print(x.shape)
    y = ray_trafo.trafo(x) 
    g = torch.Generator(device=y.device).manual_seed(1234)
    y = y +0.01 * torch.mean(y.abs()) * torch.randn(y.shape, generator=g, device=y.device)
    x_fbp = ray_trafo.fbp(y)
else:
    raise NotImplementedError

im_size = x.shape[-1]
x_test = torch.rand_like(x).view(-1, 1)
L = power_iteration(ray_trafo, x_test)

max_iter = 1000 
step_size = 1.0
tol = 1e-6

def TV_rec(y, L, x_init, step_size, alpha, max_iter=2000, tol=1e-6, track_psnr=True, x_gt=None):
    if track_psnr and x_gt is None:
        raise ValueError("x_gt must be provided when track_psnr=True")

    
    xk = torch.clone(x_init)
    x_prev = torch.clone(xk)

    prior =  dinv.optim.prior.TVPrior(n_it_max=100)

    # FISTA
    tk = torch.tensor(1.0)
    psnr_list = []
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

        if track_psnr:
            psnr = peak_signal_noise_ratio(x[0,0,mask].cpu().numpy(), xk[0,0,mask].detach().cpu().numpy(), data_range=x[0,0,mask].cpu().numpy().max())
            psnr_list.append(psnr)

        if torch.sum((xk - x_prev)**2).item() < tol:
            break

    if track_psnr:
        return xk, psnr_list 
    else:
        return xk 

for alpha in [1e-3, 1e-2, 0.1, 0.2, 0.5]:

    x_init = torch.zeros_like(x)
    x_rec, psnr_list = TV_rec(y, L, x_init, step_size, alpha, max_iter=max_iter, tol=tol, track_psnr=True, x_gt=x)

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))
    ax1.plot(psnr_list)
    ax1.set_title(f"alpha={alpha}")
    ax2.imshow(x_rec[0,0].cpu().numpy(), cmap="gray", interpolation=None)
    ax2.set_title(f"alpha={alpha}")
    plt.savefig(f"TV_for_alpha={alpha}.png")
    plt.close()
    psnr = peak_signal_noise_ratio(x[0,0,mask].cpu().numpy(), x_rec[0,0,mask].detach().cpu().numpy(), data_range=x[0,0,mask].cpu().numpy().max())
    print(f"Get PSNR = {psnr:.4f}dB with alpha={alpha}")
    # also add peak the PSNR and iteration 
    peak_psnr = max(psnr_list)
    peak_iter = np.argmax(psnr_list)
    print("Peak PSNR: ", peak_psnr, " at iteration: ", peak_iter)