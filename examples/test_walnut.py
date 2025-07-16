
import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import matplotlib.pyplot as plt 

import yaml 
from types import SimpleNamespace

from dataset.walnut import get_walnut_data
from dataset.walnut_2d_ray_trafo import get_walnut_2d_ray_trafo

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    else:
        return d
    
cfg_dict = {}
with open('configs/walnut_config.yaml', 'r') as f:
    data = yaml.safe_load(f)
    cfg_dict["data"] = data


cfg = dict_to_namespace(cfg_dict)
cfg.device = "cuda"
#cfg.data.angular_sub_sampling = 10 
print(cfg.data)

ray_trafo = get_walnut_2d_ray_trafo(
    data_path=cfg.data.data_path,
    matrix_path=cfg.data.data_path,
    walnut_id=cfg.data.walnut_id,
    orbit_id=cfg.data.orbit_id,
    angular_sub_sampling=cfg.data.angular_sub_sampling,
    proj_col_sub_sampling=cfg.data.proj_col_sub_sampling)

data = get_walnut_data(cfg, ray_trafo=ray_trafo)

y, x, xfbp = data[0]
print(y.shape, x.shape, xfbp.shape)

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(13,6))

ax1.imshow(x[0,0].cpu().numpy(),cmap="gray", interpolation=None)

ax2.imshow(xfbp[0,0].cpu().numpy(),cmap="gray", interpolation=None)

plt.show()