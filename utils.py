
from types import SimpleNamespace
import torch

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    else:
        return d
    

def create_circular_mask(size):
    """
    The output of this function is a torch tensor of size (size, size) with binary values:
        1: point is inside a circle of radius size/2
        0: point is outside a circle of radius size/2

    This method is used to only calculate the quality metrics inside of the circle.  
    
    """

    H, W = size
    Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    center = (H // 2, W // 2)
    radius = min(center[0], center[1])

    dist = (X - center[1])**2 + (Y - center[0])**2
    mask = (dist <= radius**2)
    return mask  # shape: (501, 501), values: 0 or 1


def power_iteration(ray_trafo, x0, max_iter=100,verbose=True, tol=1e-6):
    """
    Estimate the Lipschitz constant of the ray_trafo
    
    """
    x = torch.randn_like(x0)
    x /= torch.norm(x)
    zold = torch.zeros_like(x)
    for it in range(max_iter):
        y = ray_trafo.trafo_flat(x)
        y = ray_trafo.trafo_adjoint_flat(y)
        z = torch.matmul(x.conj().reshape(-1), y.reshape(-1)) / torch.norm(x) ** 2

        rel_var = torch.norm(z - zold)
        if rel_var < tol and verbose:
            print(
                f"Power iteration converged at iteration {it}, value={z.item():.2f}"
            )
            break
        zold = z
        x = y / torch.norm(y)

    return z.real