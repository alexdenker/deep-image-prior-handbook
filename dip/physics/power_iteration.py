import torch 

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
