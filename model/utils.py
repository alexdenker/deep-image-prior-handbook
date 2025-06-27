
import torch 

def tv_loss(x):
    """
    Anisotropic TV loss similar to the one in [1]_.
    From: https://github.com/jleuschn/dival/blob/master/dival/util/torch_losses.py
    Parameters
    ----------
    x : :class:`torch.Tensor`
        Tensor of which to compute the anisotropic TV w.r.t. its last two axes.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Total_variation_denoising
    """
    dh = torch.abs(x[..., :, 1:] - x[..., :, :-1])
    dw = torch.abs(x[..., 1:, :] - x[..., :-1, :])
    return torch.sum(dh[..., :-1, :] + dw[..., :, :-1])


def isotropic_tv_loss(x):
    """
    Isotropic TV loss.
    Parameters
    ----------
    x : :class:`torch.Tensor`
        Tensor of which to compute the isotropic TV w.r.t. its last two axes.

    """
    dh = x[..., :, 1:] - x[..., :, :-1]
    dw = x[..., 1:, :] - x[..., :-1, :]
    
    dw = torch.nn.functional.pad(dw, (0, 0, 0, 1))  # Pad height dimension
    dh = torch.nn.functional.pad(dh, (0, 1, 0, 0))  # Pad width dimension
    return torch.sqrt(dh**2 + dw**2 + 1e-6).sum()
