
import torch 
import torch.nn as nn 
from skimage.metrics import peak_signal_noise_ratio


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


class ScaleModule(nn.Module):

    """
    This class provides methods for normalizing the input data.

    Based on the scaling_module in
    https://github.com/ahendriksen/msd_pytorch/blob/master/msd_pytorch/msd_model.py
    """

    def __init__(self, num_channels, mean=0., std=1., conv3d=False):
        super().__init__()

        """
        Parameters
        ----------
        num_channels: int
            The number of channels.
        mean: float
            Mean of values.
        std: float
            Standard deviation of values.
        param conv3d: bool
            Indicates that the input data is 3D instead of 2D.

        * saved when the network is saved;
        * not updated by the gradient descent solvers.
        """
        self.mean = mean
        self.std = std

        self.scale_layer = nn.Conv2d(num_channels, num_channels, 1)

        self._scaling_module_set_scale(1 / self.std)
        self._scaling_module_set_bias(-self.mean / self.std)

    def _scaling_module_set_scale(self, scale):

        self.scale_layer.weight.requires_grad = False
        c_out, c_in = self.scale_layer.weight.shape[:2]
        assert c_out == c_in
        self.scale_layer.weight.data.zero_()
        for i in range(c_out):
            self.scale_layer.weight.data[i, i] = scale

    def _scaling_module_set_bias(self, bias):
        self.scale_layer.bias.requires_grad = False
        self.scale_layer.bias.data.fill_(bias)

    def forward(self, x):
        return self.scale_layer(x)

def get_scale_modules(ch_in, ch_out, mean_in=0., mean_out=0., std_in=1.,
                     std_out=1.):
    scale_in = ScaleModule(ch_in, mean_in, std_in)
    scale_out = ScaleModule(ch_out, -mean_out, 1./std_out)
    return scale_in, scale_out



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

class MaskedPSNR:
    def __init__(self, im_size, mask_fn=None):
        """
        Args:
            im_size (int): The image size (assumes square images).
            mask_fn (callable): A function that returns a boolean mask array.
        """
        self.im_size = im_size
        if mask_fn is None:
            mask_fn = create_circular_mask
        self.mask = mask_fn((im_size, im_size))

    def __call__(self, x, x_pred):
        """
        Args:
            x (torch.Tensor): Ground truth image tensor, shape (B, C, H, W)
            x_pred (torch.Tensor): Predicted image tensor, same shape as x

        Returns:
            float: PSNR computed only within the masked region
        """
        # Use the first image and first channel (assumes grayscale)
        x_masked = x[0, 0, self.mask].detach().cpu().numpy()
        x_pred_masked = x_pred[0, 0, self.mask].detach().cpu().numpy()

        data_range = x_masked.max()  # Can also use x_masked.max() - x_masked.min() if needed
        return peak_signal_noise_ratio(x_masked, x_pred_masked, data_range=data_range)
