"""
Provides the EllipsesDataset.
From https://github.com/educating-dip/subspace_dip_learning/blob/main/subspace_dip/data/datasets/ellipses.py
"""

from typing import Union, Iterator, Tuple
import numpy as np
import torch
from torch import Tensor
from itertools import repeat
from typing import Any, Optional, Sequence, Iterable
from .base_ray_trafo import BaseRayTrafo


def simulate(x: Tensor, ray_trafo: BaseRayTrafo, white_noise_rel_stddev: float,
        rng: Optional[np.random.Generator] = None):
    """
    Compute ``observation = ray_trafo(x)`` and add white noise with standard
    deviation ``white_noise_rel_stddev * mean(abs(observation))``.

    Parameters
    ----------
    x : :class:`torch.Tensor`
        Image, passed to `ray_trafo`.
    ray_trafo : callable
        Function computing the noise-free observation.
    white_noise_rel_stddev : float
        Relative standard deviation of the noise that is added.
    rng : :class:`np.random.Generator`, optional
        Random number generator. If `None` (the default),
        a new generator ``np.random.default_rng()`` is used.
    """

    observation = ray_trafo(x)

    if rng is None:
        rng = np.random.default_rng()
    noise = torch.from_numpy(rng.normal(
            scale=white_noise_rel_stddev * torch.mean(torch.abs(observation)).item(),
            size=observation.shape)).to(
                    dtype=observation.dtype, device=observation.device)

    noisy_observation = observation + noise

    return noisy_observation


class SimulatedDataset(torch.utils.data.Dataset):
    """
    CT dataset simulated from provided ground truth images.

    Each item of this dataset is a tuple ``noisy_observation, x, filtbackproj``,
    where
        * `noisy_observation = ray_trafo(x) + noise``
          (shape: ``(1,) + obs_shape``)
        * `x` is the ground truth (label)
          (shape: ``(1,) + im_shape``)
        * ``filtbackproj = FBP(noisy_observation)``
          (shape: ``(1,) + im_shape``)
    """

    def __init__(self,
            image_dataset: Union[Sequence[Tensor], Iterable[Tensor]],
            ray_trafo: BaseRayTrafo,
            white_noise_rel_stddev: float,
            use_fixed_seeds_starting_from: Optional[int] = 1,
            rng: Optional[np.random.Generator] = None,
            device: Optional[Any] = None):
        """
        Parameters
        ----------
        image_dataset : sequence or iterable
            Image data. The methods :meth:`__len__` and :meth:`__getitem__`
            directly use the respective functions of `image_dataset` and will
            fail if they are not supported. The method :meth:`__iter__` simply
            iterates over `image_dataset` and thus will only stop when
            `image_dataset` is exhausted.
        ray_trafo : :class:`bayes_dip.data.BaseRayTrafo`
            Ray trafo.
        white_noise_rel_stddev : float
            Relative standard deviation of the noise that is added.
        use_fixed_seeds_starting_from : int, optional
            If an int, the fixed random seed
            ``use_fixed_seeds_starting_from + idx`` is used for sample `idx`.
            Must be `None` if a custom `rng` is used.
            The default is `1`.
        rng : :class:`np.random.Generator`, optional
            Custom random number generator used to simulate noise; it will be
            advanced every time an item is accessed.
            Cannot be combined with `use_fixed_seeds_starting_from`.
            If both `rng` and `use_fixed_seeds_starting_from` are `None`,
            a new generator ``np.random.default_rng()`` is used.
        device : str or torch.device, optional
            If specified, data will be moved to the device. `ray_trafo`
            (including `ray_trafo.fbp`) must support tensors on the device.
        """
        super().__init__()

        self.image_dataset = image_dataset
        self.ray_trafo = ray_trafo
        self.white_noise_rel_stddev = white_noise_rel_stddev
        if rng is not None:
            assert use_fixed_seeds_starting_from is None, (
                    'must not use fixed seeds when passing a custom rng')
        self.rng = rng
        self.use_fixed_seeds_starting_from = use_fixed_seeds_starting_from
        self.device = device

    def __len__(self) -> Union[int, float]:
        return len(self.image_dataset)

    def _generate_item(self, idx: int, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:

        if self.rng is None:
            seed = (self.use_fixed_seeds_starting_from + idx
                    if self.use_fixed_seeds_starting_from is not None else None)
            rng = np.random.default_rng(seed)
        else:
            rng = self.rng

        x = x.to(device=self.device)
        noisy_observation = simulate(x[None],
                ray_trafo=self.ray_trafo,
                white_noise_rel_stddev=self.white_noise_rel_stddev,
                rng=rng)[0].to(device=self.device)
        filtbackproj = self.ray_trafo.fbp(noisy_observation[None])[0].to(
                device=self.device)

        return noisy_observation, x, filtbackproj

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor, Tensor]]:
        for idx, x in enumerate(self.image_dataset):
            yield self._generate_item(idx, x)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        return self._generate_item(idx, self.image_dataset[idx])

class EllipsesDataset(torch.utils.data.IterableDataset):
    """
    Dataset with images of multiple random ellipses.
    Creates images by rasterizing random ellipses. The images are normalized 
    to have a value range of ``[0., 1.]`` with a background value of ``0.``.
    """

    def __init__(
        self,
        shape: Tuple[int, int] = (128, 128),
        length: int = 3200,
        fixed_seed: int = 1,
        fold: str = "train",
        max_n_ellipse: int = 70,
    ):
        self.shape = shape
        self.length = length
        self.max_n_ellipse = max_n_ellipse
        self.ellipses_data = []
        self.setup_fold(fixed_seed=fixed_seed, fold=fold)

        # Create coordinate grids
        h, w = self.shape
        yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32), 
                                 torch.arange(w, dtype=torch.float32), indexing='ij')
        self.yy = yy
        self.xx = xx
        super().__init__()

    def setup_fold(self, fixed_seed: int = 1, fold: str = "train"):
        fixed_seed = None if fixed_seed in [False, None] else int(fixed_seed)
        if (fixed_seed is not None) and (fold == "validation"):
            fixed_seed = fixed_seed + 1
        self.rng = np.random.RandomState(fixed_seed)

    def __len__(self) -> Union[int, float]:
        return self.length if self.length is not None else float("inf")

    def _extend_ellipses_data(self, min_length: int) -> None:
        ellipsoids = np.empty((self.max_n_ellipse, 6))
        n_to_generate = max(min_length - len(self.ellipses_data), 0)
        for _ in range(n_to_generate):
            v = self.rng.uniform(-0.4, 1.0, (self.max_n_ellipse,))
            a1 = 0.2 * self.rng.exponential(1.0, (self.max_n_ellipse,))
            a2 = 0.2 * self.rng.exponential(1.0, (self.max_n_ellipse,))
            x = self.rng.uniform(-0.9, 0.9, (self.max_n_ellipse,))
            y = self.rng.uniform(-0.9, 0.9, (self.max_n_ellipse,))
            rot = self.rng.uniform(0.0, 2 * np.pi, (self.max_n_ellipse,))
            n_ellipse = min(self.rng.poisson(self.max_n_ellipse), self.max_n_ellipse)
            v[n_ellipse:] = 0.0
            ellipsoids = np.stack((v, a1, a2, x, y, rot), axis=1)
            self.ellipses_data.append(ellipsoids)

    def _generate_item(self, idx: int) -> Tensor:
        """Rasterize ellipses onto an image using pure PyTorch."""
        ellipsoids = self.ellipses_data[idx]
        h, w = self.shape
        
        image = torch.zeros((h, w), dtype=torch.float32)
        
        # Center coordinates
        center_h, center_w = h / 2.0, w / 2.0
        
        for ellipse_params in ellipsoids:
            v, a1, a2, x, y, rot = ellipse_params
            
            # Skip if intensity is zero or very small
            if v <= 0.0:
                continue
            
            # Convert normalized coords to pixel coords
            center_x = center_w + float(x) * center_w
            center_y = center_h + float(y) * center_h
            
            # Semi-axes in pixels
            axis_a = max(1.0, float(a1) * center_h)
            axis_b = max(1.0, float(a2) * center_h)
            
            # Rotation angle
            angle = float(rot)
            
            # Translate coordinates relative to ellipse center
            dx = self.xx - center_x
            dy = self.yy - center_y

            # Rotate coordinates
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            dx_rot = dx * cos_a + dy * sin_a
            dy_rot = -dx * sin_a + dy * cos_a
            
            # Ellipse equation: (x/a)^2 + (y/b)^2 <= 1
            ellipse_mask = (dx_rot ** 2 / (axis_a ** 2) + 
                           dy_rot ** 2 / (axis_b ** 2)) <= 1.0
            
            # Add ellipse contribution (soft blending for overlaps)
            image = torch.where(ellipse_mask, 
                               image + float(v), 
                               image)
        
        # Normalize to [0, 1]
        max_val = torch.max(image)
        if max_val > 0:
            image = image / max_val
        
        return image[None].float()  # add channel dim

    def __iter__(self) -> Iterator[Tensor]:
        it = repeat(None, self.length) if self.length is not None else repeat(None)
        for idx, _ in enumerate(it):
            self._extend_ellipses_data(idx + 1)
            yield self._generate_item(idx)

    def __getitem__(self, idx: int) -> Tensor:
        self._extend_ellipses_data(idx + 1)
        return self._generate_item(idx)



def get_ellipses_dataset(
    fold: str = "train",
    im_size: int = 128,
    length: int = 3200,
    max_n_ellipse: int = 70,
    device=None,
) -> EllipsesDataset:
    image_dataset = EllipsesDataset(
        (im_size, im_size), length=length, fold=fold, max_n_ellipse=max_n_ellipse
    )

    return image_dataset


class DiskDistributedEllipsesDataset(EllipsesDataset):
    def __init__(
        self,
        shape: Tuple[int, int] = (128, 128),
        length: int = 3200,
        fixed_seed: int = 1,
        fold: str = "train",
        diameter: float = 0.4745,
        max_n_ellipse: int = 70,
    ):
        super().__init__(
            shape=shape,
            length=length,
            fixed_seed=fixed_seed,
            fold=fold,
            max_n_ellipse=max_n_ellipse,
        )
        self.diameter = diameter

    def _extend_ellipses_data(self, min_length: int) -> None:
        ellipsoids = np.empty((self.max_n_ellipse, 6))
        n_to_generate = max(min_length - len(self.ellipses_data), 0)
        for _ in range(n_to_generate):
            v = self.rng.uniform(-0.4, 1.0, (self.max_n_ellipse,))
            a1 = 0.2 * self.diameter * self.rng.exponential(1.0, (self.max_n_ellipse,))
            a2 = 0.2 * self.diameter * self.rng.exponential(1.0, (self.max_n_ellipse,))

            c_r = self.rng.triangular(
                0.0, self.diameter, self.diameter, size=(self.max_n_ellipse,)
            )
            c_a = self.rng.uniform(0.0, 2 * np.pi, (self.max_n_ellipse,))
            x = np.cos(c_a) * c_r
            y = np.sin(c_a) * c_r
            rot = self.rng.uniform(0.0, 2 * np.pi, (self.max_n_ellipse,))
            n_ellipse = min(self.rng.poisson(self.max_n_ellipse), self.max_n_ellipse)
            v[n_ellipse:] = 0.0
            ellipsoids = np.stack((v, a1, a2, x, y, rot), axis=1)
            self.ellipses_data.append(ellipsoids)


def get_disk_dist_ellipses_dataset(
    ray_trafo: BaseRayTrafo,
    fold: str = "train",
    im_size: int = 128,
    length: int = 3200,
    diameter: float = 0.4745,
    max_n_ellipse: int = 70,
    white_noise_rel_stddev : float = .05, 
    use_fixed_seeds_starting_from : int = 1, 
    device=None,
) -> DiskDistributedEllipsesDataset:
    image_dataset = DiskDistributedEllipsesDataset(
        (im_size, im_size),
        **{"length": length, "fold": fold},
        diameter=diameter,
        max_n_ellipse=max_n_ellipse,
    )
    return SimulatedDataset(
        image_dataset=image_dataset, 
        ray_trafo=ray_trafo,
        white_noise_rel_stddev=white_noise_rel_stddev,
        use_fixed_seeds_starting_from=use_fixed_seeds_starting_from,
        device=device,
    )
