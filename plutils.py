from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

import torch
import numpy as np

# ////////////////////////////////////////////////////////
# //                    Plotting                        //
# ////////////////////////////////////////////////////////

title_kwargs = {
    "fontsize": 18,
    "fontname": "DejaVu Sans",
    "pad": 2              
}

def to_numpy(img):
    if isinstance(img, torch.Tensor):
        return img.detach().cpu().numpy()
    try:
        import cupy as cp
        if isinstance(img, cp.ndarray):
            return cp.asnumpy(img)
    except ImportError:
        pass
    return np.array(img)

def plot_nimages(images, titles=None, save_path=None, show_colorbar=False, layout=None, suptitle=None, cmap='gray'):
    """
    Plot images with flexible layout:
      - If layout=[rows, cols] is provided, use that.
      - If images is a list of tensors, each tensor is treated as a separate image.
      - Otherwise: single row if <=3 images, golden-ratio grid if >3.
      - Titles behavior:
          * If len(titles) == total number of images, use one per image.
          * If len(titles) == rows, titles[row] + f"_{col}" is used.
    """
    # Check if images is a list of lists
    images_flat = []
    for img in images:
        if len(img) == 1:
            images_flat.append(to_numpy(img))
        elif hasattr(img, "ndim") and img.ndim == 4:
            images_flat.extend([to_numpy(im) for im in img])
        else:
            images_flat.extend([to_numpy(im) for im in img])
    n = len(images_flat)
        # Layout logic
    if layout is not None:
        rows, cols = layout
    elif n <= 3:
        rows, cols = 1, n
    else:
        golden_ratio = 1.618
        cols = int(np.floor(np.sqrt(n * golden_ratio)))
        rows = int(np.ceil(n / cols))

    # Default titles
    if titles is None:
        titles = [f'Image {i+1}' for i in range(n)]

    fig = plt.figure(figsize=(5 * cols, 5 * rows))
    gs = GridSpec(rows, cols, figure=fig)

    for i, img in enumerate(images_flat):
        row, col = divmod(i, cols)
        ax = fig.add_subplot(gs[row, col])

        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)
        if img.ndim == 2:
            im = ax.imshow(img, cmap=cmap)
        elif img.ndim == 3 and img.shape[0] in [1, 3]:  # channels first
            img = np.transpose(img, (1, 2, 0))
            im = ax.imshow(img.squeeze() if img.shape[2] == 1 else img, cmap=cmap)
        elif img.ndim == 3 and img.shape[2] in [1, 3]:  # channels last
            im = ax.imshow(img.squeeze() if img.shape[2] == 1 else img, cmap=cmap)
        elif img.ndim == 1:
            img_size = int(np.sqrt(img.shape[0]))
            img = img.reshape((img_size, img_size))
            im = ax.imshow(img, cmap=cmap)
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")

        # Title logic
        if len(titles) == n:
            ax.set_title(titles[i], **title_kwargs)
        elif len(titles) == rows:
            ax.set_title(f"{titles[row]} {col+1}", **title_kwargs)
        else:
            ax.set_title(f"Image {i+1}", **title_kwargs)

        ax.axis('off')
        if show_colorbar:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if suptitle:
        plt.suptitle(suptitle, fontsize=20)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
    else:
        plt.show()