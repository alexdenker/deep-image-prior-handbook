"""

fastMRI dataset from https://github.com/facebookresearch/fastMRI/blob/main/fastmri/data/mri_data.py

"""

import os 
import torch 
import h5py
import xml.etree.ElementTree as etree
import numpy as np 
from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Union


def et_query(
    root: etree.Element,
    qlist,
    namespace: str = "http://www.ismrm.org/ISMRMRD",
) -> str:
    """
    ElementTree query function.

    This can be used to query an xml document via ElementTree. It uses qlist
    for nested queries.

    Args:
        root: Root of the xml to search through.
        qlist: A list of strings for nested searches, e.g. ["Encoding",
            "matrixSize"]
        namespace: Optional; xml namespace to prepend query.

    Returns:
        The retrieved data as a string.
    """
    s = "."
    prefix = "ismrmrd_namespace"

    ns = {prefix: namespace}

    for el in qlist:
        s = s + f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)


def complex_center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    if not (0 < shape[0] <= data.shape[-1] and 0 < shape[1] <= data.shape[-2]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]



class FastMRISingleCoil(torch.utils.data.Dataset):
    def __init__(self, root, file):
        self.root = root 
        self.file = file 

        self.metadata, self.num_slices = self._retrieve_metadata(os.path.join(self.root, self.file))
        print(self.metadata)
        recons_key = "reconstruction_esc" # single coild 
        with h5py.File(os.path.join(self.root, self.file), "r") as hf:
            print(hf.keys())
            self.kspace = torch.from_numpy(hf["kspace"][()])
            self.target = torch.from_numpy(hf[recons_key][()])
        
        #import matplotlib.pyplot as plt 
        #plt.figure()
        #plt.imshow(target[25])
        #plt.colorbar()
        #plt.show()

    def _retrieve_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            et_root = etree.fromstring(hf["ismrmrd_header"][()])

            enc = ["encoding", "encodedSpace", "matrixSize"]
            enc_size = (
                int(et_query(et_root, enc + ["x"])),
                int(et_query(et_root, enc + ["y"])),
                int(et_query(et_root, enc + ["z"])),
            )
            rec = ["encoding", "reconSpace", "matrixSize"]
            recon_size = (
                int(et_query(et_root, rec + ["x"])),
                int(et_query(et_root, rec + ["y"])),
                int(et_query(et_root, rec + ["z"])),
            )

            lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
            enc_limits_center = int(et_query(et_root, lims + ["center"]))
            enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1

            padding_left = enc_size[1] // 2 - enc_limits_center
            padding_right = padding_left + enc_limits_max

            num_slices = hf["kspace"].shape[0]

            metadata = {
                "padding_left": padding_left,
                "padding_right": padding_right,
                "encoding_size": enc_size,
                "recon_size": recon_size,
                **hf.attrs,
            }

        return metadata, num_slices

    def __len__(self):
        return len(self.num_slices)
    
    def __getitem__(self, i: int):
        

        return self.target[i].unsqueeze(0), self.kspace[i].unsqueeze(0)


if __name__ == "__main__":
    import matplotlib.pyplot as plt 
    dataset = FastMRISingleCoil(root="data/singlecoil_val", file="file1000000.h5")

    x, y = dataset[18]

    print("x: ", x.shape, " y: ", y.shape)
    kspace = torch.fft.ifftshift(y, dim=(-2, -1))
    xhat = torch.fft.ifft2(kspace, norm="ortho")
    xhat = torch.fft.ifftshift(xhat, dim=(-2, -1))
    print("xhat before cutting: ", xhat.shape)
    xhat = complex_center_crop(xhat, (320,320))
    print(xhat.shape)
    
    y_resim = torch.fft.fft2(x.abs(), s=[640,368], norm="ortho")
    y_resim = torch.fft.fftshift(y_resim, dim=(-2,-1))
    print("y_resim: ", y_resim.shape)

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))

    im = ax1.imshow(torch.log(y_resim[0].abs()))
    fig.colorbar(im, ax=ax1)
    ax1.set_title("simulated")

    im = ax2.imshow(torch.log(y[0].abs()))
    fig.colorbar(im, ax=ax2)
    ax2.set_title("real data")
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))

    im = ax1.imshow(x[0])
    fig.colorbar(im, ax=ax1)

    im = ax2.imshow(torch.abs(xhat[0]))
    fig.colorbar(im, ax=ax2)

    plt.show()