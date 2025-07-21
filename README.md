# Deep Image Prior for Image Reconstruction 

This repository contains the code for the chapter "Deep Image Prior for Image Reconstruction" (name might change later). 

We analyse the performance of different modifications and extensions of the DIP for image reconstruction of $\mu$-CT measurements of a Walnut. 

The final repository will contain:
- Vanilla DIP: The standard DIP, without any additional extension. See [vanilla dip](vanilla_dip_walnut.py)
- DIP+TV: The DIP with an additional total variation regularisation. Implementation from [Baguer et al. (2020)](https://arxiv.org/abs/2003.04989), i.e., simply adding TV to the objective function and relying on autodiff for optimisation. See *TODO*
- DIP+TV HQS: The DIP with additional total variation regularisation. Implemented using Half-Quadratic-Splitting. See [dip tv hqs](dip_tv_hqs)
- Self Guidance DIP: TODO
- aSeqDIP: TODO
- eDIP: TODO


## $\mu$-CT Walnut 

The forward operator can be constructed as a sparse matrix by running 

```
python create_walnut_ray_trafo_matrix.py
```

This construction can take up to 45min. You can also download it [here](https://zenodo.org/record/7282279/files/single_slice_ray_trafo_matrix_walnut1_orbit2_ass20_css6.mat?download=1).