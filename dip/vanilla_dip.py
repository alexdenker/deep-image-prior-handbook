import torch 
import matplotlib.pyplot as plt 
import numpy as np 
from tqdm import tqdm 
from PIL import Image 
from skimage.metrics import peak_signal_noise_ratio

from ..utils import create_circular_mask

class DeepImagePrior():
    def __init__(self, model, lr, num_steps, noise_std, L=1.0):
        self.model = model 

        self.lr = lr
        self.num_steps = num_steps
        self.noise_std = noise_std # add additional random noise to input 
        self.L = L # estimate of the Lipschitz constant of the forward operator 

    def train(self, ray_trafo, y, x_in, x_gt=None, return_metrics=True):
        """
        Training the DIP.

        y: measurements 
        x_in: input to DIP
        
        """

        mask = create_circular_mask((501, 501))

        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
                
        psnr_list = [] 
        loss_list = [] 

        best_psnr = 0 
        best_psnr_idx = 0 
        best_psnr_image = None 

        self.model.train()
        for i in tqdm(range(self.num_steps)):
            optim.zero_grad()

            if self.noise_std > 0:
                x_pred = self.model(x_in + self.noise_std**2*torch.randn_like(x_in))
            else:
                x_pred = self.model(x_in)

            loss = torch.sum((ray_trafo.trafo(x_pred) - y)**2/self.L**2) 
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optim.step() 

            loss_list.append(loss.item())
            if x_gt is not None:
                psnr_list.append(peak_signal_noise_ratio(x_gt[0,0,mask].cpu().numpy(), 
                                                         x_pred[0,0,mask].detach().cpu().numpy(), 
                                                         data_range=x_gt[0,0,mask].cpu().numpy().max()))
            else:
                psnr_list.append(0)

            if psnr_list[-1] > best_psnr:
                best_psnr = psnr_list[-1]
                best_psnr_idx = i 
                best_psnr_image = torch.clone(x_pred.detach().cpu()).numpy()

        self.model.eval()
        with torch.no_grad():
            x_out = self.model(x_in)

        if return_metrics:
            return x_out, psnr_list, loss_list, best_psnr_image, best_psnr_idx
        else:
            return x_out


