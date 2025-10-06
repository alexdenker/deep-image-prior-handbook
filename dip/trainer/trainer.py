"""
Provides :class:`ParameterSampler`.
"""
from typing import Dict, List, Optional, Any

import os
import torch
import numpy as np
import torch.nn as nn

from torch import Tensor
from copy import deepcopy
from math import ceil
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CyclicLR, OneCycleLR

from .utils import get_params_from_nn_module, PSNR, get_original_cwd

class ParameterSampler:
    """
    Wrapper for constructing a low-dimensional subspace of the NN optimisation trajectory.
    """
    def __init__(self, 
        model: nn.Module,
        dataset: Dict[str, torch.utils.data.Dataset],
        exclude_norm_layers: bool = False, 
        include_bias: bool = True, 
        device: Optional[Any] = None, 
        ):
        
        self.model = model
        self.dataset = dataset
        self.exclude_norm_layers = exclude_norm_layers
        self.include_bias = include_bias
        self.device = device or torch.device(
            ('cuda:0' if torch.cuda.is_available() else 'cpu')
        )
        self.parameters_samples = []

    def add_parameters_samples(self, use_cpu: bool = True) -> List[Tensor]:
    
        parameter_vec = get_params_from_nn_module(
                self.model,
                exclude_norm_layers=self.exclude_norm_layers,
                include_bias=self.include_bias)
        self.parameters_samples.append(
            parameter_vec if not use_cpu else parameter_vec.cpu()
        )
    
    def create_sampling_sequence(self, 
        burn_in: int, 
        num_overall_updates : int, 
        num_samples : int, 
        sampling_strategy : str = 'linear'
        ):
        if not isinstance(sampling_strategy, str) or sampling_strategy not in ['linear', 'random', 'power']:
            sampling_strategy = 'linear'

        if sampling_strategy == 'linear':
            self.sampling_sequence =  np.linspace(
                burn_in,
                num_overall_updates,
                num_samples + 1,
                dtype=int
                )
        else: 
            raise NotImplementedError
        
    def sample(
        self,
        optim_kwargs: Dict,
        save_samples: bool = False
        ):

        if optim_kwargs['torch_manual_seed']:
            torch.random.manual_seed(optim_kwargs['torch_manual_seed'])

        # create PyTorch datasets
        criterion = torch.nn.MSELoss()
        self.init_optimizer(optim_kwargs=optim_kwargs)

        # create PyTorch dataloaders
        data_loaders = {
            'train': DataLoader(
                self.dataset['train'],
                batch_size=optim_kwargs['batch_size'],
                shuffle=True
            ),
            'validation': DataLoader(
                self.dataset['validation'],
                batch_size=optim_kwargs['batch_size'],
                shuffle=False   
            )
        }

        dataset_sizes = {'train': len(self.dataset['train']), 'validation': len(self.dataset['validation'])}

        num_overall_updates = ceil(
            dataset_sizes['train'] / optim_kwargs['batch_size']
            ) * optim_kwargs['epochs']

        self.create_sampling_sequence(
            burn_in=optim_kwargs['burn_in'], 
            num_overall_updates=num_overall_updates, 
            num_samples=optim_kwargs['num_samples'] + 1, 
            sampling_strategy=optim_kwargs['sampling_strategy']
            )

        self.init_scheduler(optim_kwargs=optim_kwargs)
        if self._scheduler is not None:
            schedule_every_batch = isinstance(
                self._scheduler, (CyclicLR, OneCycleLR))
        
        best_model_wts = deepcopy(self.model.state_dict())
        best_psnr = -np.inf

        self.model.to(self.device)
        self.model.train()

        num_grad_updates = 0
        for epoch in range(optim_kwargs['epochs']):
            # Each epoch has a training and validation phase
            for phase in ['train', 'validation']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_psnr = 0.0
                running_loss = 0.0
                running_size = 0
                with tqdm(data_loaders[phase],
                          desc='epoch {:d}'.format(epoch + 1) ) as pbar:
                    for _, gt, fbp in pbar:

                        fbp = fbp.to(self.device)
                        gt = gt.to(self.device)
                        # zero the parameter gradients
                        self._optimizer.zero_grad()

                        # track gradients only if in train phase
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = self.model(fbp)
                            loss = criterion(outputs, gt)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(), max_norm=1)
                                self._optimizer.step()

                                if num_grad_updates in self.sampling_sequence:
                                    self.add_parameters_samples()

                                if (self._scheduler is not None and
                                        schedule_every_batch):
                                    self._scheduler.step()

                        for i in range(outputs.shape[0]):
                            gt_ = gt[i, 0].detach().cpu().numpy()
                            outputs_ = outputs[i, 0].detach().cpu().numpy()
                            running_psnr += PSNR(outputs_, gt_, data_range=1)

                        # statistics
                        running_loss += loss.item() * outputs.shape[0]
                        running_size += outputs.shape[0]

                        pbar.set_postfix({'phase': phase,
                                          'loss': running_loss/running_size,
                                          'psnr': running_psnr/running_size})

                        if phase == 'train':
                            num_grad_updates += 1

                    if phase == 'train':
                        if (self._scheduler is not None
                                and not schedule_every_batch):
                            self._scheduler.step()
                    
                    
                    epoch_psnr = running_psnr / dataset_sizes[phase]

                    if (phase == 'train' and (
                        optim_kwargs['save_best_learned_params_path'] is not None) and optim_kwargs['save_best_learned_params_per_epoch']):
                        self.save_learned_params(optim_kwargs['save_best_learned_params_path'], comment=f'epoch_{epoch}_')
        
                    # deep copy the model (if it is the best one seen so far)
                    if phase == 'validation' and epoch_psnr > best_psnr:
                        best_psnr = epoch_psnr
                        best_model_wts = deepcopy(self.model.state_dict())
                        if optim_kwargs['save_best_learned_params_path'] is not None:
                            self.save_learned_params(
                                optim_kwargs['save_best_learned_params_path'])

        print('Best val psnr: {:4f}'.format(best_psnr))
        self.model.load_state_dict(best_model_wts)
        
        if save_samples: 
            self.save_sampled_parameters()

    def init_optimizer(self, optim_kwargs: Dict):
        """
        Initialize the optimizer.
        """
        self._optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=optim_kwargs['optimizer']['lr'],
                weight_decay=optim_kwargs['optimizer']['weight_decay'])

    @property
    def optimizer(self):
        """
        :class:`torch.optim.Optimizer` :
        The optimizer, usually set by :meth:`init_optimizer`, which gets called
        in :meth:`train`.
        """
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    def init_scheduler(self, optim_kwargs: Dict):
        if optim_kwargs['scheduler']['name'].lower() == 'cosine':
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=optim_kwargs['epochs'],
                eta_min=optim_kwargs['scheduler']['lr_min'])
        else:
            raise KeyError

    @property
    def scheduler(self):
        """
        torch learning rate scheduler :
        The scheduler, usually set by :meth:`init_scheduler`, which gets called
        in :meth:`train`.
        """
        return self._scheduler

    @scheduler.setter
    def scheduler(self, value):
        self._scheduler = value

    def save_learned_params(self, path, comment=None):
        """
        Save learned parameters from file.
        """
        if comment is not None: 
            path += comment
        path = path if path.endswith('.pt') else path + 'nn_learned_params.pt'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_learned_params(self, path):
        """
        Load learned parameters from file.
        """
        # TODO: not suitable for nn.DataParallel
        path = path if path.endswith('.pt') else path + '.pt'
        map_location = ('cuda:0' if self.use_cuda and torch.cuda.is_available()
                        else 'cpu')
        state_dict = torch.load(path, map_location=map_location)
        self.model.load_state_dict(state_dict)

    def save_sampled_parameters(self, 
        name: str = 'parameters_samples',
        path: str = './'
        ):

        path = path if path.endswith('.pt') else path + name + '.pt'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.parameters_samples, path)

    def load_sampled_parameters(self, 
        path_to_parameters_samples: str, 
        device: Optional[Any] = None
        ):
        
        path = os.path.join(get_original_cwd(), 
            path_to_parameters_samples if path_to_parameters_samples.endswith('.pt') \
                else path_to_parameters_samples + '.pt')
        self.parameters_samples.extend(
            torch.load(path, map_location=device)
        )