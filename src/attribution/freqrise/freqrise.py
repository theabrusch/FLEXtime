#@title RELAX
from torch.fft import rfft as tfft
from torch.fft import irfft as tifft
import torch
import torch.nn as nn

class FreqRISE(nn.Module):
    def __init__(self,
                 input_data: torch.Tensor,
                 encoder: nn.Module,
                 additional_input: torch.Tensor = None,
                 batch_size: int = 10,
                 num_batches: int = 300,
                 device: str = 'cpu',
                 domain = 'fft',
                 use_softmax = True,
                 stft_params = None,
                 time_dim = 1,
                 ):

        super().__init__()

        self.batch_size = batch_size
        self.device = device
        if device != 'mps':
            input_data = input_data.unsqueeze(0).to(self.device)
        self.domain = domain
        if domain == 'fft':
            self.input_fft = tfft(input_data, dim=time_dim)
        elif domain == 'stft':
            self.stft_params = stft_params 
            self.input_fft = torch.stft(input_data.squeeze(), return_complex=True, **stft_params)
        else:
            self.input_fft = input_data
        self.additional_input = additional_input
        self.num_batches = num_batches
        self.use_softmax = use_softmax
        self.encoder = encoder.eval().to(self.device)
        self.shape = self.input_fft.shape
        self.original_shape = input_data.shape
        self.time_dim = time_dim

    def forward(self, mask_generator, keep_representations = False, **kwargs) -> None:
        i = 0 
        p = []
        mask_type = torch.complex64 if self.domain in ['fft', 'stft'] else torch.float32
        if keep_representations:
            self.representations = []
        for _ in range(self.num_batches):
            for masks in mask_generator(self.batch_size, self.shape, self.device, dtype = mask_type, **kwargs):
                if len(masks) == 2:
                    x_mask, masks = masks
                else:
                    x_mask = self.input_fft*masks
                    if self.domain == 'fft':
                        x_mask = tifft(x_mask, dim=self.time_dim).reshape((self.batch_size, *self.original_shape[1:]))
                    elif self.domain == 'stft':
                        x_mask = torch.istft(x_mask, length = self.original_shape[-1], return_complex = False, **self.stft_params)
                        x_mask = x_mask.reshape((self.batch_size, *self.original_shape))

                with torch.no_grad():
                    if keep_representations:
                        latents, predictions = self.encoder(x_mask.float().to(self.device), feats_and_class = True)
                        self.representations.append(latents.detach().cpu())
                        predictions = predictions.detach().cpu()
                    else:
                        predictions = self.encoder(x_mask.float().to(self.device), self.additional_input).detach()
                if self.device == 'mps':
                    predictions = predictions.cpu()
                if self.use_softmax:
                    predictions = torch.nn.functional.softmax(predictions, dim=1)
                sal = torch.matmul(predictions.transpose(0,1).float(), masks.view(self.batch_size, -1).abs().float()).transpose(0,1).view(1, *self.shape[1:], -1).cpu()
                if self.domain == 'stft':
                    sal = sal.view(1, *self.input_fft.shape, -1)
                p.append(sal)
                i += 1
        self.importance = torch.cat(p, dim=0).sum(dim=0)/(self.num_batches*self.batch_size)
        if keep_representations:
            self.representations = torch.cat(self.representations, dim=0)
        return None

class CaptumFreqRISE():
    def __init__(self, 
                 model, 
                 target,
                 additional_input = None,
                 batch_size: int = 10,
                 num_batches: int = 300,
                 device: str = 'cpu',
                 domain = 'fft',
                 use_softmax = True,
                 stft_params = None,
                 time_dim = 1,
                 mask_generator = None,
                 mask_kwargs = {}):
                 
        self.model = model
        self.target = target
        self.device = device
        self.domain = domain
        self.stft_params = stft_params
        self.time_dim = time_dim
        self.model.eval().to(self.device)
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.use_softmax = use_softmax
        self.additional_input = additional_input
        self.mask_kwargs = mask_kwargs
        self.mask_generator = mask_generator
        if self.additional_input is not None:
            self.additional_input = additional_input.repeat(self.batch_size, 1)

    
    def fit(self, input_data):
        rise = FreqRISE(input_data, self.model, self.additional_input, self.batch_size, self.num_batches, self.device, self.domain, self.use_softmax, self.stft_params, self.time_dim)
        rise.forward(self.mask_generator, **self.mask_kwargs)
        return rise.importance
    
    def __call__(self, inputs):
        if isinstance(inputs, tuple):
            inputs = inputs[0]
        masks = []
        for x in inputs:
            mask = self.fit(x)
            mask = (mask.cpu()[...,self.target]/self.mask_kwargs['probablity_of_drop']).squeeze(-1)
            # min max normalize
            mask = (mask - mask.min()) / (mask.max() - mask.min())
            masks.append(mask.unsqueeze(0))
        masks = torch.stack(masks)
        if self.time_dim == -1:
            # reshape all other dimensions than time to fit input
            masks.squeeze()
            masks = masks.reshape(*inputs.shape[:-1], masks.shape[-1])
        else:
            masks = masks.reshape(*inputs.shape[:-2], masks.shape[-2], inputs.shape[-1])
        return masks
