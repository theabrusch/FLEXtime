import torch
import numpy as np
from src.attribution.lrp import dft_lrp
from src.attribution.lrp import lrp_utils

def lrp_stft(relevance_time, sample, window_length, cuda):
    dftlrp = dft_lrp.DFTLRP(window_length, 
                            leverage_symmetry=True, 
                            precision=32,
                            cuda = cuda,
                            create_stdft=False,
                            create_inverse=False
                            )
    freq_relevance = np.zeros((sample.shape[0], window_length//2+1, sample.shape[-1]//window_length))

    for i in range(sample.shape[-1]//window_length):
        signal_freq, relevance_freq = dftlrp.dft_lrp(relevance_time[...,i*window_length:(i+1)*window_length], sample[...,i*window_length:(i+1)*window_length].float(), real=False, short_time=False)
        freq_relevance[...,i] = relevance_freq[:,0,0,:]
    return freq_relevance


def compute_gradient_scores(model, testloader, attr_method, domain = 'fft', stft_params = None):
    lrp_scores = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cuda = torch.cuda.is_available()
    model.to(device)
    for data in testloader:
        if len(data) == 2:
            sample, target = data
            additional_input = None
        elif len(data) == 3:
            sample, time_axis, target = data
            additional_input = time_axis.to(device)
        sample = sample.float().to(device)
        output = model(sample, additional_input)
        # argmax
        output = output.max(1)[1]
        
        relevance_time = lrp_utils.zennit_relevance(sample, model, additional_input = additional_input, target=output, attribution_method=attr_method, cuda=cuda)
        if domain == 'fft':
            if additional_input is not None:
                sample = sample.permute(0, 2, 1)
                relevance_time = np.transpose(relevance_time, axes = (0, 2, 1))
            dftlrp = dft_lrp.DFTLRP(sample.shape[-1], 
                                    leverage_symmetry=True, 
                                    precision=32,
                                    cuda = cuda,
                                    create_stdft=False,
                                    create_inverse=False
                                    )
            signal_freq, relevance_freq = dftlrp.dft_lrp(relevance_time, sample.float(), real=False, short_time=False)
            if additional_input is not None:
                relevance_freq = np.transpose(relevance_freq, axes = (0, 2, 1))
            lrp_scores.append(torch.tensor(relevance_freq))
        elif domain == 'stft':
            relevance_freq = lrp_stft(relevance_time, sample, stft_params['n_fft'], cuda)
            lrp_scores.append(torch.tensor(relevance_freq))
        else:
            lrp_scores.append(torch.tensor(relevance_time))
    return lrp_scores

class CaptumFFTIG():
    def __init__(self, method, model, target, device, time_dim, additional_input = None):
        self.method = method
        self.model = model.to(device)
        self.device = device
        self.additional_input = additional_input
        self.time_dim = time_dim
        self.target = target

    def __call__(self, inputs, device = 'cpu', **kwargs):
        if isinstance(inputs, tuple):
            inputs = inputs[0]
        inputs = inputs.to(self.device)
        if self.additional_input is None:
            additional_input = None
        else:
            additional_input = self.additional_input.to(self.device)
        cuda = torch.cuda.is_available()
        dftlrp = dft_lrp.DFTLRP(inputs.shape[self.time_dim], 
                                        leverage_symmetry=True, 
                                        precision=32,
                                        cuda = cuda,
                                        create_stdft=False,
                                        create_inverse=False
                                        )
        relevances = []
        for x in inputs:
            x = x.unsqueeze(0)
            relevance_time = lrp_utils.zennit_relevance(x, self.model, additional_input = additional_input, target=self.target, attribution_method=self.method, cuda=cuda)
            if self.time_dim == 1:
                x = x.permute(0, 2, 1)
                relevance_time = np.transpose(relevance_time, axes = (0, 2, 1))
            _, relevance_freq = dftlrp.dft_lrp(relevance_time, x, real=False, short_time=False)
            if self.time_dim == 1:
                relevance_freq = np.transpose(relevance_freq, axes = (0, 2, 1))
            relevances.append(torch.tensor(relevance_freq))
        return torch.stack(relevances).squeeze(1)
