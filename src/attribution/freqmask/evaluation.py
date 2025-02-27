import torch
import numpy as np
from src.attribution.freqmask.freqmask import FreqMask

def compute_freqmask_scores(model, 
                            testloader, 
                            optimization_params = {'n_epoch': 500, 'learning_rate': 1.0e-5, 'momentum': 0.9, 'keep_ratio': 0.1, 'reg_factor_init': 30},
                            device = 'cpu'):
    masks = []
    epochs = []
    maskoptimizer = FreqMask(model, device = device)

    i = 0
    for batch in testloader:
        batch_scores = []
        print("Computing batch", i+1, "/", len(testloader))
        i+=1
        j = 0
        for data in zip(*batch):
            if len(data) == 2:
                sample, target = data
                additional_input = None
            elif len(data) == 3:
                sample, time_axis, target = data
                additional_input = time_axis.unsqueeze(0).to(device)

            print("Computing sample", j+1, "/", len(batch[0]))
            j+=1
            # optimize mask
            mask, epoch = maskoptimizer.fit(sample.unsqueeze(0), additional_inputs=additional_input, verbose = False, **optimization_params)
            mask = mask.squeeze().cpu().detach()
            # remove 
            epochs.append(epoch)
            # min max normalize
            batch_scores.append(mask)
        masks.append(torch.stack(batch_scores))
    return masks, epochs
