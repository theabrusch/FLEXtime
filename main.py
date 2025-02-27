from src.utils import load_data_and_model
import torch
import numpy as np
from src.attribution import compute_fft_attribution
from src.evaluation import mask_and_predict
import pickle
import os
import argparse

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args.validate = False
    test_dloader, model, Fs, time_dim, time_length = load_data_and_model(args, device, subsample = False)
    args.Fs = Fs
    args.time_dim = time_dim
    args.time_length = time_length
    if args.dataset == 'audio':
        dset = f'audio_{args.labeltype}'
    elif 'synth' in args.dataset:
        if args.synth_length != 2000:
            dset = f'{args.dataset}_{args.synth_length}_{args.noise_level}'
        else:
            dset = f'{args.dataset}_{args.noise_level}'
    else:
        dset = args.dataset

    # get predictions and labels
    if args.random_model:
        dset = f'{dset}_random'
    dset_path = f'{args.output_path}/{dset}_{args.split}.pkl'
    if not os.path.exists(dset_path):
        attributions = {}
        compute_predictions = True
        attributions['deletion'] = {}
        attributions['insertion'] = {}
    else:
        compute_predictions = False
        with open(dset_path, 'rb') as f:
            attributions = pickle.load(f)

    if args.save_data or compute_predictions:
        predictions = []
        labels = []
        data_collect = []
        if 'synth' in args.dataset:
            data_collect = []
        for batch in test_dloader:
            if len(batch) == 2:
                data, target = batch
                data = data.to(device)
                output = model(data.float())
            else:
                data, time_axis, target = batch
                data, time_axis = data.to(device), time_axis.to(device)
                output = model(data.float(), time_axis, captum_input = True)
            if args.save_data:
                data_collect.append(data.detach().cpu())
            predictions.append(output.detach().cpu())
            labels.append(target)
            if 'synth' in args.dataset:
                data_collect.append(data)
        predictions = torch.cat(predictions, dim=0)
        labels = torch.cat(labels, dim=0)
        if args.save_data:
            data_collect = torch.cat(data_collect, dim=0)
        attributions['predictions'] = predictions
        attributions['labels'] = labels
        attributions['data'] = data_collect
        if 'synth' in args.dataset:
            attributions['data'] = torch.cat(data_collect, dim=0)
        
    for method in args.explanation_methods:
        if method == 'freqrise':
            key_ = f'{method}_{args.freqrise_samples}_{args.num_cells}_{args.probability_of_drop}'
        elif method == 'flextime':
            if args.time_reg_strength > 1e-6:
                key_ = f'{method}_{args.nfilters}_{args.numtaps}_{args.keep_ratio_filterbank}_{args.time_reg_strength}'
            else:
                key_ = f'{method}_{args.nfilters}_{args.numtaps}_{args.keep_ratio_filterbank}'
        elif method == 'freqmask':
            if args.time_reg_strength > 1e-6:
                key_ = f'{method}_{args.keep_ratio_freqmask}_perturb_{args.perturb}_{args.time_reg_strength}'
            else:
                key_ = f'{method}_{args.keep_ratio_freqmask}_perturb_{args.perturb}'
        else:
            key_ = method
        if not key_ in attributions.keys() or args.force_compute:
            if method == 'filterbank':
                attribution, masks = compute_fft_attribution(method, model, test_dloader, device, args)
                attributions[key_] = attribution
                attributions[f'filtermasks_{key_}'] = masks
            else:
                attribution = compute_fft_attribution(method, model, test_dloader, device, args)
                attributions[key_] = attribution
            # dump to dset path
            with open(dset_path, 'wb') as f:
                pickle.dump(attributions, f)
    
    if args.compute_accuracy_scores:
        sampling_percent = np.arange(0, 1.05, 0.05)
        for mode in ['deletion', 'insertion']:
            if not mode in attributions.keys():
                attributions[mode] = {}
            for key in attributions.keys():
                print(key.split('_')[0])
                print(args.explanation_methods)
                if key.split('_')[0] in args.explanation_methods:
                    print('Computing', mode, 'for', key.split('_')[0])
                    acc_scores = mask_and_predict(model, test_dloader, attributions[key], sampling_percent, mode = mode, domain='fft', device=device)
                    attributions[mode][key] = acc_scores
            for key in ['random', 'amplitude']:
                acc_scores = mask_and_predict(model, test_dloader, key, sampling_percent, mode = mode, domain='fft', device=device)
                attributions[mode][key] = acc_scores

    # dump to dset path
    with open(dset_path, 'wb') as f:
        pickle.dump(attributions, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Synthetic data explainability')
    parser.add_argument('--dataset', type = str, default = 'pam', help='Dataset to use')
    parser.add_argument('--data_path', type = str, default = '/Users/theb/Desktop/data/dataverse_files/PAM/', help='Path to AudioMNIST data')#
    parser.add_argument('--save_data', type = eval, default = False, help='Save data')
    parser.add_argument('--force_compute', type = eval, default = False, help='Force compute attributions')
    parser.add_argument('--validate', type = eval, default = False, help='Use validation set')
    parser.add_argument('--synth_length', type = int, default = 2000, help='Length of synthetic data')
    parser.add_argument('--noise_level', type = float, default = 0.2, help='Noise level for synthetic data')
    parser.add_argument('--split', type = int, default = 1, help='Split to use for TimeX datasets')
    parser.add_argument('--explanation_methods', type = str, default = ['filterbank'], nargs = '+', help='Attribution methods to compute')
    parser.add_argument('--random_model', type = eval, default = False, help='Use random model')
    parser.add_argument('--compute_accuracy_scores', type = eval, default = True, help='Compute accuracy scores')
    parser.add_argument('--labeltype', type = str, default = 'digit', help='Type of label to use')
    # argumenst for Filteropt and Freqopt
    parser.add_argument('--regstrength', type = float, default = 1., help='Regularization strength for Filteropt')
    parser.add_argument('--time_reg_strength', type = float, default = 1., help='Time regularization strength for Filteropt')
    parser.add_argument('--keep_ratio_freqmask', type = float, default = 0.1, help='Keep ratio for Filteropt')
    parser.add_argument('--keep_ratio_filterbank', type = float, default = 0.1, help='Keep ratio for Filteropt')
    parser.add_argument('--perturb', type = bool, default = True, help='Use dynamic perturbation')
    parser.add_argument('--nfilters', type = int, default = 128, help='Number of filters in the filterbank')
    parser.add_argument('--numtaps', type = int, default = 51, help='Number of taps in the filterbank')
    parser.add_argument('--n_samples', type = int, default = 100, help='Number of samples to compute attributions for')
    parser.add_argument('--output_path', type = str, default = 'local_outputs', help='Path to save outputs')
    # arguments for FreqRISE
    parser.add_argument('--freqrise_samples', type = int, default = 3000, help='Number of samples to compute attributions for')
    parser.add_argument('--num_cells', type = int, default = 128, help='Number of cells to drop')
    parser.add_argument('--probability_of_drop', type = float, default = 0.5, help='Probability of dropping a cell')
    args = parser.parse_args()
    main(args)
