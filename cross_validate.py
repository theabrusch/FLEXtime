from src.utils import load_data_and_model
import torch
import numpy as np
from src.attribution import compute_fft_attribution
from src.evaluation import mask_and_predict, compute_complexity
import pickle
import os
import argparse
import json
import copy


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args.nfilters)

    args.random_model = False
    args.validate = True
    args.synth_length = 2000
    test_dloader, model, Fs, time_dim, time_length = load_data_and_model(args, device, subsample = False)
    args.Fs = Fs
    args.time_dim = time_dim
    args.time_length = time_length
    if args.dataset == 'audio':
        dset = f'audio_{args.labeltype}'
    else:
        dset = args.dataset
    if args.time_reg_strength == 0:
        dset_path = f'{args.output_path}/{dset}_{args.split}_cv.pkl'
    else:
        dset_path = f'{args.output_path}/{dset}_{args.split}_cv_time_reg_{args.time_reg_strength}.pkl'
    if not os.path.exists(dset_path):
        attributions = {}
        attributions['complexity'] = {}
        attributions['deletion'] = {}
        attributions['insertion'] = {}
    else:
        with open(dset_path, 'rb') as f:
            attributions = pickle.load(f)

    sampling_percent = np.arange(0, 1.05, 0.05)
    arguments = copy.deepcopy(args)
    for method in args.explanation_methods:
        if method == 'flextime':
            for nfilters in args.nfilters:
                print(f'Computing attributions for {nfilters} filters')
                for numtaps in args.numtaps:
                    for keep_ratio in args.keep_ratio:
                        arguments.nfilters = nfilters
                        arguments.numtaps = numtaps
                        arguments.keep_ratio_filterbank = keep_ratio
                        key_ = f'{method}_{nfilters}_{numtaps}_{keep_ratio}'
                        attribution, _ = compute_fft_attribution(method, model, test_dloader, device, arguments)
                        comp = compute_complexity(attribution)
                        attributions['complexity'][key_] = comp
                        for mode in ['deletion', 'insertion']:
                            acc_scores = mask_and_predict(model, test_dloader, attribution, sampling_percent, mode = mode, domain='fft', device=device)
                            attributions[mode][key_] = acc_scores
                        # dump to dset path
                        with open(dset_path, 'wb') as f:
                            pickle.dump(attributions, f)
        elif method == 'freqmask':
            for keep_ratio in args.keep_ratio:
                key_ = f'{method}_{keep_ratio}_perturb_{args.perturb}'
                arguments.keep_ratio_freqmask = keep_ratio
                attribution = compute_fft_attribution(method, model, test_dloader, device, arguments)
                comp = compute_complexity(attribution)
                attributions['complexity'][key_] = comp
                for mode in ['deletion', 'insertion']:
                    acc_scores = mask_and_predict(model, test_dloader, attribution, sampling_percent, mode = mode, domain='fft', device=device)
                    attributions[mode][key_] = acc_scores
                # dump to dset path
                with open(dset_path, 'wb') as f:
                    pickle.dump(attributions, f)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Synthetic data explainability')
    parser.add_argument('--dataset', type = str, default = 'audio', help='Dataset to use')
    parser.add_argument('--data_path', type = str, default = '/Users/theb/Desktop/data/AudioMNIST/', help='Path to AudioMNIST data')
    parser.add_argument('--split', type = int, default = 0, help='Split to use for TimeX datasets')
    parser.add_argument('--explanation_methods', type = str, default = ['filterbank'], nargs = '+', help='Attribution methods to compute')
    parser.add_argument('--labeltype', type = str, default = 'digit', help='Type of label to use')
    # argumenst for Filteropt and Freqopt
    parser.add_argument('--regstrength', type = float, default = 1., help='Regularization strength for Filteropt')
    parser.add_argument('--time_reg_strength', type = float, default = 0, help='Regularization strength for time dimension')
    parser.add_argument('--keep_ratio', type = float, default = [0.1], nargs = '+', help='Keep ratio for Filteropt')
    parser.add_argument('--perturb', type = bool, default = True, help='Use dynamic perturbation')
    parser.add_argument('--nfilters', type = int, default = [128], nargs = '+', help='Number of filters in the filterbank')
    parser.add_argument('--numtaps', type = int, default = [51], nargs = '+', help='Number of taps in the filterbank')
    parser.add_argument('--n_samples', type = int, default = 20, help='Number of samples to compute attributions for')
    parser.add_argument('--output_path', type = str, default = 'local_outputs', help='Path to save outputs')
    # arguments for FreqRISE
    parser.add_argument('--freqrise_samples', type = int, default = 3000, help='Number of samples to compute attributions for')
    parser.add_argument('--num_cells', type = int, default = 128, help='Number of cells to drop')
    parser.add_argument('--probability_of_drop', type = float, default = 0.5, help='Probability of dropping a cell')
    args = parser.parse_args()
    main(args)
