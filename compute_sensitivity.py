from src.utils import load_data_and_model
from src.evaluation import compute_sensitivity
import torch
import numpy as np
import pickle
import argparse
import os

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args.validate = False
    test_dloader, model, Fs, time_dim, time_length = load_data_and_model(args, device, subsample = False, seed = 1)
    args.Fs = Fs
    args.time_dim = time_dim
    args.time_length = time_length
    if args.dataset == 'audio':
        dset = f'audio_{args.labeltype}'
    elif 'synth' in args.dataset:
        if args.synth_length != 2000:
            dset = f'{args.dataset}_{args.synth_length}'
        else:
            dset = args.dataset
    else:
        dset = args.dataset
    dset_path = f'{args.output_path}/{dset}_{args.split}.pkl'
    if not os.path.exists(dset_path):
        attributions = {}
    else:
        with open(dset_path, 'rb') as f:
            attributions = pickle.load(f)
    if not args.sensitivity_score in attributions:
        attributions[args.sensitivity_score] = {}
    for method in args.explanation_methods:
        print(f'Computing sensitivity for {method}')
        if method == 'freqrise':
            key_ = f'{method}_{args.freqrise_samples}_{args.num_cells}_{args.probability_of_drop}'
        elif method == 'flextime':
            key_ = f'{method}_{args.nfilters}_{args.numtaps}_{args.keep_ratio_filterbank}_time_reg_{args.time_reg_filterbank}'
        elif method == 'freqmask':
            key_ = f'{method}_{args.keep_ratio_freqmask}_perturb_{args.perturb}_time_reg_{args.time_reg_freqmask}'
        else:
            key_ = method
        if not key_ in attributions[args.sensitivity_score] or args.force_compute:
            if method == 'flextime':
                args.keep_ratio = args.keep_ratio_filterbank
            else:
                args.keep_ratio = args.keep_ratio_freqmask
            
            sens = compute_sensitivity(args.sensitivity_score, method, model, test_dloader, device, args)
            attributions[args.sensitivity_score][key_] = sens
            # dump to dset path
            with open(dset_path, 'wb') as f:
                pickle.dump(attributions, f)
            print(f'Sensitivity for {method} computed and saved')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Synthetic data explainability')
    parser.add_argument('--dataset', type = str, default = 'ecg', help='Dataset to use')
    parser.add_argument('--data_path', type = str, default = '/Users/theb/Desktop/data/dataverse_files/MITECG/', help='Path to AudioMNIST data') #'/Users/theb/Desktop/data/AudioMNIST/' '/Users/theb/Desktop/data/sleep_edf/epoched/'
    parser.add_argument('--split', type = int, default = 1, help='Split to use for TimeX datasets')
    parser.add_argument('--synth_length', type = int, default = 4000, help='Length of synthetic data')
    parser.add_argument('--noise_level', type = float, default = 0.2, help='Noise level for synthetic data')
    parser.add_argument('--explanation_methods', type = str, default = ['freqrise'], nargs = '+', help='Attribution methods to compute')
    parser.add_argument('--sensitivity_score', type = str, default = 'relative_input_fft_sensitivity', help='Sensitivity score to use')
    parser.add_argument('--random_model', type = bool, default = False, help='Use random model')
    parser.add_argument('--compute_accuracy_scores', type = eval, default = True, help='Compute accuracy scores')
    parser.add_argument('--labeltype', type = str, default = 'digit', help='Type of label to use')
    parser.add_argument('--force_compute', type = eval, default = False, help='Force computation of attributions')
    # argumenst for Filteropt and Freqopt
    parser.add_argument('--regstrength', type = float, default = 1., help='Regularization strength for Filteropt')
    parser.add_argument('--keep_ratio_filterbank', type = float, default = 0.1, help='Keep ratio for Filteropt')
    parser.add_argument('--keep_ratio_freqmask', type = float, default = 0.1, help='Keep ratio for Freqopt')
    parser.add_argument('--perturb', type = bool, default = True, help='Use dynamic perturbation')
    parser.add_argument('--nfilters', type = int, default = 128, help='Number of filters in the filterbank')
    parser.add_argument('--numtaps', type = int, default = 51, help='Number of taps in the filterbank')
    parser.add_argument('--time_reg_freqmask', type = float, default = 1, help='Regularization strength for time dimension in Freqmask')
    parser.add_argument('--time_reg_filterbank', type = float, default = 0, help='Regularization strength for time dimension in Filterbank')
    parser.add_argument('--n_samples', type = int, default = 1, help='Number of samples to compute attributions for')
    parser.add_argument('--output_path', type = str, default = 'local_outputs', help='Path to save outputs')
    # arguments for FreqRISE
    parser.add_argument('--freqrise_samples', type = int, default = 500, help='Number of samples to compute attributions for')
    parser.add_argument('--num_cells', type = int, default = 128, help='Number of cells to drop')
    parser.add_argument('--probability_of_drop', type = float, default = 0.5, help='Probability of dropping a cell')
    args = parser.parse_args()
    main(args)
