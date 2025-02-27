from .flextime import *
from .freqmask import *
from .freqrise import *
from .lrp import *

def compute_fft_attribution(method, model, test_dloader, device, args, test_convergence = False):
    if test_convergence:
        stop_criterion = 1e-10
    else:
        stop_criterion = None
    if method in ['ig', 'saliency', 'guided-backprop', 'gxi']:
        attribution = compute_gradient_scores(model, test_dloader, attr_method = method, domain = 'fft')
    elif method == 'freqmask':
        if args.perturb:
            if args.dataset == 'audio':
                perturb = 'window'
            else:
                perturb = 'window'
        else:
            perturb = None

        attribution, epochs = compute_freqmask_scores(model, 
                                                        test_dloader, 
                                                        optimization_params = {'n_epoch': 1000, 'learning_rate': 1.0e-0, 'momentum': 1., 'keep_ratio': args.keep_ratio_freqmask, 
                                                                            'reg_factor_init': args.regstrength, 'stop_criterion': stop_criterion, 'time_dim': args.time_dim,
                                                                            'perturb': perturb, 'return_epochs': test_convergence, 'time_reg_strength': args.time_reg_strength},
                                                        device = device)
    elif method == 'flextime':
        opt_params = {'n_epoch': 1000, 'learning_rate': 1.0e-0, 'momentum': 1., 'keep_ratio': args.keep_ratio_filterbank, 
                      'reg_factor_init': args.regstrength, 'stop_criterion': stop_criterion, 'time_dim': args.time_dim, 'time_reg_strength': args.time_reg_strength,
                      'return_epochs': test_convergence}
        attribution, masks, epochs = compute_flextime_scores(model, 
                                                        test_dloader, 
                                                        filterbank_params = {'numtaps': args.numtaps, 'fs': args.Fs, 'numbanks': args.nfilters, 'time_length': args.time_length},
                                                        optimization_params = opt_params, 
                                                        device = device)
        return attribution, masks
    elif method == 'freqrise':
        attribution = compute_freqrise_scores(model, 
                                                test_dloader, 
                                                exp_label=None, 
                                                n_samples = args.freqrise_samples, 
                                                num_cells = args.num_cells, 
                                                probability_of_drop = args.probability_of_drop, 
                                                domain = 'fft',
                                                use_softmax=False,
                                                stft_params = None,
                                                device = device, 
                                                time_dim=args.time_dim)
    else:
        raise ValueError(f"Method {method} not recognized. Please choose from 'lrp', 'freqmask', 'flextime', 'freqrise'")
    if test_convergence:
        return attribution, epochs
    return attribution