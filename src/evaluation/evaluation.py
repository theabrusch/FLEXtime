import torch
import torch.nn.functional as F
from src.attribution import FreqRISE, FilterBank, CaptumFreqMask, CaptumFLEXtime, CaptumFFTIG, CaptumFreqRISE
from src.attribution.freqrise.masking import mask_generator
from captum.metrics import sensitivity_max


def mask_and_predict(model, test_loader, importance, quantiles, mode = 'deletion', domain = 'fft', device = 'cpu', stft_params = None):
    model.eval().to(device)
    with torch.no_grad():
        accs = []
        mean_true_class_probs = []
        ce_losses = []
        for quantile in quantiles:
            correct = 0
            total = 0
            ce_loss = 0
            mean_true_class_prob = 0
            for i, batch in enumerate(test_loader):
                if len(batch) == 2:
                    data, true_label = batch
                    time_axis = None
                    time_dim = -1
                else:
                    data, time_axis, true_label = batch
                    time_dim = 1
                if domain == 'fft':
                    data = torch.fft.rfft(data, dim=time_dim).to(device)
                elif domain == 'stft':
                    data_shape = data.shape
                    data = torch.stft(data.squeeze().to(device), **stft_params, return_complex = True, dim=time_dim)
                else:
                    data = data.to(device)
                shape = data.shape
                if importance == 'random':
                    imp = torch.rand_like(data).float()
                elif importance == 'amplitude':
                    imp = torch.abs(data)
                else:
                    if domain in ['fft', 'time']:
                        imp = importance[i].reshape(data.shape)
                    else:
                        imp = importance[i].squeeze()
                # take q percent largest values 
                flattened_imp = imp.reshape(shape[0], -1)
                k = int(quantile * flattened_imp.size(1))
                # Find top 10% (T * D * 10%)
                topk_values, topk_indices = torch.topk(flattened_imp, k=k, dim=1)
                mask = torch.zeros_like(flattened_imp, dtype=torch.bool)
                # Set the positions of the top-k elements to True
                mask.scatter_(1, topk_indices, True)
                mask = mask.view(shape).to(device)
                if mode == 'insertion':
                    data = data * mask
                elif mode == 'deletion':
                    # remove the top quantile percent of values
                    data = data * (~mask)


                if domain == 'fft':
                    data = torch.fft.irfft(data, dim=time_dim)
                elif domain == 'stft':
                    data = torch.istft(data, length = data_shape[time_dim], **stft_params, dim = time_dim, return_complex = False)
                    data = data.view(data_shape)
                if time_axis is not None:
                    output = model(data.float(), time_axis.to(device), captum_input = True).detach().cpu()
                else:
                    output = model(data.float()).detach().cpu()
                _, predicted = torch.max(output, 1)
                total += true_label.size(0)
                correct += (predicted == true_label).sum().item()
                # one hot encode true label
                mean_true_class_prob += torch.take_along_dim(F.softmax(output, dim=1), true_label.unsqueeze(1), dim = 1).sum().item()
                ce_loss += F.cross_entropy(output, true_label).item()/len(batch)
            accs.append(correct/total)
            mean_true_class_probs.append(mean_true_class_prob/total)
            ce_losses.append(ce_loss)
    return accs, mean_true_class_probs, ce_losses


def compute_sensitivity(sensitivity_score, method, model, testloader, device, args):
    sensitivities = []
    model.to(device)
    if method == 'flextime':
        optimization_parameters = {'n_epoch': 1000, 'learning_rate': 1.0e-0, 'momentum': 1., 'keep_ratio': args.keep_ratio_filterbank, 
                                    'reg_factor_init': args.regstrength, 'stop_criterion': None, 'time_dim': args.time_dim, 'time_reg_strength': args.time_reg_filterbank}
        filterbank_params = {'numtaps': args.numtaps, 'fs': args.Fs, 'numbanks': args.nfilters, 'time_length': args.time_length}
        filterbank = FilterBank(**filterbank_params)
    elif method == 'freqmask':
        if args.perturb:
            perturb = 'window'
        optimization_parameters = {'n_epoch': 1000, 'learning_rate': 1.0e-0, 'momentum': 1., 'keep_ratio': args.keep_ratio_freqmask, 
                                   'perturb': perturb, 'time_dim': args.time_dim, 'time_reg_strength': args.time_reg_freqmask}
        
    i = 0
    for batch in testloader:
        print("Computing batch", i+1, "/", len(testloader))
        i+=1
        j = 0
        for data in zip(*batch):
            if len(data) == 2:
                sample, target = data
                sample = sample.float().to(device)
                additional_input = None
                with torch.no_grad():
                    model_output = model(sample.unsqueeze(0).to(device))
            elif len(data) == 3:
                sample, time_axis, target = data
                additional_input = time_axis.float().unsqueeze(0).to(device)
                with torch.no_grad():
                    model_output = model(sample.unsqueeze(0).to(device), additional_input)

            if method in ['freqmask', 'flextime']:
                target = torch.nn.functional.softmax(model_output, dim = 1)
            else:
                target = torch.argmax(model_output, dim = 1)
            print("Computing sample", j+1, "/", len(batch[0]))
            j+=1
            # optimize mask
            if method == 'freqmask':
                explainer = CaptumFreqMask(model, target, optimization_parameters, additional_input=additional_input, device = device)
            elif method == 'flextime':
                explainer = CaptumFLEXtime(model, target, filterbank, optimization_parameters, additional_input=additional_input, device = device)
            elif method == 'freqrise':
                explainer = CaptumFreqRISE(model, target.detach().cpu(), device = device, additional_input = additional_input, batch_size = 500, 
                                           num_batches = args.freqrise_samples//500, time_dim=args.time_dim, mask_generator = mask_generator, 
                                           mask_kwargs = {'num_cells': args.num_cells, 'probablity_of_drop': args.probability_of_drop, 'time_dim': args.time_dim})
            else:
                explainer = CaptumFFTIG(method, model, target, additional_input = additional_input,
                                        device = device, time_dim = args.time_dim)
            if sensitivity_score == 'sensitivity':
                score = sensitivity_max(explanation_func=explainer, inputs = sample.unsqueeze(0).float().to(device))
            else:
                score = compute_relative_sensitivity(sensitivity_score, explainer, sample.unsqueeze(0).float().to(device), model, 10, testloader.dataset.std, additional_input=additional_input, time_dim = args.time_dim)
            sensitivities.append(score)
    return sensitivities


def compute_relative_sensitivity(sens_type, explanation_func, inputs, model, n_samples, std, additional_input = None, time_dim = -1):
    # create n_samples perturbed versions of the input
    perturbed_inputs = inputs + torch.randn(n_samples, *inputs.shape[1:]).to(inputs.device) * std * 0.05
    orig_explanation = explanation_func(inputs)
    explanations = explanation_func(perturbed_inputs)
    if additional_input is not None:
        perturbed_outputs = model(perturbed_inputs, additional_input.repeat(n_samples, 1))
        orig_output = model(inputs, additional_input)
    else:
        perturbed_outputs = model(perturbed_inputs)
        orig_output = model(inputs)
    eps_min = 1e-6

    # compute the nominator over all dims except the batch dimension
    ndims = len(orig_explanation.shape)
    if ndims == 2:
        norm_function = lambda x: torch.norm(x, p=2, dim=1)
    elif ndims == 3:
        norm_function = lambda x: torch.norm(x, p=2, dim=(1, 2))
    elif ndims == 4:
        norm_function = lambda x: torch.norm(torch.norm(x, p=2, dim=(1, 2)), p = 2, dim = 1)

    nominator = norm_function((explanations - orig_explanation)/(orig_explanation + (orig_explanation == 0) * eps_min)).cpu()
    if sens_type == 'relative_input_sensitivity':
        denominator = norm_function((perturbed_inputs - inputs)/(inputs + (inputs == 0)*eps_min )).cpu()
    elif sens_type == 'relative_output_sensitivity':
        denominator = torch.norm((perturbed_outputs - orig_output), p=2, dim=1).cpu() 
    elif sens_type == 'relative_input_fft_sensitivity':
        input_fft = torch.fft.rfft(inputs, dim = time_dim)
        perturbed_input_fft = torch.fft.rfft(perturbed_inputs, dim = time_dim)
        denominator = norm_function((perturbed_input_fft - input_fft)/(input_fft + (input_fft == 0)*eps_min)).cpu()
    
    denominator += (denominator == 0) * eps_min
    # get indices of changed predictions
    changed_predictions = perturbed_outputs.argmax(dim=1) != orig_output.argmax(dim=1)
    relative_change = nominator/denominator
    relative_change[changed_predictions] = torch.nan
    score = torch.max(relative_change)
    return score.item()


