import pickle 
import io
import glob
import torch
import numpy as np
from quantus.metrics import Complexity

# import with torch to cpu
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

datasets = ['audio_digit', 'audio_gender', 'epilepsy', 'ecg', 'pam', 'sleepedf']
filterbank_params = {
    'nfilters': [128, 128, 32, 64, 32, 256],
    'ntaps': [501, 501, 75, 105, 95, 901],
    'keepratio': [0.1, 0.1, 0.1, 0.05, 0.1, 0.1]
}
dynamask_keep_ratios = [0.05, 0.1, 0.05, 0.05, 0.05, 0.1]

collect_results = {}
for dataset in datasets:
    files = glob.glob('outputs/' + dataset + '_[0-9].pkl')
    attributions = []
    for file in files:
        with open(file, 'rb') as f:
            attributions.append(CPU_Unpickler(f).load())
    
    insertion = {}
    deletion = {}
    complexities = {}
    grad_complexties = {}
    sensitivity = {}
    rel_out_sens = {}
    comp = Complexity()
    for att in attributions:
        for key in att['insertion'].keys():
            if not key in insertion.keys():
                insertion[key] = []
                deletion[key] = []
            insertion[key].append(att['insertion'][key][1])
            deletion[key].append(att['deletion'][key][1])
            if not key in ['random', 'amplitude']:
                #compute complexity
                if not key in complexities.keys():
                    complexities[key] = []
                    grad_complexties[key] = []
                scores = []
                grad_scores = []
                for i in range(len(att[key])):
                    if not 'filtermasks' in key:
                        expl = att[key][i].flatten(start_dim = 1).numpy()
                    else:
                        expl = np.reshape(att[key][i], (att[key][i].shape[0], -1))
                    if 'audio' in dataset or 'sleepedf' in dataset:
                        ex = np.maximum(att[key][i], 0).numpy()
                        if 'filterbank' in key:
                            ex = np.transpose(ex, (0, 2, 1))
                        # min max normalize
                        ex = (ex - np.min(ex, axis = -1, keepdims=True)) / (np.max(ex, axis = -1, keepdims=True) - np.min(ex, axis = -1, keepdims=True))
                        expl_grad = np.abs(np.diff(ex, axis = -1)).sum(axis=-1)
                        expl_grad = np.reshape(expl_grad, (att[key][i].shape[0], -1))
                    else:
                        ex = np.maximum(att[key][i], 0).numpy()
                        # min max normalize
                        ex = (ex - np.min(ex, axis = 1, keepdims=True)) / (np.max(ex, axis = 1, keepdims=True) - np.min(ex, axis = 1, keepdims=True))
                        expl_grad = np.abs(np.diff(ex, axis = 1)).sum(axis = 1)
                        expl_grad = np.reshape(expl_grad, (att[key][i].shape[0], -1))


                    expl = np.maximum(expl, 0)
                    complexity = comp.evaluate_batch(expl, expl)
                    complexity = np.nan_to_num(complexity)
                    expl_grad = np.nan_to_num(expl_grad)
                    scores += complexity.tolist()
                    grad_scores += list(expl_grad)
                complexities[key].append(np.mean(scores))
                grad_complexties[key].append(np.mean(grad_scores))
        for key in att['sensitivity'].keys():
            sens = torch.tensor(att['sensitivity'][key])
            # remove samples that are larger than 5 std away from the mean
            if (sens > (3 * sens.std() + sens.mean())).sum() > 5:
                print(key, (sens >(3 * sens.std() + sens.mean())).sum())
            sens = sens[sens < (3 * sens.std() + sens.mean())]
            sens = sens.mean().item()
            if not key in sensitivity.keys():
                sensitivity[key] = []
            sensitivity[key].append(sens)
        for key in att['relative_output_sensitivity'].keys():
            sens = torch.tensor(att['relative_output_sensitivity'][key])
            sens = np.log(np.nanmean(sens))
            if not key in rel_out_sens.keys():
                rel_out_sens[key] = []
            rel_out_sens[key].append(sens)

    dataset_results = {
        'insertion': insertion,
        'complexity': complexities,
        'smoothness': grad_complexties,
        'sensitivity': sensitivity,
        'relative_output_sensitivity': rel_out_sens
    }
    collect_results[dataset] = dataset_results


with open('outputs/collect.pkl', 'wb') as f:
    pickle.dump(collect_results, f)

