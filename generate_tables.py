import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
with open('outputs/collect.pkl', 'rb') as f:
    results = pickle.load(f)

datasets = ['audio_gender', 'audio_digit', 'pam', 'epilepsy', 'ecg', 'sleepedf']

filterbank_params = {
    'nfilters': [128, 128, 32, 32, 64, 256],
    'ntaps': [501, 501, 95, 75, 105, 901],
    'keepratio': [0.1, 0.1, 0.1, 0.1, 0.05, 0.1],
    'time_reg': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}

# filterbank_params = {
#     'nfilters': [512, 512, 128, 64, 32, 256],
#     'ntaps': [501, 901, 205, 75, 105, 901],
#     'keepratio': [0.05, 0.05, 0.05, 0.1, 0.1, 0.1],
#     'time_reg': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# }

freqrise_cells = [128, 128, 32, 32, 64, 256]
dynamask_keep_ratios = [0.1, 0.05, 0.05, 0.05, 0.05, 0.1]
#freqrise_samples = [10001, 10001, 5001, 5001, 5001, 10000]
freqrise_samples = [10001, 10001, 5000, 5000, 5000, 10000]

methods = ['saliency', 'gxi', 'guided-backprop', 'freqrise', 'ig', 'freqmask', 'filterbank']
method_names = ['Saliency*', 'G$\\times$I*', 'GB*', 'FreqRISE', 'IG*', 'FreqMask', 'FLEXtime']

# gather results
collected_results = {
    'insertion': {},
    'smoothness': {},
    'complexity': {},
    'sensitivity': {},
    'relative_output_sensitivity': {}
}

for metric in ['sensitivity', 'relative_output_sensitivity', 'insertion', 'smoothness', 'complexity']:
    for method, method_name in zip(methods, method_names):
        table_line = f"{method_name} "
        for i, dataset in enumerate(datasets):
            if method == 'freqrise':
                method_string = f'{method}_{freqrise_samples[i]}_{freqrise_cells[i]}_0.5'
            elif method == 'freqmask':
                method_string = f'{method}_{dynamask_keep_ratios[i]}_perturb_True_1.0'
                if metric == 'sensitivity' or metric == 'relative_output_sensitivity':
                    method_string = f'{method}_{dynamask_keep_ratios[i]}_perturb_True_time_reg_1.0'
            elif method == 'filterbank':
                if filterbank_params['time_reg'][i] > 1e-6:
                    method_string = f'{method}_{filterbank_params['nfilters'][i]}_{filterbank_params['ntaps'][i]}_{filterbank_params['keepratio'][i]}_{filterbank_params['time_reg'][i]}'
                else:
                    if metric == 'sensitivity' or metric == 'relative_output_sensitivity':
                        method_string = f'{method}_{filterbank_params['nfilters'][i]}_{filterbank_params['ntaps'][i]}_{filterbank_params['keepratio'][i]}_time_reg_0'
                    else:
                        method_string = f'{method}_{filterbank_params['nfilters'][i]}_{filterbank_params['ntaps'][i]}_{filterbank_params['keepratio'][i]}'
            else:
                method_string = method
            if method_string in results[dataset][metric].keys():
                values = results[dataset][metric][method_string]
                if metric == 'insertion':
                    values = np.stack(values)[:,2]
                mean_ = np.mean(values)
                stderror = np.std(values)/np.sqrt(len(values))
                if len(values) < 5 and dataset != 'audio_gender':
                    print(f"Warning: {metric} {method} on {dataset} has less than 5 samples")
                if dataset in collected_results[metric].keys():
                    collected_results[metric][dataset]['mean'].append(mean_)
                    collected_results[metric][dataset]['stderror'].append(stderror)
                else:
                    collected_results[metric][dataset] = {'mean': [mean_], 'stderror': [stderror]}
            else:
                if dataset in collected_results[metric].keys():
                    collected_results[metric][dataset]['mean'].append(np.nan)
                    collected_results[metric][dataset]['stderror'].append(np.nan)
                else:
                    collected_results[metric][dataset] = {'mean': [np.nan], 'stderror': [np.nan]}

table_to_generate = 'relative_output_sensitivity'

table = ""
metric_results = collected_results[table_to_generate]

for j, method_name in enumerate(method_names):
    table_line = f"{method_name} "
    summed_rank = 0
    if method_name == 'FLEXtime':
        print('hej')
    for i, dataset in enumerate(datasets):
        method_string = method_name
        mean_ = metric_results[dataset]['mean'][j]
        stderror = metric_results[dataset]['stderror'][j]
        if table_to_generate == 'insertion':
            # write e.g. 0.967 as .967
            mean_string = f"{mean_:.3f}"[1:]
            std_string = f"{stderror:.3f}"[1:]
            # get methods rank for dataset
            sorted_means = np.argsort(-np.array(metric_results[dataset]['mean']))
        else:
            mean_string = f"{mean_:.2f}"
            std_string = f"{stderror:.2f}"
            sorted_means = np.argsort(metric_results[dataset]['mean'])
        rank = np.where(sorted_means == j)[0][0] + 1
        summed_rank += rank
        if table_to_generate in ['smoothness', 'complexity', 'sensitivity', 'relative_output_sensitivity']:
            best_res = np.argmin(metric_results[dataset]['mean'])
        else:
            best_res = np.argmax(metric_results[dataset]['mean'])
        if j == best_res:
            table_line += f"& \\textbf{{{mean_string}}}({std_string})"
        else:
            # get CI of best result
            best_mean = metric_results[dataset]['mean'][best_res]
            best_stderror = metric_results[dataset]['stderror'][best_res]
            z = 1.96
            best_ci = z * best_stderror
            current_ci = z * stderror
            # test if the two CIs overlap
            if (best_mean - best_ci) <= (mean_ + current_ci) and (mean_ - current_ci) <= (best_mean + best_ci):
                table_line += f"& \\textbf{{{mean_string}}}({std_string})"
            else:
                # test if it is the second best
                if table_to_generate in ['smoothness', 'complexity', 'sensitivity', 'relative_output_sensitivity']:
                    second_best_res = np.argsort(metric_results[dataset]['mean'])[1]
                else:
                    second_best_res = np.argsort(metric_results[dataset]['mean'])[-2]
                if j == second_best_res:
                    table_line += f"& \\underline{{{mean_string}}}({std_string})"
                else:
                    table_line += f"& {mean_string}({std_string})"
    mean_rank = summed_rank/len(datasets)
    table_line += f"& {mean_rank:.1f}"
    table_line += '\\\\ \n'
    table+=table_line
print(table)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10
plt.figure(figsize=(4, 2))
normalized_insertion = np.stack([collected_results['insertion'][dataset]['mean'] for dataset in datasets])
normalized_complexity = np.stack([(collected_results['complexity'][dataset]['mean']-np.min(collected_results['complexity'][dataset]['mean'], keepdims=True))/np.max(collected_results['complexity'][dataset]['mean'], keepdims=True) for dataset in datasets])
normalized_complexity = np.stack([collected_results['complexity'][dataset]['mean'] for dataset in datasets])
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
markers = ['o', 's', 'D', '^', 'v', 'p', 'P']
# make plot of insertion vs complexity
for j, method_name in enumerate(method_names):
    #plt.errorbar(np.mean(normalized_insertion[:,j]), np.mean(normalized_complexity[:, j]),  xerr=np.std(normalized_insertion[:,j])/np.sqrt(len(normalized_insertion[:,j])), yerr=np.std(normalized_complexity[:, j])/np.sqrt(len(normalized_complexity[:, j])),
                #label=method_name, marker = 'o')
    plt.scatter(np.mean(normalized_insertion[:,j]), np.mean(normalized_complexity[:, j]), label=method_name, marker = 'o')
        
plt.xlabel('Faithfulness')
plt.ylabel('Complexity')
plt.legend(ncols = 3, loc = 'upper left', fontsize = 8, columnspacing = 0.5)
plt.tight_layout()
plt.savefig('/Users/theb/Documents/PhD/Udveksling/Explainability/flextime/faithfulness_vs_complexity.pdf')
plt.show()