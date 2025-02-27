import numpy as np
import torch
from quantus.metrics import Complexity


def compute_relevance_score(relevance, true_label):
    positive_attr = relevance[true_label]
    # sum over all relevance except for the excluded ones
    rel_sum = relevance.sum()
    return positive_attr.sum()/rel_sum

def relevance_rank_accuracy(relevance, true_label):
    if isinstance(relevance, torch.Tensor):
        relevance = relevance.numpy()
    if isinstance(true_label, torch.Tensor):
        true_label = true_label.numpy()
    true_label = true_label
    K = len(true_label)
    top_K = np.argsort(relevance)[-K:]
    accuracy = len(set(top_K).intersection(set(true_label)))/K
    return accuracy

def speech_fundamental_freq(relevance, freq_axis, label):
    male_fundamental_freq = [50, 155]
    female_fundamental_freq = [165, 300]
    fundamental_freqs = [female_fundamental_freq, male_fundamental_freq]
    # test if the highest relevance is in the correct frequency range
    max_freq = freq_axis[np.argmax(relevance)]
    if max_freq >= fundamental_freqs[label][0] and max_freq <= fundamental_freqs[label][1]:
        eval = 1
    else:
        eval = 0
    return eval

def compute_complexity(explanations):
    comp = Complexity(return_aggregate=True)
    scores = []
    for i in range(len(explanations)):
        expl = explanations[i].flatten(start_dim = 1).numpy()
        complexity = comp.evaluate_batch(expl, expl)
        complexity = np.nan_to_num(complexity)
        scores += complexity.tolist()
    return np.mean(scores)

def estimate_masked_energy(masked_sample, ideal_masked_sample, bin_size = None):
    # function to estimate the difference in energy between the true masked signal
    # and the idealized masked signal

    if bin_size is not None:
        # estimate energy in frequency bins of bin_size Hz
        # make sure the length is divisable by bin_size
        masked_sample = masked_sample[1:].view(-1, bin_size).sum(dim=1)
        ideal_masked_sample = ideal_masked_sample[1:].abs().view(-1, bin_size).sum(dim=1)

    masked_sample = masked_sample.numpy()
    ideal_masked_sample =  ideal_masked_sample.numpy()

    return ((masked_sample - ideal_masked_sample)**2).mean()