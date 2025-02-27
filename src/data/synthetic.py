import numpy as np
from scipy.fft import rfftfreq
from statsmodels.tsa.arima_process import arma_generate_sample
import torch.nn as nn
import torch
from itertools import chain, combinations
from scipy.special import voigt_profile
from scipy.stats import truncnorm
from scipy.signal import butter, lfilter, sosfiltfilt

class TSGenerator():
    def __init__(self, length, fs, num_variables = 1, n_freqs = 1):
        assert num_variables == 1, "Only one variable is supported at the moment"
        self.num_variables = num_variables
        self.length = length
        self.fs = fs
        self.n_freqs = n_freqs
        target_weights = []
        ranges_ = [(-2, -0.5), (0.5, 3)]
        for i in range(n_freqs):
            r = ranges_[i]
            target_weights.append(np.random.uniform(*r, (1, 1)))
        self.target_weights = np.concatenate(target_weights, axis = 1)
        self.target_bias = np.random.normal(0, 1, (1, 1))
    
    def generate(self, 
                 n_samples = 100, 
                 type='sine', 
                 phase=0, 
                 noiselevel=None):
        freqs = []
        max_freq = self.fs/2
        for i in range(self.n_freqs):
            freqs.append(np.random.uniform(0.1, max_freq, (n_samples, 1)))
        
        standardized_freqs = np.concatenate(freqs, axis = 1)
        target = standardized_freqs@self.target_weights.T + self.target_bias + np.random.normal(0, 0.1, (n_samples, 1))
        amp = []
        for _ in range(self.n_freqs):
            amp.append(np.random.uniform(0.8, 1, (n_samples, 1)))

        if type == 'sine':
            t = np.repeat(np.expand_dims(np.arange(0, self.length, 1/self.fs), axis = 0), n_samples, axis = 0)
            data = np.zeros(t.shape)
            for i in range(self.n_freqs):
                freq = freqs[i]
                data += amp[i] * np.sin(2*np.pi*freq*t + phase)
            if noiselevel is not None:
                data += np.random.normal(0, noiselevel, (n_samples, data.shape[1]))
                
            return data, freqs, target
        else:
            raise NotImplementedError("Only sine is supported at the moment")

class variable_freq_generator():
    def __init__(self, length, fs):
        self.length = length
        self.fs = fs

    def generate(self, 
                 n_samples = 100, 
                 max_frequencies = 5, 
                 noiselevel=None):
        freqs = []
        max_freq = self.fs/2
        t = np.arange(0, self.length, 1/self.fs)
        collect_freqs = []

        data = np.zeros((n_samples, len(t)))
        for i in range(n_samples):
            if np.random.rand() > 0.1:
                n_freqs = np.random.randint(1, max_frequencies+1)
                freqs = np.random.uniform(0.1, max_freq, n_freqs)
                amps = np.random.uniform(0.2, 1, n_freqs)
                collect_freqs.append(freqs)
                for amp, freq in zip(amps, freqs):
                    data[i] += amp*np.sin(2*np.pi*freq*t)
            else:
                collect_freqs.append([0])
        
        if noiselevel is not None:
            data += np.random.normal(0, noiselevel, (n_samples, data.shape[1]))
        return data, collect_freqs

class TStestGenerator():
    def __init__(self, length, fs, num_variables = 1):
        assert num_variables == 1, "Only one variable is supported at the moment"
        self.num_variables = num_variables
        self.length = length
        self.fs = fs
    
    def generate(self, 
                 n_samples = 100, 
                 type='sine', 
                 n_freqs = 2,
                 fix_frequency = True,
                 phase=0, 
                 use_digital_freqs = False,
                 noiselevel=None,
                 target_weights = None,
                 target_bias = None):
        if n_freqs > 1 and fix_frequency:
            if use_digital_freqs:
                true_freqs = rfftfreq(int(self.length/2)*self.fs, 1/self.fs)
                freq_1 = np.repeat(np.random.choice(true_freqs, 1), n_samples)[:, np.newaxis]
            else:
                freq_1 = np.expand_dims(np.repeat(np.array(np.random.uniform(0.1, self.fs/2)), n_samples), 1)
        else:
            if use_digital_freqs:
                true_freqs = rfftfreq(int(self.length/2)*self.fs, 1/self.fs)
                freq_1 = np.random.choice(true_freqs, n_samples)[:, np.newaxis]
            else:
                freq_1 = np.random.uniform(0.1, self.fs/2, (n_samples, 1))
        freqs = [freq_1]
        for _ in range(n_freqs-1):
            if use_digital_freqs:
                freq_2 = np.random.choice(true_freqs, n_samples)[:, np.newaxis]
            else:
                freq_2 = np.random.uniform(0.1, self.fs/2, (n_samples, 1))
            freqs.append(freq_2)
        amp = []
        for _ in range(n_freqs):
            amp.append(1)
            
        if target_weights is not None:
            standardized_freqs = np.concatenate(freqs, axis = 1)
            target = standardized_freqs@target_weights.T + target_bias + np.random.normal(0, 0.1, (n_samples, 1))
            # normalize target
            target = (target - np.min(target))/(np.max(target) - np.min(target))
        else:
            target = None

        if type == 'sine':
            t = np.repeat(np.expand_dims(np.arange(0, self.length, 1/self.fs), axis = 0), n_samples, axis = 0)
            data = np.zeros(t.shape)
            for i in range(n_freqs):
                freq = freqs[i]
                data += amp[i] * np.sin(2*np.pi*freq*t + phase)
            if noiselevel is not None:
                data += np.random.normal(0, noiselevel, (n_samples, data.shape[1]))
                
            return data, freqs, target
        else:
            raise NotImplementedError("Only sine is supported at the moment")

class RareModel(nn.Module):
    def __init__(self, t, f):
        self.t = t
        self.f = f
    def forward(self, x):
        out = torch.zeros(x.shape[0], x.shape[-1])
        out = torch.sum(x[:, self.f, self.t]**2, axis = 1)
        return out
        
def RareGenerator(saliency = 'time'): 
    data = arma_generate_sample(ar=[2, 0.5, 0.2, 0.1], ma = [2.], scale = 1.0, nsample=(50,50), axis = 1)
    target = np.zeros(50)
    if saliency == 'time':
        t = np.random.randint(0, 45)
        target[t:t+5] = np.sum(data[12:38, t:t+5]**2, axis = 0)
    elif saliency == 'feature':
        f = np.random.choice(np.arange(50), size = 5, replace = False)
        target[12:38] = np.sum(data[f, 12:38]**2, axis = 0)
    return data, target

def frequency_lrp_dataset(samples, length =  2560, noiselevel = 0.01, M_min=None, M_max = None, integer_freqs = True, return_ks = False, seed = 42):
    ks = np.array([5, 16, 32, 53])
    if not integer_freqs:
        # set seed
        np.random.seed(seed)
        ks = ks + np.random.uniform(0, 1, ks.shape)
        # remove seed
        np.random.seed(None)
    classes_ = powerset(ks)
    all_freqs = np.linspace(1, 60, 60, dtype = np.int32)
    for k in ks:
        idx = np.where(all_freqs == k)[0]
        all_freqs = np.delete(all_freqs, idx)
    data = np.zeros((samples, length))
    labels = []
    for i in range(samples):
        class_ = np.random.randint(0, len(classes_))
        freqs = np.array(classes_[class_])
        data[i] += np.random.normal(0, noiselevel, length)
        #data[i] += np.sum([np.sin(2*np.pi*freq/length*np.arange(0, length) + np.random.uniform(0, 2*np.pi)) for freq in freqs], axis = 0)
        # if M is a number then we add M random frequencies
        if M_min is not None:
            M = np.random.randint(M_min, M_max)
            if integer_freqs:
                # append to freqs
                freqs = np.append(freqs, np.random.choice(all_freqs, M-len(freqs), replace = False))
            else:
                # sample uniformly, but exclude a range of 1 Hz around frequencies in ks
                while len(freqs) < M:
                    f = np.random.uniform(1, 60)
                    if np.all(np.abs(ks - f) > 1):
                        freqs = np.append(freqs, f)

            data[i] += np.sum([np.sin(2*np.pi*freq/length*np.arange(0, length) + np.random.uniform(0, 2*np.pi)) for freq in freqs], axis = 0)
        labels.append(class_)
    if return_ks:
        return data, labels, ks
    return data, labels


def get_class_label(boolean_vector):

    # Initialize class label as 0
    class_label = 0
    
    # Iterate over the boolean vector and frequencies
    for i, present in enumerate(boolean_vector):
        if present:
            # Use bitwise OR to set the bit corresponding to the frequency
            class_label |= (1 << i)
    
    return class_label

def powerset(s):
    """Return the powerset of a set `s` in the order of binary representation."""
    # The length of the input set
    n = len(s)
    # Generate all subsets using combinations
    result = []
    for i in range(2**n):
        subset = [s[j] for j in range(n) if (i & (1 << j))]
        result.append(subset)
    return result

def DataSimulator(samples, length =  2560, noiselevel = 0.01, max_freq = 100, M_min = 10, M_max = 50, seed = 42):
    np.random.seed(seed)
    ks = np.sort(np.random.randint(0, max_freq, 4))
    np.random.seed(None)
    data = np.zeros((samples, length))
    labels = []
    all_freqs = []
    for i in range(samples):
        data[i] += np.random.normal(0, noiselevel, length)
        # if M is a number then we add M random frequencies
        if M_max is None:
            M = M_min
        else:
            M = np.random.randint(M_min, M_max)
        freqs = np.random.uniform(1, max_freq, M)
        class_dist = freqs[:, np.newaxis] - ks[np.newaxis, :]
        # check class label based on distance to ks, if distance is less than 1 then it is in the class
        class_freqs = (np.abs(class_dist) < 2.5).any(axis = 0)
        # get class index since we know all classes are the powerset of these frequencies 
        class_ = get_class_label(class_freqs)
        data[i] += np.sum([np.sin(2*np.pi*freq/length*np.arange(0, length) + np.random.uniform(0, 2*np.pi)) for freq in freqs], axis = 0)
        all_freqs.append(freqs)
        labels.append(class_)

    return data, labels, ks, all_freqs  

def DataSimulatorSimple(samples, length =  1000, Fs = 500, width = None, noiselevel = 0.01, M_min = 10, M_max = 50, max_freq = None):
    # divide frequency axis into 4 equal bins and select center frequencies
    if max_freq is None:
        max_freq = Fs/2
    ks = ks = np.linspace(max_freq/8, max_freq + max_freq/8, 4, endpoint = False)
    classes_ = powerset(ks)
    if width is None:
        width = Fs/16
    data = np.zeros((samples, length))
    labels = []
    all_freqs = []
    time_axis = np.arange(0, length)/Fs
    for i in range(samples):
        data[i] += np.random.normal(0, noiselevel, length)
        # if M is a number then we add M random frequencies
        if M_max is None:
            M = M_min
        else:
            M = np.random.randint(M_min, M_max)
        # sample a class
        class_ = np.random.randint(0, len(classes_))
        class_freqs_bins = classes_[int(class_)]
        freqs = []
        for bin in class_freqs_bins:
            freqs.append(np.random.uniform(bin - width, bin + width))
        while len(freqs) < M:
            # sample bin within which to sample
            f = np.random.uniform(0, max_freq)
            if not any(np.abs(np.array(ks) - f) < width):
                freqs.append(f)
        data[i] += np.sum([np.sin(2*np.pi*freq*time_axis + np.random.uniform(0, 2*np.pi)) for freq in freqs], axis = 0)
        all_freqs.append(freqs)
        labels.append(class_)

    return data, labels, ks, all_freqs  

def voigt(frequency, peak_frequency, gamma, sigma):
    """
    Voigt function to broaden the peak frequency across a band using a combination of Lorentzian and Gaussian shapes.
    
    Args:
        frequency: The frequency at which to evaluate the Voigt profile.
        peak_frequency: The center frequency of the Voigt peak.
        gamma: The Lorentzian width parameter.
        sigma: The Gaussian width parameter.
    
    Returns:
        Amplitude modulation value based on the Voigt profile.
    """
    return voigt_profile(frequency - peak_frequency, sigma, gamma)

# Lorentzian function definition
def lorentzian(frequency, peak_frequency, gamma):
    """
    Lorentzian function to broaden the peak frequency across a band.
    
    Args:
        frequency: The frequency at which to evaluate the Lorentzian.
        peak_frequency: The center frequency of the Lorentzian peak.
        gamma: The width parameter (broaden the peak).
    
    Returns:
        Amplitude modulation value based on the Lorentzian curve.
    """
    return (gamma ** 2) / ((frequency - peak_frequency) ** 2 + gamma ** 2)

# Function to divide the frequency axis into X regions
def divide_frequency_axis(sampling_frequency, num_regions):
    """
    Divide the frequency axis into X non-overlapping regions.
    
    Args:
        sampling_frequency: The total frequency range (from 0 to Nyquist frequency).
        num_regions: Number of regions to divide the frequency axis into.
    
    Returns:
        List of tuples (start_freq, end_freq) for each frequency region.
    """
    nyquist = sampling_frequency / 2
    region_width = nyquist / num_regions
    regions = [(i * region_width, (i + 1) * region_width) for i in range(num_regions)]
    return regions

# Function to generate time series data with noise regions modulated by Lorentzian curves and broadened peaks for salient regions
def generate_time_series_voigt(num_samples_per_class=100, num_frequencies=25, num_sample_regions = 3, salient_regions=None, 
                         all_regions=None, gamma=0.05, sigma = 0.1, noise_level=0.01, sampling_frequency=100, duration=1, seed = 42):
    """
    Generate a balanced time series dataset with noise regions modulated by Lorentzian curves and 
    broadened salient frequency regions.
    
    Args:
        num_samples_per_class: Number of time series samples per class to generate.
        num_frequencies: Number of frequency components in band.
        num_sample_regions: Number of regions to sample for each time series.
        salient_regions: List of tuples (start_freq, end_freq) for salient regions in frequency domain.
        all_regions: List of all frequency regions.
        gamma: Width parameter for the Lorentzian modulation (controls broadening).
        noise_level: Amplitude of background noise outside salient regions.
        sampling_frequency: Number of samples collected per second (sampling frequency).
        duration: Duration of each time series in seconds.
    
    Returns:
        time_series: Generated time series data (num_samples, num_timesteps).
        labels: Corresponding class labels (as integers representing the powerset class).
    """
    num_timesteps = int(sampling_frequency * duration)  # Total number of time steps based on sampling frequency
    num_regions = len(salient_regions)
    
    # Generate the powerset of the regions to represent the 16 classes
    powerset_classes = [list(combo) for r in range(num_regions + 1) for combo in combinations(range(num_regions), r)]
    num_classes = len(powerset_classes)
    
    time_series = np.zeros((num_samples_per_class * num_classes, num_timesteps))
    labels = np.zeros(num_samples_per_class * num_classes, dtype=int)
    
    t = np.linspace(0, duration, num_timesteps)  # Time array from 0 to duration with num_timesteps points
    noise_regions = [region for region in all_regions if region not in salient_regions]
    # generate noise regions
    noise_freq_bands = []
    np.random.seed(seed)
    phases = np.random.uniform(0, 2 * np.pi, len(noise_regions))
    i = 0
    for region_start, region_end in noise_regions:
        signal = np.zeros(num_timesteps)
        peak_frequency = (region_start + region_end) / 2  # Center of the noise region        
        frequencies_in_band = np.linspace(region_start, region_end, num_frequencies)
        for frequency in frequencies_in_band:
            amplitude = voigt(frequency, peak_frequency, gamma, sigma)  # Lorentzian modulation for noise
            signal += amplitude * np.sin(2 * np.pi * frequency * t + phases[i])
        i += 1
        noise_freq_bands.append(signal)
    # generate salient regions
    salient_freq_bands = []
    np.random.seed(seed)
    phases = np.random.uniform(0, 2 * np.pi, len(salient_regions))
    i = 0
    for region_start, region_end in salient_regions:
        signal = np.zeros(num_timesteps)
        peak_frequency = (region_start + region_end) / 2
        frequencies_in_band = np.linspace(region_start, region_end, num_frequencies)
        for frequency in frequencies_in_band:
            amplitude = voigt(frequency, peak_frequency, gamma, sigma)
            signal += amplitude * np.sin(2 * np.pi * frequency * t + phases[i])
        i += 1
        salient_freq_bands.append(signal)

    sample_idx = 0
    for class_idx, active_regions in enumerate(powerset_classes):
        for _ in range(num_samples_per_class):
            signal = np.zeros(num_timesteps)
            
            # Generate background noise modulated by Lorentzian curves in the non-salient regions
            num_noise_regions_to_sample = num_sample_regions - len(active_regions)
            np.random.seed(seed+sample_idx)
            num_noise_regions_to_sample = np.random.randint(0, num_noise_regions_to_sample, 1)
            if num_noise_regions_to_sample > 0:
                np.random.seed(seed+sample_idx)
                sampled_noise_regions = np.random.choice(len(noise_regions), num_noise_regions_to_sample, replace=False)
                for region_idx in sampled_noise_regions:
                    signal += noise_freq_bands[region_idx]

            # Add the salient peaks for active regions modulated by Lorentzian curves
            for region in active_regions:
                signal += salient_freq_bands[region]
            # Store the generated signal and the corresponding class label
            np.random.seed(seed+sample_idx)
            time_series[sample_idx, :] = signal + noise_level * np.random.randn(num_timesteps)
            labels[sample_idx] = class_idx  # The class label corresponds to the powerset index
            sample_idx += 1
    # shuffle the data
    np.random.seed(seed)
    idx = np.random.permutation(len(time_series))
    time_series = time_series[idx]
    labels = labels[idx]
    np.random.seed(None)
    return time_series, labels

def generate_time_series_voigt_noisy(num_samples_per_class=100, num_frequencies=25, num_sample_regions = 3, salient_regions=None, 
                                    all_regions=None, gamma=0.05, sigma = 0.1, noise_level=0.01, sampling_frequency=100, duration=1, seed = 42,
                                    scale = 1):
    """
    Generate a balanced time series dataset with noise regions modulated by Lorentzian curves and 
    broadened salient frequency regions.
    
    Args:
        num_samples_per_class: Number of time series samples per class to generate.
        num_frequencies: Number of frequency components in band.
        num_sample_regions: Number of regions to sample for each time series.
        salient_regions: List of tuples (start_freq, end_freq) for salient regions in frequency domain.
        all_regions: List of all frequency regions.
        gamma: Width parameter for the Lorentzian modulation (controls broadening).
        noise_level: Amplitude of background noise outside salient regions.
        sampling_frequency: Number of samples collected per second (sampling frequency).
        duration: Duration of each time series in seconds.
    
    Returns:
        time_series: Generated time series data (num_samples, num_timesteps).
        labels: Corresponding class labels (as integers representing the powerset class).
    """
    num_timesteps = int(sampling_frequency * duration)  # Total number of time steps based on sampling frequency
    num_regions = len(salient_regions)
    
    # Generate the powerset of the regions to represent the 16 classes
    powerset_classes = [list(combo) for r in range(num_regions + 1) for combo in combinations(range(num_regions), r)]
    num_classes = len(powerset_classes)
    
    time_series = np.zeros((num_samples_per_class * num_classes, num_timesteps))
    labels = np.zeros(num_samples_per_class * num_classes, dtype=int)
    
    t = np.linspace(0, duration, num_timesteps)  # Time array from 0 to duration with num_timesteps points
    noise_regions = [region for region in all_regions if region not in salient_regions]
    sample_idx = 0
    for class_idx, active_regions in enumerate(powerset_classes):
        for _ in range(num_samples_per_class):
            signal = np.zeros(num_timesteps)
            
            # Generate background noise modulated by Lorentzian curves in the non-salient regions
            num_noise_regions_to_sample = num_sample_regions - len(active_regions)
            if seed is not None:
                np.random.seed(seed+sample_idx)
            num_noise_regions_to_sample = np.random.randint(0, num_noise_regions_to_sample, 1)
            if num_noise_regions_to_sample > 0:
                if seed is not None:    
                    np.random.seed(seed+sample_idx)
                sampled_noise_regions = np.random.choice(len(noise_regions), num_noise_regions_to_sample, replace=False)
                phases = np.random.uniform(0, 2 * np.pi, len(sampled_noise_regions))
                for i, region_idx in enumerate(sampled_noise_regions):
                    region_start, region_end = noise_regions[region_idx]
                    center_frequency = (region_start + region_end) / 2
                    a, b = (region_start+2*gamma - center_frequency) / scale, (region_end-2*gamma - center_frequency) / scale
                    peak_frequency = truncnorm.rvs(a, b, loc=center_frequency, scale=scale)
                    frequencies = np.linspace(region_start, region_end, num_frequencies)
                    for frequency in frequencies:
                        amplitude = voigt(frequency, peak_frequency, gamma, sigma)  # Lorentzian modulation for noise
                        signal += amplitude * np.sin(2 * np.pi * frequency * t + phases[i])
                    
            # Add the salient peaks for active regions modulated by Lorentzian curves
            phases = np.random.uniform(0, 2 * np.pi, len(active_regions))
            for i, region in enumerate(active_regions):
                region_start, region_end = salient_regions[region]
                center_frequency = (region_start + region_end) / 2
                a, b = (region_start+2*gamma - center_frequency) / scale, (region_end-2*gamma - center_frequency) / scale
                peak_frequency = truncnorm.rvs(a, b, loc=center_frequency, scale=1)
                frequencies = np.linspace(region_start, region_end, num_frequencies)
                for frequency in frequencies:
                    amplitude = voigt(frequency, peak_frequency, gamma, sigma)
                    signal += amplitude * np.sin(2 * np.pi * frequency * t + phases[i])
            # Store the generated signal and the corresponding class label
            time_series[sample_idx, :] = signal + noise_level * np.random.randn(num_timesteps)
            labels[sample_idx] = class_idx  # The class label corresponds to the powerset index
            sample_idx += 1
    # shuffle the data
    idx = np.random.permutation(len(time_series))
    time_series = time_series[idx]
    labels = labels[idx]
    np.random.seed(None)
    return time_series, labels

def generate_time_series_width(samples, duration, fs, order = 2, noise_level = 0.1):
    length = duration * fs
    data = np.zeros((samples, length))
    classes = np.zeros(samples)
    class_bands = []
    for i in range(samples):
        class_ = np.random.randint(0, 3)
        # sample 2 non-overlapping bands with width between 10 and 20 Hz
        band_freqs = []
        if class_ in [1, 2]:
            widths = [[9, 11], [14, 16]]
            width = np.random.uniform(widths[class_-1][0], widths[class_-1][1])
            freq = np.random.uniform(widths[class_-1][0]+1, fs / 2 - widths[class_-1][1] - 1)
            band_freqs.append([freq - width / 2, freq + width / 2])
        elif class_ == 3:
            widths = [[9, 11], [14, 16]]
            for width_range in widths:
                width = np.random.uniform(width_range[0], width_range[1])
                freq = np.random.uniform(width+1, fs / 2 - width - 1)
                if all([freq + 10 < band[0] or freq - 10 > band[1] for band in band_freqs]):
                    band_freqs.append([freq - width / 2, freq + width / 2])
        filt_sample = np.random.randn(length) * noise_level
        for band in band_freqs:
            try:
                sos = butter(order, band, btype='bandpass', fs=fs, output='sos')
            except:
                print(band)
                raise ValueError
            sample = np.random.randn(length)
            filt_sample += sosfiltfilt(sos, sample)
        data[i] = filt_sample
        classes[i] = class_
    return data, classes, class_bands