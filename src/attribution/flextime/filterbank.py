import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, fft

class FilterBank:
    def __init__(self, numtaps, fs, numbanks, time_length, order = 2, filter_type = 'fir', bandwidth = None, **kwargs):
        # check inputs
        assert numbanks >= 2, 'Number of banks must be at least 2'
        assert fs > 0, 'Sampling frequency must be greater than 0'
        assert numtaps > 0, 'Number of taps must be greater than 0'
        assert numtaps % 2 == 1, 'Number of taps must be odd'
        assert filter_type in ['fir', 'butter'], 'Filter type must be either fir or butter'
        assert order > 0, 'Order must be greater than 0'

        self.numtaps = numtaps
        self.order = order
        self.fs = fs
        self.numbanks = numbanks
        self.group_delay = (numtaps - 1) // 2  # filter is symmetric
        self.banks = []
        if bandwidth is None:
            self.bandwidth = (fs / 2) /  numbanks
        else:
            self.bandwidth = bandwidth
        self.shift = (fs / 2) / numbanks
        self.filter_type = filter_type
        self.time_length = time_length

        # create filterbank
        # end bands are 0 and fs/2
        # create first filter as low pass
        if filter_type == 'fir':
            self.create_fir_filterbank()
        elif filter_type == 'butter':
            self.create_butter_filterbank()

    def create_fir_filterbank(self):
        # create filterbank
        # end bands are 0 and fs/2
        # create first filter as low pass
        h = signal.firwin(self.numtaps, self.bandwidth, fs=self.fs, pass_zero='lowpass')
        self.banks.append(h)
        band_start = self.bandwidth

        for _ in range(1, self.numbanks - 1):
            h = signal.firwin(self.numtaps, [band_start, band_start + self.bandwidth], fs=self.fs, pass_zero='bandpass')
            self.banks.append(h)
            band_start += self.shift

        # create last filter as high pass
        h = signal.firwin(self.numtaps, band_start, fs=self.fs, pass_zero='highpass')
        self.banks.append(h)
    
    def create_butter_filterbank(self):
        # create filterbank
        # end bands are 0 and fs/2
        # create first filter as low pass
        h = signal.butter(self.order, self.bandwidth, btype='low', fs=self.fs, output='sos')
        self.banks.append(h)
        band_start = self.bandwidth

        for _ in range(1, self.numbanks - 1):
            h = signal.butter(self.order, [band_start, band_start + self.bandwidth], btype='band', fs=self.fs, output='sos')
            self.banks.append(h)
            band_start += self.shift

        # create last filter as high pass
        h = signal.butter(self.order, band_start, btype='high', fs=self.fs, output='sos')
        self.banks.append(h)
    
    def get_collect_filter_response(self, mask = None):
        worN = len(fft.rfftfreq(self.time_length, 1/self.fs))
        if len(mask.shape) == 1:
            mask = mask[np.newaxis,:]
        collect_freq_resp = np.zeros((worN, *mask.shape))
        for j in range(mask.shape[0]):
            for i, h in enumerate(self.banks):
                if self.filter_type == 'fir':
                    w, H = signal.freqz(h, 1, worN=worN)
                else:
                    w, H = signal.sosfreqz(h, worN=worN, include_nyquist = True)
                H = abs(H)*mask[j, i] if mask is not None else abs(H)
                collect_freq_resp[:, j, i] = H

        return np.sum(collect_freq_resp, axis=-1)
    
    def plot_collect_filter(self, ax = None, mask = None, time_dim = -1):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        collect_freq_resp = np.zeros((2000, self.numbanks))
        for i, h in enumerate(self.banks):
            if self.filter_type == 'fir':
                w, H = signal.freqz(h, 1, worN=2000)
            else:
                w, H = signal.sosfreqz(h, worN=2000)
            H = abs(H)*mask[i] if mask is not None else abs(H)
            collect_freq_resp[:, i] = H
        ax.plot((self.fs * 0.5 / np.pi) * w, np.sum(collect_freq_resp, axis=1))
        ax.set_title('Sum of all filters')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Gain')
        
    def plot_filter_bank(self, mask=None, savepath = None, plot_sum = True, suptitle = False, figsize = None, plot_db = True, plot_gain = True, title = True):
        if plot_sum:
            n_figs = 3
            size = (10, 8)
        elif plot_db and plot_gain:
            n_figs = 2
            size = (10, 6)
        else:
            n_figs = 1
            size = (10, 4)
        if figsize is not None:
            size = figsize
        fig, ax = plt.subplots(n_figs, 1, figsize=size)
        if suptitle:
            fig.suptitle(f'Filterbank with {self.numbanks} banks, {self.numtaps} taps, {self.bandwidth:.2f} bandwidth')
        collect_freq_resp = np.zeros((2000, self.numbanks))
        for i, h in enumerate(self.banks):
            if self.filter_type == 'fir':
                w, H = signal.freqz(h, 1, worN=2000)
            else:
                w, H = signal.sosfreqz(h, worN=2000)
            H = abs(H)*mask[i] if mask is not None else abs(H)
            collect_freq_resp[:, i] = H
            if plot_gain and not plot_db:
                ax.plot((self.fs * 0.5 / np.pi) * w, H)
                if title:
                    ax.set_title('Magnitude Response')
                ax.set_xlabel('Frequency [Hz]')
                ax.set_ylabel('Gain')
                ax.grid(True)
            elif plot_db and not plot_gain:
                H_db = 20 * np.log10(H)
                ax.plot((self.fs * 0.5 / np.pi) * w, H_db)
                ax.set_ylim(-80, 1)
                if title:
                    ax.set_title('Magnitude Response')
                ax.set_xlabel('Frequency [Hz]')
                ax.set_ylabel('Gain [dB]')
                ax.grid(True)
            elif plot_db and plot_gain:
                ax[0].plot((self.fs * 0.5 / np.pi) * w, H)
                H_db = 20 * np.log10(H)
                ax[1].plot((self.fs * 0.5 / np.pi) * w, H_db)       
                ax[1].set_ylim(-80, 1)         
                if title:
                    ax[0].set_title('Magnitude Response')
                ax[1].set_xlabel('Frequency [Hz]')
                ax[0].set_ylabel('Gain')
                ax[1].set_ylabel('Gain [dB]')
                ax[0].grid(True)

                ax[1].grid(True)

        # plot the sum of all filters
        if plot_sum:
            ax[2].plot((self.fs * 0.5 / np.pi) * w, np.sum(collect_freq_resp, axis=1))
            ax[2].set_title('Sum of all filters')
            ax[2].set_xlabel('Frequency [Hz]')
            ax[2].set_ylabel('Gain')

        
        fig.tight_layout()
        if savepath is not None:
            fig.savefig(savepath)
        fig.show()

    def apply_filter_bank(self, x, time_dim = -1, adjust_for_delay = False, plot=False):
        y = np.zeros((*x.shape, self.numbanks))
        for inx, h in enumerate(self.banks):
            if self.filter_type == 'fir':
                y_temp = signal.lfilter(h, 1, x, axis = time_dim)
            else:
                y_temp = signal.sosfilt(h, x, axis = time_dim)
            if adjust_for_delay:
                y_temp = np.roll(y_temp, -self.group_delay, axis = time_dim)
            y[..., inx] = y_temp

        if plot:
            # plot each individual filter output
            fig, ax = plt.subplots(self.numbanks, 1, figsize=(16, 10))
            for inx in range(self.numbanks):
                title = f'Filter {inx} - band: {inx*self.bandwidth} - {(inx+1)*self.bandwidth} Hz'
                ax[inx].plot(y[:, inx])
                ax[inx].set_title(title)
                ax[inx].grid(True)

            fig.tight_layout()
            fig.show()
        return y

    def forward(self, x, mask=None, adjust_for_delay = False, time_dim = -1):
        # apply filterbank on the signal
        y = self.apply_filter_bank(x, adjust_for_delay = adjust_for_delay, time_dim=time_dim)
        # apply mask
        if mask is not None:
            if len(mask.shape) == 1:
                mask = mask[np.newaxis, :]
                y = y * mask
            else:
                y = y * mask[:, np.newaxis, :]
        return np.sum(y, axis=-1)