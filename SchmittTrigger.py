import h5py
import numpy as np
import itertools
import os
from HDF5Data import HDF5Data
from sklearn.preprocessing import normalize
from scipy.signal import find_peaks

class SchmittTrigger:

    def __init__(self, data=None, trace=None, sampling_rate=None):
        self.data = data
        self.trace = trace
        self.sampling_rate = sampling_rate

    def set_data(self, data, group_name, dataset_name):
        self.data = data
        if 'traces' in dataset_name.lower():
            self.trace = data[f'{group_name}/{dataset_name}']
            print('Shape of traces:', self.trace.shape)

    def set_sampling_rate(self, sr):
        self.sampling_rate = sr


    def normalize_data(self):
        pass

    def correct_fft_noise(self, trace, f_low=0, f_high=10000):
        fft_signal = np.fft.fft(trace)
        frequencies = np.fft.fftfreq(len(trace), d=(1/self.sampling_rate))

        fft_signal_shifted = np.fft.fftshift(fft_signal)
        frequencies_shifted = np.fft.fftshift(frequencies)

        #fft_signal_shifted[(np.abs(frequencies_shifted) > f_low) & (np.abs(frequencies_shifted) < f_high)] = 0

        freq_peaks, heights = find_peaks(np.abs(fft_signal_shifted), height=80, distance=10)
        fft_signal_shifted_corr = fft_signal_shifted

        if len(freq_peaks) == 1:
            fft_signal_shifted_corr[freq_peaks[0] - 6: freq_peaks[0] + 6] = 0
            print(f'Corrected for:', freq_peaks)

        elif len(freq_peaks) == 3:

            fft_signal_shifted_corr[freq_peaks[0] - 6: freq_peaks[0] + 6] = 0
            fft_signal_shifted_corr[freq_peaks[1] - 6: freq_peaks[1] + 6] = 0
            fft_signal_shifted_corr[freq_peaks[2] - 6: freq_peaks[2] + 6] = 0
            print(f'Corrected for:', freq_peaks)

        elif len(freq_peaks) == 5:

            fft_signal_shifted_corr[freq_peaks[0] - 6: freq_peaks[0] + 6] = 0
            fft_signal_shifted_corr[freq_peaks[1] - 6: freq_peaks[1] + 6] = 0
            fft_signal_shifted_corr[freq_peaks[2] - 6: freq_peaks[2] + 6] = 0
            print(f'Corrected for:', freq_peaks)

        elif len(freq_peaks) > 5:
            fft_signal_shifted_corr[freq_peaks[0] - 6: freq_peaks[0] + 6] = 0
            fft_signal_shifted_corr[freq_peaks[1] - 6: freq_peaks[1] + 6] = 0
            fft_signal_shifted_corr[freq_peaks[2] - 6: freq_peaks[2] + 6] = 0
            fft_signal_shifted_corr[freq_peaks[-1] - 6: freq_peaks[-1] + 6] = 0
            #print(f'Corrected for:', freq_peaks)

        else:
            print(freq_peaks)

        original_phase = np.angle(fft_signal_shifted)
        modified_magnitude = np.abs(fft_signal_shifted_corr)

        modified_magnitude = np.maximum(modified_magnitude, 0)

        new_fft_data = modified_magnitude * np.exp(1j * original_phase)

        # Unshift the new FFT data before performing the inverse FFT
        new_fft_data_unshifted = np.fft.ifftshift(new_fft_data)

        # Perform the inverse FFT to get the new time domain signal
        new_time_domain_signal = np.fft.ifft(new_fft_data_unshifted).real

        self.trace = new_time_domain_signal

    def detect_events(self):
        pass

    def gaussian_fit(self):
        pass

    def plot_hist(self):
        pass

    def plot_tunnel_rates(self):
        pass