import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
data_dir = os.path.join(current_dir, 'data_bias')
sys.path.append(data_dir)
figures_dir = os.path.join(current_dir, 'Figures_bias')
sys.path.append(figures_dir)

from matplotlib.ticker import FuncFormatter
import time_resolved_CD_lib_08 as lib
import style_sheet as style
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

'''
Cells can be executed from top to bottom
'''

#%%
alpha = 0.0538 # lever arm in eV/V FG
alpha = 0.03509 # lever arm in eV/V PG
k_B_eV = 8.61733*1e-5 #boltzmann constant in eV/K

'''
Import file here
    in first run set read_file = True, will create numpy file and give data dimensions including n_FG 
    in second run set read_file = False, will import time from hdf5 and use numpy file
'''

file_name = "476_FG_trace_time_traces"
FG_min = 7.89 # FG range in V
FG_max = 7.9
n_FG = 51 # number of FG traces
B_perp = 2.7 # [T]
start_params = [3.5e5,-0.82, 0.01, 4.3e5,-0.74,0.01,10] # start parameters for Gaussian fit, must be determined manually 
bounds_gaussian = ([0,-0.1,0],[1000000,0.1,0.04])
bounds_double_gaussian = ([0,-1,0,0,-1,0,0],[1000000,-0.5,0.04,1000000,-0.5,0.04,1000])
n_FG_list = np.arange(19,38,1) # indices of FG values which will be used for histogram (read from map below)

hdf5_file_path = os.path.join(current_dir, 'data_bias', '{}.hdf5'.format(file_name))
np_file_path = os.path.join(current_dir, 'data_bias', '{}.npy'.format(file_name))

group = "Traces"
dataset = "Traces/Alazar Slytherin - Ch1 - Data" 
time_spacing= "Traces/Alazar Slytherin - Ch1 - Data_t0dt" 

FG_array = lib.make_FG_array(n_FG, FG_min, FG_max)

read_file = False

if read_file == True: 
    time, traces_array = lib.read_file(hdf5_file_path, group, dataset, time_spacing, False)
    print(np.shape(traces_array))
    np.save(os.path.join(data_dir, "{}.npy".format(file_name)), traces_array)
elif read_file == False: 
    traces_array = np.load(np_file_path) 
    n_points = len(traces_array[0])
    time_list = lib.read_time(hdf5_file_path, group, dataset, time_spacing, n_points)

#%%

'''
Code snippet for plotting map with histograms. From this you can choose trace number range (ideally containing two peaks without noticeable jumps)
from which double Gaussian fit will be performed. adjust n_FG_list accordingly. 
'''
plt.close("all")
histograms = np.array([np.histogram(trace, bins=100)[0] for trace in traces_array])

plt.figure()
plt.imshow(histograms, aspect='auto', cmap='viridis')
plt.colorbar()
plt.xlabel('Detector signal (a.u.)')
plt.ylabel('Trace Index')
plt.show()

#%%
# Compute the FFT of the signal for correction of low frequency noise
# If used, the arrays used for following code snippets need to be adjusted

def correct_fft_noise(time, trace):
    fft_signal = np.fft.fft(trace)
    frequencies = np.fft.fftfreq(len(trace), d=(time[1] - time[0]))
    
    fft_signal_shifted = np.fft.fftshift(fft_signal)
    frequencies_shifted = np.fft.fftshift(frequencies)
    
    freq_peaks, heights = find_peaks(np.abs(fft_signal_shifted), height = 10000, distance = 10)
    fft_signal_shifted_corr = fft_signal_shifted
    
    if len(freq_peaks) == 3: 
        
        fft_signal_shifted_corr[freq_peaks[0]-6 : freq_peaks[0]+6] = 0 
        fft_signal_shifted_corr[freq_peaks[2]-6 : freq_peaks[2]+6] = 0 
        
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
    
    return new_time_domain_signal
    
new_traces_array = []    
for i in range(n_FG):
    trace = traces_array[i]
    
    trace_corr = correct_fft_noise(time_list, trace)
    
    new_traces_array.append(trace_corr)
    

#%%
'''
create histogram of selected traces and set start parameters for fit below
'''

plt.close("all")
# Set values m and b for determining parameter a, giving the threshold for trigger. With m=3/7 and b=-6/7 we obtain a(SNR=2)=0 and a(SNR=9)=3
m = 3/7
b = -6/7
n_bins = 150
trace = np.array(traces_array)[n_FG_list].flatten()

hist, bins = np.histogram(trace, bins = n_bins, density = False)
bin_centers = 0.5*(bins[1:] + bins[:-1])

hist_smoothed = lib.moving_average(hist, 5) # smooth histogram data, might have to be adjusted depending on nb of bins
style.width(0.5,1)
style.lines()
style.fonts()
plt.scatter(bin_centers,hist_smoothed/1e5, s = 0.5, c = style.return_rwth_color(0))
#%%

'''
Fit double Gaussian to histogram using start parameters and determine threshold parameter a
'''

params, cov = lib.fit_double_gaussian(bin_centers, hist_smoothed, start_params, bounds_double_gaussian)
print(params)

plt.plot(bin_centers, lib.double_gaussian(bin_centers, *params)/1e5, c = style.return_rwth_color(45))

plt.plot(bin_centers, lib.gaussian(bin_centers, *params[0:3])/1e5)
plt.plot(bin_centers, lib.gaussian(bin_centers, *params[3:6])/1e5)

snr = lib.snr_calc(params)
a = lib.det_a(snr, m, b)

print(snr)
print(a)
plt.xlim(min(bin_centers),max(bin_centers))
plt.xlabel("Detector signal (mV)")
plt.ylabel(r"Counts $(10^5)$")
plt.title("SNR = {}".format(round(snr,2)))
#style.savefig(os.path.join(figures_dir, "{}_histogram.pdf".format(file_name)), format = "pdf")
#%%

'''
Detection algorithm. 
In case of corrected frequencies, traces_array needs to be corrected first.
'''
thresh_lower = params[1]+a*params[2]
thresh_upper = params[4]-a*params[5]

gamma_up_list, gamma_down_list = [],[]


for i in range(len(FG_array)):
    x_result, diff_result, up_list, down_list, up_times, down_times, rectangular_signal = lib.detect_events_vec(time_list, traces_array[i], thresh_upper, thresh_lower)
    
    gamma_up = lib.gamma(up_times)[0]
    gamma_down = lib.gamma(down_times)[0]
    
    gamma_up_list.append(gamma_up)
    gamma_down_list.append(gamma_down)
     
#%%    
'''
plot tunneling rates. In case of zero bias, Fermi function can be fitted. 
'''
E_array = alpha*FG_array
plt.close("all")
style.width(1,0.6)
style.lines()
style.fonts()
plt.plot(1000*E_array, 0.001*np.array(gamma_up_list), ls = '--', marker = 'o', markersize = 3, color = style.return_rwth_color(0), label = r"$\Gamma_\mathrm{in}$")
plt.plot(1000*E_array,  0.001*np.array(gamma_down_list),ls = '--', marker = 'o', markersize = 3, color = style.return_rwth_color(45),label = r"$\Gamma_\mathrm{out}$")
#plt.plot(1000*E_array[0:11],  0.001*np.array(gamma_down_list[0:11]), ls = '--', marker = 'o', markersize = 3, color = style.return_rwth_color(7),label = "Not considered in fit")
params_fermi_up, _ = lib.fit_fermi(E_array,gamma_up_list , [3.3e3, 0.27706, -5e-5])
#params_fermi_down, _ = lib.fit_fermi(E_array[15:-5],gamma_down_list[15:-5] , [4.5e3, 0.27704, -5e-5])
#plt.plot(1000*E_array, 0.001*lib.fermi(E_array, *params_fermi_up),color = style.return_rwth_color(0))
#plt.plot(1000*E_array, 0.001*lib.fermi(E_array, *params_fermi_down), color = style.return_rwth_color(45))

plt.ylabel(r"Tunneling Rate $\Gamma$ (kHz)")
plt.xlabel(r"$E$ (meV)")
#plt.xlim(1000*E_array[20], 1000*E_array[50])
# plt.xlim(1000*min(E_array), 1000*max(E_array))
plt.legend()
print(params_fermi_up[2]/k_B_eV)
style.savefig(os.path.join(figures_dir, "{}_Tunneling_rates.png".format(file_name)),format = "png")

#%% find optimal a 
plt.close("all")
trace_test = traces_array[25]

a_range = np.linspace(0.05,2,100)
n_events_list = []


for a in a_range:
    thresh_lower = params[1]+a*params[2]
    thresh_upper = params[4]-a*params[5]

    x_result, diff_result, up_list, down_list, up_times, down_times, rectangular_signal = lib.detect_events_vec(time_list, trace_test, thresh_upper, thresh_lower)
    
    n_events = len(up_list) + len(down_list)
    
    n_events_list.append(n_events)
    
plt.figure()
plt.scatter(a_range,n_events_list)    

    