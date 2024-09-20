import h5py
from matplotlib.ticker import FuncFormatter
import time_resolved_CD_lib_08 as lib
#import style_sheet as style
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from func_lib import load_file
from HDF5Data import HDF5Data
from ElzerData import ElzerData
from SchmittTrigger import SchmittTrigger
from sklearn.preprocessing import minmax_scale
from scipy.ndimage import gaussian_filter1d
import sys
import os
import time

"""
with h5py.File(hdf5_file_path, 'r') as file:
    #print(file['Traces/Alazar Slytherin - Ch1 - Data_N'].keys())
    print(file['Traces/Alazar Slytherin - Ch1 - Data_N'][0])

group = "Traces"
dataset = "Traces/Alazar Slytherin - Ch1 - Data"
time_spacing= "Traces/Alazar Slytherin - Ch1 - Data_t0dt"
"""

def create_cut_file(file_path, dir):
    start_time = time.time()
    test = ElzerData(readpath=file_path, wdir=dir, t_ini=1000, t_read=1000,  sampling_rate=5)
    test.create_new_file_only_traces('2.1T_elzerman')
    elapsed_time = time.time() - start_time
    print(elapsed_time)

def main():
    # Get the current working directory
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, 'data')
    figures_dir = os.path.join(current_dir, 'Figures_bias')

    file_name = '538_2.1T_TP_vs_tload'
    hdf5_file_path = os.path.join(current_dir, data_dir, '{}.hdf5'.format(file_name))

    create_cut_file(hdf5_file_path, data_dir)


if __name__ == "__main__":
    main()