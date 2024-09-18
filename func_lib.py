import h5py as h5
import numpy as np
import HDF5Data as HDF5Data

def load_file(hdf5Data):
    file = hdf5Data
    file.set_measure_dim()
    print(file.measure_dim)
    #extract traces from file
    #file.set_traces()
    #traces_array = file.traces
    #extract time trace from file
    file.set_traces_dt()
    time_spacing = file.traces_dt
    #get t_burst
    file.set_arrays()
    file.set_array_tags()
    t_burst = file.arrays[1]
    print(file.array_tags)
    #save traces in data_bias
    #file.save_traces_in_wdir()
    file.trace_loading_with_reference()
    traces_hdf = file.trace_reference
    print(file.trace_reference)
    table = file.trace_order
    #reset file
    file.reset()
    return time_spacing, traces_hdf, t_burst[0], table

