from obspy import read
import xarray as xr
import os
from scipy import signal
import numpy as np
from datetime import datetime
import dask.array as da
import multiprocessing


def find_mseed_files(directory):
    """
    Recursively location and list all files with the extension *.mseed (miniseed) within the provided directory
    
    Args:
        directory (str): path to the directory containing miniseed files or folders containinig miniseed files

    Returns:
        list: list of paths to miniseed files
    
    """
    mseed_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".mseed"):
                mseed_files.append(os.path.join(root, file))
    return mseed_files


def make_xarray(file):
    
    time_string = file.split("YDH-")[1].split('.mseed')[0]
    time_val = datetime.strptime(time_string, '%Y-%m-%dT%H:%M:%S.%fZ')
    
    stream = read(file)
    data = stream.detrend().split().decimate(16).merge(method=1, fill_value="interpolate").decimate(10)
    
    fs = data[0].stats["sampling_rate"]

    # Set spectrogram parameters
    segment_length_seconds = .5
    nperseg = int(segment_length_seconds * fs)
    noverlap_percent = 15
    noverlap = int(nperseg * noverlap_percent/100)

    f, t, Sxx = signal.spectrogram(data[0].data, fs, nperseg = nperseg, noverlap = noverlap)
    array_out = xr.DataArray(np.log10(Sxx), 
                           dims=('frequency', 'seconds'),
                           coords={'frequency':f, 'seconds':t})
    array_out.coords['time'] = time_val

    return array_out



if __name__ == "__main__":
    
    instrument_day_path = "../../ooi/san_data/RS03AXBS-LJ03A-09-HYDBBA302/2024/01/05"