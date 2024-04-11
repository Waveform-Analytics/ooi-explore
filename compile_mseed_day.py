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
    """
    Load a miniseed file, generate a spectrogram, and return an xarray object including a start time coordinate extracted from the file name. 
    
    The particular filtering and spectrogram parameters are hard-coded for the time being. These could be made flexible in a future version.
    
    Args:
        file (str): path to miniseed file
        
    Returns:
        xarray object
    
    """
    
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


def parallel_process(data, func=make_xarray, sub_segment_length=6, n_processes=None):
    """
    Read in a list of miniseed files and build a 3D xarray object containing a spectrogram for each 5-minute file. Set up python to distribute analysis tasks across all available CPUs. 
    
    Args:
        data (list): A list of paths to individual miniseed files
        func (function name, Optional): A function to use in parallel processsing, it should read in a miniseed file path and return an xarray object. Default is make_xarray, defined above. 
        sub_segment_length (int, Optional): Integer number of segments to use in each of the parallel processing segments. These are handled in segments to reduce the number of arrays being concatenated (and held in memory) at any one point in time. Default is 6. 
    
    """
    if n_processes is None:
        n_processes = multiprocessing.cpu_count() 

    pool = multiprocessing.Pool(processes=n_processes)
    
    # Initialize a list to hold the xarray sub-arrays
    results_all = []
    
    # Break into the requested number of 5-minute segments at a time (30-mins total)
    segments = [data[i:i + sub_segment_length] for i in range(0, len(data), sub_segment_length)]
    for idx, segment in enumerate(segments):
        print( "Segment " + str(idx+1) + " of " + str(len(segments)) )
        results = pool.map(func, segment)
        # Use dask for efficiency
        dask_datasets = [da.chunk({'seconds': 50, 'frequency': 50}) for da in results]
        results_all.append(xr.concat(dask_datasets, dim='time'))
        del results
        
    pool.close() 
    pool.join()
    
    results_full = xr.concat(results_all, dim='time').sortby('time')

    return results_full


def mseed_to_xarray(instrument, year, month, day):
    """
    Given an instrument name and a day, return an xarray of spectrograms.
    
    Args:
        instrument (str): Instrument name/ID
        year (int): 4-digit year
        month (int): month number
        day (int): day of month 
        
    Returns:
        xarray object
    """
    date = f"{year}/{month:02}/{day:02}"  
    
    # Build path to the requested date folder
    instrument_day_path = "../../ooi/san_data/" + instrument + "/" + date
    
    # Look for all the mseed files in the folder provided above
    mseed_files = find_mseed_files(instrument_day_path)
    
    # Assembe all spectrograms into a single xarray object for one day
    return parallel_process(mseed_files)


def mseed_to_netcdf(instrument, year, month, day, file_name):
    """
    Given an instrument name and a day, generate spectrograms for each 5-minute segment and export to the requested file location
    
    Args:
        instrument (str): Instrument name/ID
        year (int): 4-digit year
        month (int): month number
        day (int): day of month 
        file_name (str): Path to the output file location. Output file should have a .nc extension
    
    Returns: None
    
    """
    # Generate an xarray of spectrograms
    results = mseed_to_xarray(instrument, year, month, day)
    
    # Save to netcdf format
    results.to_netcdf("data/" + instrument + "-YMD." + date.replace("/",".") + ".nc")


if __name__ == "__main__":
    
    # If you run this python file as a script, this section will run and call the functions defined above.
    
    # Path to requested instrument/date
    instrument = "RS03AXBS-LJ03A-09-HYDBBA302"
    date = "2024/01/05"
    instrument_day_path = "../../ooi/san_data/" + instrument + "/" + date
    
    # Look for all the mseed files in the folder provided above
    mseed_files = find_mseed_files(instrument_day_path)
    
    # Use a subset of miniseed files for initial testing
    mseed_files_sub = mseed_files[0:100]
    
    # Assembe all spectrograms into a single xarray object for one day
    results = parallel_process(mseed_files_sub)
    
    # Save to netcdf format
    results.to_netcdf("data/" + instrument + "-YMD." + date.replace("/",".") + ".nc")
