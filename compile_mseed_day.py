from obspy import read
import xarray as xr
import os
from scipy import signal
import numpy as np
from datetime import datetime
import dask.array as da
import multiprocessing
from typing import List, Callable, Optional, Union



def find_mseed_files(directory):
    """
    Recursively locate and list all files with the extension *.mseed (miniseed) within the provided directory
    
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


def make_xarray(file: str, option: int = 2) -> xr.DataArray:
    """
    Load a miniseed file, generate a spectrogram, and return an xarray object 
    including a start time coordinate extracted from the file name.
    
    The particular filtering and spectrogram parameters are hard-coded for the
    time being. These could be made flexible in a future version.
    
    Args:
        file (str): Path to the miniseed file.
        option (int): 1 for full sample rate, 2 for custom decimation for
            fin whale calls. Default is 2.
        
    Returns:
        xr.DataArray: An xarray DataArray containing the spectrogram.
    """
    # Extract time from filename
    try:
        time_string = file.split("YDH-")[1].split('.mseed')[0]
        time_val = datetime.strptime(time_string, '%Y-%m-%dT%H:%M:%S.%fZ')
    except (IndexError, ValueError) as e:
        raise ValueError("Filename does not contain a valid datetime string") from e
    
    # Read the miniseed file
    stream = read(file)
    
    # Decimate the data according to the option parameter
    if option == 2:
        data = (stream.detrend().split().decimate(16)
                .merge(method=1, fill_value="interpolate").decimate(10))
    elif option == 1:
        data = stream.detrend().split().merge(method=1, fill_value="interpolate")
    else:
        raise ValueError("Invalid option. Use 1 for full sample rate or 2 for custom decimation.")
    
    # Get the sampling rate
    fs = data[0].stats["sampling_rate"]

    # Set spectrogram parameters
    segment_length_seconds = 0.5
    nperseg = int(segment_length_seconds * fs)
    noverlap_percent = 15
    noverlap = int(nperseg * noverlap_percent / 100)

    # Generate spectrogram
    f, t, Sxx = signal.spectrogram(data[0].data, fs, nperseg=nperseg, noverlap=noverlap)
    array_out = xr.DataArray(
        np.log10(Sxx),
        dims=('frequency', 'seconds'),
        coords={'frequency': f, 'seconds': t}
    )
    array_out.coords['time'] = time_val

    return array_out


def parallel_process(
    data: List[str],
    func: Callable[[str], xr.DataArray] = make_xarray,
    sub_segment_length: int = 6,
    n_processes: Optional[int] = None,
    do_print: bool = False,
) -> xr.DataArray:
    """
    Read in a list of miniseed files and build a 3D xarray object containing 
    a spectrogram for each 5-minute file. Set up Python to distribute analysis
    tasks across all available CPUs.
    
    Args:
        data: A list of paths to individual miniseed files.
        func: A function to use in parallel processing. It should read in a 
        miniseed file path and return an xarray 
            object. Default is make_xarray.
        sub_segment_length: Number of segments to use in each of the parallel 
            processing segments. These are handled in segments to reduce the 
            number of arrays being concatenated (and held in memory) at any 
            one point in time. Default is 6.
        n_processes: Number of processes to use for parallel processing. 
            Default is the number of CPUs.
        do_print: If True, print the progress of the processing segments. 
            Default is False.
    
    Returns:
        A concatenated xarray object containing the spectrograms.
    """
    if n_processes is None:
        n_processes = multiprocessing.cpu_count()

    with multiprocessing.Pool(processes=n_processes) as pool:
        # Initialize a list to hold the xarray sub-arrays
        results_all = []
        
        # Break into the requested number of 5-minute segments at a time (30-mins total)
        segments = [data[i:i + sub_segment_length] 
                    for i in range(0, len(data), sub_segment_length)]
        
        for idx, segment in enumerate(segments):
            if do_print:
                print(f"Segment {idx + 1} of {len(segments)}")
            
            results = pool.map(func, segment)
            
            # Use dask for efficiency
            dask_datasets = [da.chunk({'seconds': 50, 'frequency': 50}) for da in results]
            results_all.append(xr.concat(dask_datasets, dim='time'))
        
        results_full = xr.concat(results_all, dim='time').sortby('time')

    return results_full


def mseed_to_xarray(
    instrument: str, 
    year: int, 
    month: int, 
    day: int, 
    do_print: bool = False,
):
    """
    Given an instrument name and a day, return an xarray of spectrograms.
    
    Args:
        instrument: Instrument name/ID
        year: 4-digit year
        month: month number
        day: day of month 
        do_print: If True, print progress. Default is False.
        
    Returns:
        xarray object
    """
    date = f"{year}/{month:02}/{day:02}"  
    
    # Build path to the requested date folder
    instrument_day_path = "../../ooi/san_data/" + instrument + "/" + date
    
    # Look for all the mseed files in the folder provided above
    mseed_files = find_mseed_files(instrument_day_path)
    
    # Assembe all spectrograms into a single xarray object for one day
    return parallel_process(mseed_files, do_print=do_print)


def mseed_to_netcdf(instrument, year, month, day, file_name):
    """
    Given an instrument name and a day, generate spectrograms for each 
    5-minute segment and export to the requested file location
    
    Args:
        instrument (str): Instrument name/ID
        year (int): 4-digit year
        month (int): month number
        day (int): day of month 
        file_name (str): Path to the output file location. Output file should 
            have a .nc extension
    
    Returns: None
    
    """
    # Generate an xarray of spectrograms
    results = mseed_to_xarray(instrument, year, month, day)
    
    # Save to netcdf format
    results.to_netcdf("data/" + instrument + "-YMD." + date.replace("/",".") + ".nc")


if __name__ == "__main__":
    
    # If you run this python file as a script, this section will run and call
    # the functions defined above.
    
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
