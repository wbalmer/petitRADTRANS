"""Command line interface to download petitRADTRANS input data files.
"""

import datetime
import functools
import http.client
import os
import sys
import time
import urllib.parse
import urllib.request
import warnings

from petitRADTRANS.config import petitradtrans_config_parser


megabyte = 9.5367431640625e-07  # 1 / 1024 ** 2 (B to MB)


def _reporthook(count: int, block_size: int, total_size: int, time_start=0.0):
    """Display downloading progress on the terminal.

    Args:
        count: number of the data block being downloaded
        block_size: (B) size of a data block
        total_size: (B) total size of the data
        time_start: (s) time at which the download started
    """
    # Data download progress
    progress_size = count * block_size
    percent = min(int(progress_size * 100 / total_size), 100)

    # Convert from B to MB
    progress_size *= megabyte
    total_size *= megabyte

    # Speed and ETA
    duration = max(time.time() - time_start, 1e-6)
    speed = max(progress_size / duration, 1e-6)

    eta = datetime.timedelta(
        seconds=int(total_size / speed - duration)
    )

    _reporthook_sys_output(
        percent=percent,
        progress_size=progress_size,
        total_size=total_size,
        speed=speed,
        eta=eta
    )


def _reporthook_sys_output(percent: int, progress_size: float, total_size: float, speed: float,
                           eta: [datetime.timedelta, str]):
    """Terminal output of download progress

    Args:
        percent: progression percentage
        progress_size: (MB) amount of data downloaded
        total_size: (MB) amount of data to download
        speed: (MB.s-1) data downloading rate
        eta: (hh:mm:ss) time before download completion
    """

    columns_str = (
        f"\r {percent}% | "
        f"{progress_size:.1f}/{total_size:.1f} MB | "
        f"{speed:.1f} MB/s | "
        f"eta {eta}"
    )

    sys.stdout.write(columns_str)
    sys.stdout.flush()


def download_input_data(destination, source=None, rewrite=False,
                        path_input_data=petitradtrans_config_parser['Paths']['prt_input_data_path'],
                        url_input_data=petitradtrans_config_parser['URLs']['prt_input_data_url'],
                        byte_amount=8192) -> [http.client.HTTPResponse, None]:
    """Download a petitRADTRANS input data file.
    If source is None, the source URL is automatically deduced from the destination file.

    Args:
        destination: file to be downloaded (path to the file where the downloaded data are saved)
        source: URL of the source
        rewrite: if True, the data are downloaded even if destination is an already existing local file
        path_input_data: local path to petitRADTRANS' input_data folder
        url_input_data: URL petitRADTRANS' input data
        byte_amount: amount of bytes to be read from the URL response

    Returns:
        If a new file has been downloaded, return a HTTPResponse object, and None otherwise.
    """
    # Ensure destination is the absolute path
    destination = os.path.abspath(destination)

    # Checks before download
    if os.path.isfile(destination) and not rewrite:
        print(f"file '{destination}' already exists, skipping download (set rewrite=True to force re-download)...")
        return

    if path_input_data != petitradtrans_config_parser['Paths']['prt_input_data_path']:
        warnings.warn(f"path_input_data ('{destination}') "
                      f"is not the configured one ('{petitradtrans_config_parser['Paths']['prt_input_data_path']}'); "
                      f"the downloaded file will not be loaded by petitRADTRANS\n"
                      f"Change the destination, move the file manually after download, or re-set the input data path")

    if destination[:len(path_input_data)] != path_input_data:
        warnings.warn(f"destination path ('{destination}') "
                      f"is not within the input data folder ('{path_input_data}')")

    # Automatically get the source URL
    if source is None:
        if path_input_data in destination:
            destination = destination.rsplit(path_input_data, 1)[-1]

        url_path = destination.replace(os.path.sep, '/')

        if url_path[0] != '/':
            url_path = '/' + url_path

        source = urllib.parse.quote(url_input_data + url_path, safe='') + '&dl=1'

    # Start download
    destination = os.path.join(path_input_data, destination)

    print(f"Downloading '{destination}'...")

    with urllib.request.urlopen(source) as response:
        with open(destination, "wb") as f:
            headers = response.info()

            # Get the total size of the downloaded data
            if "content-length" in headers:
                total_size = int(headers["content-length"])
            else:
                total_size = -1

            # Initialize the response reading, using a fixed block size for reporthook
            response_read = functools.partial(response.read, byte_amount)

            # Initialize reporthook, memorizing the download starting time for speed and ETA
            time_start = time.time()
            reporthook = functools.partial(_reporthook, time_start=time_start)

            # Read the response, block by block, and save it to the destination file
            for i, data in enumerate(iter(response_read, b"")):
                f.write(data)
                reporthook(i + 1, byte_amount, total_size)

    # Ensure a clean terminal output after the download
    if (i + 1) * byte_amount >= total_size:
        total_size *= megabyte
        _reporthook_sys_output(
            percent=100,
            progress_size=total_size,
            total_size=total_size,
            speed=total_size / (time.time() - time_start),
            eta="0:00:00"
        )
    else:
        warnings.warn(f"not all data has been downloaded")

    sys.stdout.write("\n")

    return response
