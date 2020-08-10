# Written by Metison Wood <wubqmail(at)163.com> under DWYW license
# Import python libraries
# {{{
from __future__ import (absolute_import, print_function, division,
        unicode_literals)
import sys
import os
import numpy as np
# }}}
def peak_valley(datas=None, ranges=10, norm=True, outpeaks=True,
        outvalleys=True, outfile=True, outpath=''):
    # {{{
    """
    Find the tmp_peaks and tmp_valleys from two columns file.

    Parameters
    ----------
    datas: numpy.array
        the datas to find the tmp_peaks and tmp_valleys
    ranges: int, optional
        Neighbor comparison ranges
    norm: bool, optional
        True indicates the length of tmp_peaks is same as tmp_valleys
    peaks: bool, optional
        True indicates only the tmp_peaks return
    valleys: bool, optional
        True indicates only the tmp_valleys return
    outfile: bool, optional
        True indicates output the results into file
    outpath: str, optional
        the path of output

    Returns
    -------
    tmp_peaks, tmp_valleys: numpy.ndarray
        row 1: index row 2: tmp_peaks or tmp_valleys
    file: optional
        output file

    Raises
    ------
    ValueError : datas is None
    """

    # Check argvs
    if datas is None:
        raise ValueError('data is None')
        sys.exit(1)
    if len(datas.shape) == 1:
        datas = np.vstack((np.array([i for i\
            in range(datas.shape[0])]), datas)).T
    # Define variables
    tmp_peak = []
    tmp_valley = []
    # Find the peaks and valleys
    flag1, flag2 = 0 , 0
    for i in range(ranges, len(datas[:,1])-ranges):
        s1, s2 = 0, 0
        for n in range(1,ranges+1):
            if datas[i,1] >= datas[i-n,1] and datas[i,1] >= datas[i+n,1]:
                s1 += 1
            if datas[i,1] <= datas[i-n,1] and datas[i,1] <= datas[i+n,1]:
                s2 += 1
        if s1 == ranges:
            if flag1 == 0:
                tmp_peak.append(list(datas[i,:]))
            # Find the same peaks
            elif flag1 == 1:
                tmp_peak[-1] = [(tmp_peak[-1][0]+datas[i,0])/2, tmp_peak[-1][1]]
            flag1, flag2 = 1, 0
        elif s2 == ranges:
            if flag2 == 0:
                tmp_valley.append(list(datas[i,:]))
            # Find the same valleys
            elif flag2 == 1:
                tmp_valley[-1] = [(tmp_valley[-1][0]+datas[i,0])/2, tmp_valley[-1][1]]
            flag1, flag2 = 0, 1
    # normlize datas
    if norm is False:
        peaks, valleys = np.array(tmp_peak).T, np.array(tmp_valley).T
    elif norm is True:
        if tmp_peak[0][0] > tmp_valley[0][0]:
            del(tmp_valley[0])
        if len(tmp_peak) == len(tmp_valley):
            peaks, valleys = np.array(tmp_peak).T, np.array(tmp_valley).T
        elif len(tmp_peak) > len(tmp_valley):
            peaks, valleys = np.array(tmp_peak[:-1]).T, np.array(tmp_valley).T
        elif len(tmp_peak) < len(tmp_valley):
            peaks, valleys = np.array(tmp_peak).T, np.array(tmp_valley[:-1]).T
    # output results
    if outpeaks and not outvalleys:
        if outfile:
            np.savetxt(os.path.join(outpath, 'peaks.dat'),
                    peaks.T, fmt="%.6f", header="Index Peaks")
        return peaks.T
    elif not outpeaks and outvalleys:
        if outfile:
            np.savetxt(os.path.join(outpath, 'valleys.dat'),
                    valleys.T, fmt="%.6f", header="Index Valleys")
        return valleys.T
    else:
        if outfile:
            np.savetxt(os.path.join(outpath, 'peaks.dat'),
                    peaks.T, fmt="%.6f", header="Index Peaks")
            np.savetxt(os.path.join(outpath, 'valleys.dat'),
                    valleys.T, fmt="%.6f", header="Index Valleys")
        return peaks.T, valleys.T
    # }}}

if __name__ == "__main__":
    datas = np.genfromtxt('movave.dat', skip_header=1, usecols=[0,1])
    print(datas)
    peak_valley(datas=datas, outfile=False, ranges=15, outvalleys=True)
