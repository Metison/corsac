# Written by Metison Wood <wubqmail(at)163.com> under DWYW license

# Python Standard libraries
# {{{
from __future__ import (absolute_import, print_function, division,
        unicode_literals)
import os
import sys
import numpy as np
import pandas as pd
# }}}

def hist(filepath=None, col=1, number=True, step=0.01, outfile=True, outpath=''):
    # {{{
    """Count the column in file.

    Parameters
    ----------
    filepath: str
        the path of data file.
    col: int or sequence, optional
        columns to read.
    number: bool, optional
        True for indicating the data is number, False for str.
    step: float, optional
        only valid with number is switching on.
    outfile: bool, optional
        True for output file.
    outpath: str, optional
        the path of output, default is current workdir.

    Returns
    -------
    pd.Series
    file: optional
    """
    # Check args
    if not os.path.isfile(filepath):
        raise ValueError('file not exist')
        sys.exit(1)
    if not isinstance(col, int):
        raise ValueError("col must be int.")
    filename = input('Please input the output file name[hist.dat]: ')
    filename = 'hist.dat' if not filename else filename
    col = col-1
    # Handling the file
    tmp_data = pd.read_csv(filepath, sep=' ', header=None,
            index_col=False)
    if number:
        tmp_col = tmp_data[col].to_numpy()
        bins = int((np.max(tmp_col) - np.min(tmp_col))/step)
        ranges = [np.min(tmp_col), np.min(tmp_col)+bins*step]
        hist, edges = np.histogram(tmp_col, bins=bins, range=ranges)
        results = pd.Series(hist, index=(edges[1:]+edges[:-1])/2)
    if not number:
        results = tmp_data[col].value_counts()
    # Output the results
    if outfile:
        results.to_csv(os.path.join(outpath, filename),
                float_format="%.6f", header=["Histogram"],
                index_label="Index", sep=' ')
    return results
    # }}}
def hist2d(x=None, y=None, bins=100):
    # {{{
    """Compute the 2d histogram of two datas.
    
    Parameters
    ----------
    x: np.ndarray
        one dimension datas.
    y: np.ndarray
        one dimension datas.
    bins: int, optional
        the bins of histogram.

    Returns
    -------
    X, Y, Z: np.ndarray
        one dimension datas.
    """
    H, xedges, yedges = np.histogram2d(x, y, bins=bins)
    x1 = np.mean(np.vstack((xedges[:-1], xedges[1:])), axis=0)
    y1 = np.mean(np.vstack((yedges[:-1], yedges[1:])), axis=0)
    X, Y, Z = [], [], []
    for i in range(len(x1)):
        for j in range(len(y1)):
            X.append(x1[i])
            Y.append(y1[i])
            Z.append(H[i, j])
    X, Y, Z = np.array(X), np.array(Y), np.array(Z)
    return X, Y, Z
    # }}}

if __name__ == "__main__":
    results = Hist(filepath='ave.dat', number=False, col=3, step=0.1)
