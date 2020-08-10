# Written by Metison Wood <wubqmail(at)163.com> under DWYW license

# Standard Python libraries
# {{{
import os
import sys
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d, splrep, splev
from scipy.optimize import curve_fit
# }}}

def gaussian_smooth(filepath=None, index_col=1, data_cols=2,
        header=0, datas=None, M=60, sigma=6.6,
        outfile=True, outpath=''):
    # {{{
    """Filtering the data by using FIR filter.

    Parameters
    ----------
    filepath: str
        The path of data file.
    index_col: int, optional
        The column of index
    data_col: int or sequence, optional
        The columns of data
    header: int, optional
        The skip rows
    datas: numpy.ndarray
        The datas for aving, only valid if no files.
    M: int, optional
        Length of the filter.
    sigma: float, optional
        The standard deviation, sigma.
    outfile: bool, optional
        True indicates output the results into file.
    outpath: str, optional
        The path of output

    Returns
    -------
    numpy.ndarray
    file: optional
    """
    wk = [np.exp(-(i/sigma)**2) for i in range(-M, M+1)]
    wk = np.array(wk)
    wk = wk/np.sum(wk)
    # Get data
    if filepath is not None:
        if isinstance(index_col, int):
            index_col = [index_col-1,]
        if isinstance(data_cols, int):
            data_cols = [data_cols,]
        data_cols = [i-1 for i in data_cols]
        usecols = index_col + data_cols
        num_cols = len(usecols)
        datas = np.genfromtxt(filepath, skip_header=header,
                usecols=usecols)
    else:
        datas = datas
        num_cols = datas.shape[1]
    # smooth the data
    results = datas[:,0]
    for i in range(1,num_cols):
        fdatas = []
        tmp_data = np.concatenate((datas[-M:, i], datas[:, i],
            datas[:M, i]))
        for j in range(len(datas[:, 0])):
            sel_data = tmp_data[j:j+2*M+1]
            fdatas.append(np.sum(wk * sel_data))
        fdatas = np.array(fdatas)
        results = np.vstack((results, fdatas))
        s1, s2, s3 = fdatas[:-2], fdatas[1:-1], fdatas[2:]
        S = np.sum(np.square(s1 + s3 - 2*s2))
        print(S)
    results = results.T
    # output the results
    if outfile:
        np.savetxt(os.path.join(outpath, 'firsm.dat'), results,
                fmt="%.6f", header="Index Data...")
    return results
    # }}}
def moving_averaging(filepath=None, index_col=1, data_cols=2,
        header=0, datas=None, windowsize=5,
        outfile=True, outpath=''):
    # {{{
    """
    as int the original MATLAB implementation.
    yy = smooth(y) # smooths the data in the column vector y
    yy(1) = y(1)
    yy(2) = (y(1) + y(2) + y(3)) / 3
    yy(3) = (y(1) + y(2) + y(3) + y(4) + y(5)) / 5
    yy(4) = (y(2) + y(3) + y(4) + y(5) + y(6)) / 5
    moving averaging the data.

    Parameters
    ----------
    filepath: str
        the path of data file.
    index_col: int, optional
        the column of index
    data_col: int or sequence, optional
        the columns of data
    header: int, optional
        the skip rows
    datas: numpy.ndarray
        the datas for aving, only valid if no files.
    windowsize: int, optional
        must be odd int.
    outfile: bool, optional
        True indicates output the results into file.
    outpath: str, optional
        the path of output

    Returns
    -------
    numpy.ndarray
    file: optional
    """
    # check args
    if not isinstance(windowsize, int) and windowsize%2 != 1:
        raise ValueError('windowsize(odd) improper')
        sys.exit(1)
    # Get data
    if filepath is not None:
        if isinstance(index_col, int):
            index_col = [index_col-1,]
        if isinstance(data_cols, int):
            data_cols = [data_cols,]
        data_cols = [i-1 for i in data_cols]
        usecols = index_col + data_cols
        num_cols = len(usecols)
        datas = np.genfromtxt(filepath, skip_header=header,
                usecols=usecols)
    else:
        datas = datas
        num_cols = datas.shape[1]
    # Moving averaging data
    results = datas[:,0]
    for i in range(1,num_cols):
        tmp_data = datas[:, i]
        middle = np.convolve(tmp_data, np.ones(windowsize,
            dtype=int), 'valid')/windowsize
        r = np.arange(1, windowsize-1, 2)
        start = np.cumsum(tmp_data[:windowsize-1])[::2]/r
        stop = (np.cumsum(tmp_data[:-windowsize:-1])[::2]/r)[::-1]
        results = np.vstack((results, np.concatenate((start, middle, stop))))
    # output the results
    results = results.T
    if outfile:
        np.savetxt(os.path.join(outpath, 'movave.dat'), results,
                fmt="%.6f", header="Index Data...")
    return results
    # }}}
def savitzky_golay(filepath=None, index_col=1, data_cols=2,
        header=0, datas=None, windowsize=5, polyorder=2,
        mode='interp', outfile=True, outpath=''):
    # {{{
    """
    Apply a Sacitzky-Golay filter to an array.

    Parameters
    ----------
    filepath: str
        the path of data file.
    index_col: int, optional
        the column of index
    data_col: int or sequence, optional
        the columns of data
    header: int, optional
        the skip rows
    datas: numpy.ndarray
        the datas for aving, only valid if no files.
    windowsize: int, optional
        must be odd positive int.
    polyorder: int, optional
        less than the windowsize
    mode: str, optional
        Must be 'mirror', 'constant', 'nearst', 'warp' or 'interp'
    outfile: bool, optional
        True indicates output the results into file.
    outpath: str, optional
        the path of output

    Returns
    -------
    numpy.ndarray
    file: optional
    """
    # check args
    if not isinstance(windowsize, int) and windowsize%2 != 1:
        raise ValueError('windowsize(odd) improper')
        sys.exit(1)
    # Get data
    if filepath is not None:
        if isinstance(index_col, int):
            index_col = [index_col-1,]
        if isinstance(data_cols, int):
            data_cols = [data_cols,]
        data_cols = [i-1 for i in data_cols]
        usecols = index_col + data_cols
        num_cols = len(usecols)
        datas = np.genfromtxt(filepath, skip_header=header,
                usecols=usecols)
    else:
        datas = datas
        num_cols = datas.shape[1]
    # Handling data
    results = datas[:,0]
    for i in range(1,num_cols):
        tmp_data = datas[:, i]
        results = np.vstack((results, savgol_filter(tmp_data,
            windowsize, polyorder, mode=mode)))
    # output the results
    results = results.T
    if outfile:
        np.savetxt(os.path.join(outpath, 'sgfilter.dat'),
                results, fmt="%.6f", header="Index Data...")
    return results
    # }}}
def interp(filepath=None, x_col=1, y_col=2, header=0,
        datas=None, kind='cubic', xbins=300,
        outfile=True, outpath=''):
    # {{{
    """
    Interpolate a 1D function, and output based on xbins.

    Parameters
    ----------
    filepath: str
        the path of data file.
    x_col: int, optional
        the column of x
    y_col: int, optional
        the column of y
    header: int, optional
        the skip rows
    datas: numpy.ndarray
        the datas for aving, only valid if no files.
    kind: str, optional
        'zero', 'sliner', 'quadratic', 'cubic', 'previous',
        'next', 'nearest', 'linear'
    xbins: int, optional
        the bins of x in output data
    outfile: bool, optional
        True indicates output the results into file.
    outpath: str, optional
        the path of output

    Returns
    -------
    numpy.ndarray
    file: optional
    """
    if filepath is not None:
        datas = np.genfromtxt(filepath, skip_header=header,
                usecols=[x_col-1, y_col-1])
    else:
        datas = datas
    x, y = datas[:,0], datas[:,1]
    # Handling data
    func = interp1d(x, y, kind=kind)
    xnew = np.linspace(np.min(x), np.max(x), num=xbins+1, endpoint=True)
    ynew = func(xnew)
    results = np.vstack((xnew, ynew))
    # output the results
    results = results.T
    if outfile:
        np.savetxt(os.path.join(outpath, 'interp.dat'), results,
                fmt="%.6f", header="X Y")
    return results
    # }}}
def spline(filepath=None, x_col=1, y_col=2, header=0,
        datas=None, weight=None, smooth=0., xbins=300,
        outfile=True, outpath=''):
    # {{{
    """
    Cubic spline of interpolate.

    Parameters
    ----------
    filepath: str
        the path of data file.
    x_col: int, optional
        the column of x
    y_col: int, optional
        the column of y
    header: int, optional
        the skip rows
    datas: numpy.ndarray
        the datas for aving, only valid if no files.
    weight: int, optional
        the column of weight
    smooth: float, optional
        smooth condition, default is 0.0
    xbins: int, optional
        the bins of x in output data
    outfile: bool, optional
        True indicates output the results into file.
    outpath: str, optional
        the path of output

    Returns
    -------
    numpy.ndarray
    file: optional
    """
    # Check args
    cols = [x_col-1, y_col-1]
    if isinstance(weight, int):
        cols.append(weight)
    if filepath is not None:
        datas = np.genfromtxt(filepath, skip_header=header,
                usecols=cols)
    else:
        datas = datas
    x, y = datas[:,0], datas[:,1]
    if isinstance(weight, int):
        weight = datas[:,2]
    # Handling file
    tck = splrep(x, y, w=weight, s=smooth)
    xnew = np.linspace(np.min(x), np.max(x), num=xbins+1, endpoint=True)
    ynew = splev(xnew, tck, der=0)
    results = np.vstack((xnew, ynew))
    # Output results
    results = results.T
    if outfile:
        np.savetxt(os.path.join(outpath, 'spline.dat'), results,
                fmt="%.6f", header="Index Spline")
    return results
    # }}}
def fit(filepath=None, x_col=1, y_col=2, header=0, datas=None,
        func=None, p0=None, xnew=None, xbins=300,
        outfile=True, outpath=''):
    # {{{
    """
    Curve fit

    Parameters
    ----------
    filepath: str
        the path of data file.
    x_col: int, optional
        the column of x
    y_col: int, optional
        the column of y
    header: int, optional
        the skip rows
    datas: numpy.ndarray
        the datas for aving, only valid if no files.
    func: fuction
        the function defined by del
    xnew: [int, int]
        the range of x
    xbins: int, optional
        the bins of x in output data
    outfile: bool, optional
        True indicates output the results into file.
    outpath: str, optional
        the path of output

    Returns
    -------
    popt, datas: list, numpy.ndarray(xnew, ynew)
    file: optional
    """
    # Check args
    if func is None:
        raise ValueError('function is needed')
        sys.exit(1)
    if filepath is not None:
        cols = [x_col-1, y_col-1]
        datas = np.genfromtxt(filepath, skip_header=header,
                usecols=cols)
    else:
        datas = datas
    x, y = datas[:,0], datas[:,1]
    # Handling file
    popt, pcov = curve_fit(func, x, y, p0=p0)
    if xnew is not None:
        xnew = np.linspace(xnew[0], xnew[1], num=xbins+1, endpoint=True)
    else:
        xnew = np.linspace(np.min(x), np.max(x), num=xbins+1, endpoint=True)
    ynew = func(xnew, *popt)
    datas = np.vstack((xnew, ynew))
    # Output results
    header = " ".join(['p'+str(i)+': '+str(popt[i])+' '
            for i in range(len(popt))])
    datas = datas.T
    if outfile:
        np.savetxt(os.path.join(outpath, 'fit.dat'), datas,
                fmt="%.6f", header=header)
    return popt, datas
    # }}}
def nodical(datas=None, x=None, y=None):
    # {{{
    """
    Find the intersection coordinates

    Parameters
    ----------
    datas: numpy.ndarray
        contain 2 columns
    x or y: float

    Returns
    -------
    (x, y)
    """
    # Check args
    if x is None and y:
        x, y = datas[:, 0]-0, np.abs(datas[:, 1]-y)
        xnew = np.mean(x[np.argsort(y)][:2])
        ynew = np.mean(datas[:, 1][np.argsort(y)][:2])
        return xnew, ynew
    elif x and y is None:
        x, y = np.abs(datas[:, 0]-x), datas[:, 1]-0
        xnew = np.mean(datas[:, 0][np.argsort(x)][:2])
        ynew = np.mean(y[np.argsort(x)][:2])
        return xnew, ynew
    else:
        raise ValueError("only x or y is provided")
        sys.exit(1)
    # }}}
def trape(x=None, y=None):
# {{{
    """Trapezoidal integration.
    
    Parameters
    ----------
    x: numpy.ndarray
        a list of float.
    y: numpy.ndarray
        a list of float.

    Returns
    -------
    results: float
    """
    distance = x[1:] - x[:-1]
    s1 = y[:-1]
    s2 = y[1:]
    results = np.sum((s1 + s2) * distance/2)
    return results
# }}}

if __name__ == "__main__":
    #datas = np.genfromtxt('gr.dat')
    #def func(x, a, b, c, d):
    #    return a*x**3 + b*x**2 +c*x + d
    #results = fit(datas=datas, func=func)
    #results = moving_averaging(filepath='gr.dat', windowsize=15,
    #        index_col=1, data_cols=2)
    #results = savitzky_golay(filepath='ave.dat', windowsize=21,
    #        index_col=2, data_cols=[3, 4], mode='mirror')
    #results = interp(filepath='test.dat', x_col=2, y_col=4,
    #        xbins=100, kind='cubic')
    #results = spline(filepath='test.dat', x_col=2, y_col=4,
    #        weight=3)
    #def func(x, a, b, c, d):
    #    return a*x**3 + b*x**2 +c*x + d
    #prop, datas = fit(filepath='test.dat', x_col=2, y_col=4,
    #        func=func)
    #datas = np.genfromtxt('gr.dat')
    #results = nodical(datas=datas, x=0.5)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    datas = pd.read_csv('./avend.dat').values
    res = gaussian_smooth(datas=datas)

    #windows = signal.windows.gaussian(141, 24)
    x = res[:, 0]
    y = res[:, 1]
    # Plot
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(x, y, ls='-')
    ax.set_xlim(40, 75)
    #ax.plot(windows)
    plt.show()
    #print(results)
