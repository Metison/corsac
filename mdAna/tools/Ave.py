# Written by Metison Wood <wubqmail(at)163.com> under DWYW license

# Standard Python libraries
# {{{
from __future__ import (absolute_import, print_function, division,
        unicode_literals)
import os
import sys
import numpy as np
import pandas as pd
# }}}

class Ave(object):
    """Class for aving the data."""
    def __init__(self, filepath=None, Std=False, Var=False,
            Sum=False, Abs=False, Max=False, Min=False):
    # {{{
        """Initial a ave object.

        Parameters
        ----------
        filepath: str
            file needs to konw its average value.
        Std: bool, optional
            True: return standard deviation, False: not(default).
        Var: bool, optional
            True: return variance, False: not(default).
        Sum: bool, optional
            True: return sum, False: not(default).
        Abs: bool, optional
            True: return absolute value, False: not(default).
        Max: bool, optional
            True: return max value, False: not(default).
        Min: bool, optional
            True: return min value, False: not(default).
        """
        if isinstance(Std, bool) and isinstance(Var, bool) and\
                isinstance(Sum , bool) and isinstance(Abs, bool)\
                and isinstance(Max, bool) and isinstance(Min, bool):
            self.std = Std
            self.var = Var
            self.sum = Sum
            self.abs = Abs
            self.max = Max
            self.min = Min
        else:
            raise ValueError('bool is needed for std...')
            sys.exit(1)
        self.filepath = filepath
    # }}}
    def single(self, data=None, axis=0, header=0, max_rows=None, cols=1):
    # {{{
        """Averaging the single data from onefile.

        Parameters
        ----------
        data: str, file
            the str of data or file
        axis: int, optional
            the axis for averaging.
        header: int, optional
            lines to skip at the beginning of the file.
        max_rows: int, optional
            the maximum rows.
        cols: int or sequence
            columns to read.

        Returns
        -------
        numpy.ndarray: datas, list: columns,
        """
        # Check args
        if isinstance(cols, int):
            cols = (cols,)
        cols = [i-1 for i in cols]
        # Load data
        tmp_np = np.genfromtxt(data, skip_header=header,
                max_rows=max_rows, usecols=cols)
        datas = []
        columns = []
        tmp_mean = np.mean(tmp_np, axis=axis)
        columns.append('Index')
        datas.append([i+1 for i in range(len(tmp_mean))])
        columns.append('Mean')
        datas.append(tmp_mean)
        if self.std:
            columns.append('Std')
            datas.append(np.std(tmp_np, axis=axis))
        if self.var:
            columns.append('Var')
            datas.append(np.var(tmp_np, axis=axis))
        if self.sum:
            columns.append('Sum')
            datas.append(np.sum(tmp_np, axis=axis))
        if self.abs:
            columns.append('Abs')
            datas.append(np.absolute(np.mean(tmp_np, axis=axis)))
        if self.max:
            columns.append('Max')
            datas.append(np.amax(tmp_np, axis=axis))
        if self.min:
            columns.append('Min')
            datas.append(np.amin(tmp_np, axis=axis))
        datas = np.array(datas).T
        return datas, columns
    # }}}
    def single_data(self, axis=0, header=0, max_rows=None, cols=1,
            outfile=True, outpath=''):
    # {{{
        """Averaging the single data from onefile.

        Parameters
        ----------
        axis: int, optional
            the axis for averaging.
        header: int, optional
            lines to skip at the beginning of the file.
        max_rows: int, optional
            the maximum rows.
        cols: int or sequence
            columns to read.
        outfile: bool, optional
            True indicates output the results into file.
        outpath: str, optional
            the path of output.

        Returns
        -------
        pd.DataFrame
        """
        # Check args
        if not os.path.isfile(self.filepath):
            raise ValueError('file not exist')
            sys.exit(1)
        # Handling data
        with open(self.filepath, 'r') as f:
            datas, columns = self.single(data=f, axis=axis,
                    header=header, max_rows=max_rows, cols=cols)
            results = pd.DataFrame(datas, columns=columns)
            results['Index'] = results['Index'].astype(int)
        if outfile:
            results.to_csv(os.path.join(outpath, 'ave.dat'),
                    sep=' ', index=False, float_format='%.6f')
        return results
    # }}}
    def multi_data(self, aheader=0, dlines=None, header=0,
            max_rows=None, cols=1, outfile=True, outpath=''):
    # {{{
        """Averaging multiple datas to one data, not on axis.

        Parameters
        ----------
        aheader: int, optional
            lines to skip at the beginning of the file.
        dlines: int
            the number of lines for each data.
        header: int, optional
            lines to skip at the beginning of the each data.
        max_rows: int, optional
            the maximum rows.
        cols: int or sequence
            columns to read.
        outfile: bool, optional
            True for if output file is needed.
        outpath: str, optional
            the path of output

        Returns
        -------
        pandas.DataFrame
        files
        """
        # Check args
        if isinstance(cols, int):
            cols = (cols,)
        cols = [i-1 for i in cols]
        if max_rows is None:
            max_rows = dlines - header
        if not os.path.isfile(self.filepath):
            raise ValueError('file not exist')
            sys.exit(1)
        with open(self.filepath, 'r') as f:
            nlines = 0
            for line in f:
                nlines += 1
            num_file = int(nlines/dlines)
            s = []
            for i in range(num_file):
                s.append(np.genfromtxt(self.filepath, comments='#',
                skip_header=aheader+header+i*dlines,
                max_rows=max_rows, usecols=cols))
            columns = []
            datas = []
            for i in range(len(cols)):
                tmp_data = s[0][:,i]
                for j in s[1:]:
                    tmp_data = np.vstack((tmp_data, j[:,i]))
                datas.append(np.mean(tmp_data, axis=0))
                columns.append('col'+str(cols[i]+1))
                if self.std:
                    datas.append(np.std(tmp_data, axis=0))
                    columns.append('col'+str(cols[i]+1)+'-std')
                if self.var:
                    datas.append(np.var(tmp_data, axis=0))
                    columns.append('col'+str(cols[i]+1)+'-var')
                if self.sum:
                    datas.append(np.sum(tmp_data, axis=0))
                    columns.append('col'+str(cols[i]+1)+'-sum')
                if self.abs:
                    datas.append(np.absolute(np.mean(tmp_data, axis=0)))
                    columns.append('col'+str(cols[i]+1)+'-abs')
                if self.max:
                    datas.append(np.amax(tmp_data, axis=0))
                    columns.append('col'+str(cols[i]+1)+'-max')
                if self.min:
                    datas.append(np.amin(tmp_data, axis=0))
                    columns.append('col'+str(cols[i]+1)+'-min')
            results = pd.DataFrame(np.array(datas).T, columns=columns)
        if outfile:
            results.to_csv(os.path.join(outpath, 'ave.dat'),
                    sep=' ', index=False, float_format='%.6f')
        return results
    # }}}
    def multi_file(self, filepaths='', header=None, max_rows=None,
            cols=1, outfile=True, outpath=''):
    # {{{
        """Averaging multiple file to one file, not on axis.

        Parameters
        ----------
        filepaths: lits
            list contain filepaths
        header: int, optional
            lines to skip at the beginning of the each data.
        max_rows: int, optional
            the maximum rows.
        cols: int or sequence
            columns to read.
        outfile: bool, optional
            True for if output file is needed.
        outpath: str, optional
            the path of output

        Returns
        -------
        pandas.DataFrame
        files
        """
        # Check args
        if isinstance(cols, int):
            cols = (cols,)
        cols = [i-1 for i in cols]
        for f in filepaths:
            if not os.path.isfile(f):
                raise ValueError(f+' not exist')
                sys.exit(1)
        s = []
        for f in filepaths:
            s.append(np.genfromtxt(f, comments='#',
                skip_header=header, max_rows=max_rows, usecols=cols))
        columns = []
        datas = []
        for i in range(len(cols)):
            tmp_data = s[0][:,i]
            for j in s[1:]:
                tmp_data = np.vstack((tmp_data, j[:,i]))
            datas.append(np.mean(tmp_data, axis=0))
            columns.append('col'+str(cols[i]))
            if self.std:
                datas.append(np.std(tmp_data, axis=0))
                columns.append('col'+str(cols[i])+'-std')
            if self.var:
                datas.append(np.var(tmp_data, axis=0))
                columns.append('col'+str(cols[i])+'-var')
            if self.sum:
                datas.append(np.sum(tmp_data, axis=0))
                columns.append('col'+str(cols[i])+'-sum')
            if self.abs:
                datas.append(np.absolute(np.mean(tmp_data, axis=0)))
                columns.append('col'+str(cols[i])+'-abs')
            if self.max:
                datas.append(np.amax(tmp_data, axis=0))
                columns.append('col'+str(cols[i])+'-max')
            if self.min:
                datas.append(np.amin(tmp_data, axis=0))
                columns.append('col'+str(cols[i])+'-min')
        results = pd.DataFrame(np.array(datas).T, columns=columns)
        if outfile:
            results.to_csv(os.path.join(outpath, 'ave.dat'),
                    sep=' ', index=False, float_format='%.6f')
        return results
    # }}}

if __name__ == '__main__':
    ave = Ave(filepath='lf.nd')#, Std=True, Abs=True, Min=True)
    #results = ave.single_data(axis=0,
    #        header=9, max_rows=None, cols=[3, 4])
    results = ave.multi_data(aheader=3, dlines=3198,
            header=1, max_rows=None, cols=[2, 4])
    #with open('ave.dat', 'r') as f:
    #    datas, columns= ave.single(data=f, axis=0, header=1, max_rows=99, cols=[1, 4, 7])
    #print(pd.DataFrame(datas, columns=columns))
    #results = ave.multi_file(filepaths=['atoms.data', 'atoms1.data',
    #    'atoms2.data'], header=9, max_rows=999, cols=[2,3,4])
    print(results)
