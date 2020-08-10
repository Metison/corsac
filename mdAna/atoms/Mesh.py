"""Mesh class for analyzing the atomic properties of each block area in the
 atomman.System"""

# Standard Python libraries
# {{{
from __future__ import (absolute_import, print_function, division,
        unicode_literals)
import os
import sys
import numpy as np
import atomman as am
import pandas as pd
from ..compatibility import iteritems, range, inttype, stringtype
# }}}

class Mesh(object):
    """Class for handling mesh."""
    def __init__(self, systems=am.System(), bins=10,
            ranges=None, style='count', prop=None, *args):
        # {{{
        """Initialize variables.
        
        Parameters
        ----------
        systems: atomman.System, optional
            The underlying system object of atomman.
        bins: int or sequence, optional
            1D: the bins specification:
                int: num of bins.
                sequence: monotonically increasing bin edges.
            2D: the bins spec:
                int: num of bins for all dims.
                [int, int]: num of bins for each dim.
                array: bin edges for all dims.
                [array, array]: bin edges for each dim.
            3D: the bins spec:
                int: num of bins for all dims.
                [int, int, int]: num of bins for each dim.
                array: bin edges for all dims.
                [array, array, array]: bin edges for each dim.
        ranges: sequence, optional
            1D: [xlo, xhi].
            2D: [xlo, xhi, ylo, yhi].
            3D: [xlo, xhi, ylo, yhi, zlo, zhi].
        style: str, optional
            statistical type: count, mean, sum...
        prop: str, optional
            the properties of atoms
        """

        # Check data types
        if not isinstance(systems, am.System):
            raise ValueError('Invalid systems type')
        if not isinstance(style, str):
            raise ValueError('Invalid style type')
        if not isinstance(prop, str) and prop is not None:
            raise ValueError('Invalid prop type')
        # Set properties
        self.systems = systems
        self.bins = bins
        self.ranges = ranges
        self.style = style
        self.prop = prop
        # }}}
    def m1d(self, direction='z'):
# {{{
        """Statistics in 1D.
        
        Parameters
        ----------
        direction: str.
            the direction of bins.

        Returns
        -------
        numpy.array
        """
        # Check parms
        if direction == "x":
            colnum = 0
        elif direction == "y":
            colnum = 1
        elif direction == "z":
            colnum = 2
        else:
            raise ValueError("direction not exist")
        hists, edges = np.histogram(self.systems.atoms.pos[:,colnum],
                bins=self.bins, range=self.ranges)
        results = np.zeros((len(hists), 2))
        mbins = np.array([edges[:-1], edges[1:]]).T
        results[:,0] = np.mean(mbins, axis=1)
        if self.style == 'count':
            results[:,1] = hists
        if self.style != 'count':
            tmp_mean = []
            tmp_sum = []
            tmp_std = []
            for i in mbins:
                bools = np.logical_and(self.systems.atoms.pos[:,colnum] > i[0],
                        self.systems.atoms.pos[:,colnum] <= i[1])
                tmp_mean.append(np.mean(self.systems.atoms.prop(self.prop)[bools]))
                tmp_sum.append(np.sum(self.systems.atoms.prop(self.prop)[bools]))
                tmp_std.append(np.std(self.systems.atoms.prop(self.prop)[bools]))
            if self.style == 'mean':
                results[:,1] = np.array(tmp_mean)
            elif self.style == 'sum':
                results[:,1] = np.array(tmp_sum)
            elif self.style == 'std':
                results[:,1] = np.array(tmp_std)
        return results
# }}}
    def m2d(self, direction='z'):
# {{{
        """Statistics in 2D.
        
        Parameters
        ----------
        direction: str.
            the vertical direction of bins.

        Returns
        -------
        numpy.ndarray
        """
        # Check parms
        if direction == "x":
            colnum = (1,2)
        elif direction == "y":
            colnum = (0,2)
        elif direction == "z":
            colnum = (0,1)
        else:
            raise ValueError("plane not exist")
        hists, xedges, yedges = np.histogram2d(self.systems.atoms.pos[:, colnum[0]],
                self.systems.atoms.pos[:, colnum[1]],
                bins=self.bins, range=self.ranges)
        xbins = np.array([xedges[:-1], xedges[1:]]).T
        ybins = np.array([yedges[:-1], yedges[1:]]).T
        results = np.zeros((len(xbins)*len(ybins), 3))
        results[:,[0, 1]] = np.array([[i, j] for i in np.mean(xbins, axis=1)
            for j in np.mean(ybins, axis=1)])
        if self.style == 'count':
            results[:,2] = np.ravel(hists)
        if self.style != 'count':
            tmp_mean = []
            tmp_sum = []
            tmp_std = []
            for i in xbins:
                for j in ybins:
                    xbools = np.logical_and(
                            self.systems.atoms.pos[:,colnum[0]] > i[0],
                            self.systems.atoms.pos[:,colnum[0]] <= i[1])
                    ybools = np.logical_and(
                            self.systems.atoms.pos[:,colnum[1]] > j[0],
                            self.systems.atoms.pos[:,colnum[1]] <= j[1])
                    bools = np.logical_and(xbools, ybools)
                    tmp_mean.append(np.mean(self.systems.atoms.prop(self.prop)[bools]))
                    tmp_sum.append(np.sum(self.systems.atoms.prop(self.prop)[bools]))
                    tmp_std.append(np.std(self.systems.atoms.prop(self.prop)[bools]))
            if self.style == 'mean':
                results[:,2] = np.array(tmp_mean)
            elif self.style == 'sum':
                results[:,2] = np.array(tmp_sum)
            elif self.style == 'std':
                results[:,2] = np.array(tmp_std)
        return results
# }}}
    def m3d(self):
# {{{
        """Statistics in 3D.
        
        Parameters
        ----------

        Returns
        -------
        numpy.ndarray
        """
        # Check parms
        hists, edges = np.histogramdd(self.systems.atoms.pos,
                bins=self.bins, range=self.ranges)
        xbins = np.array([edges[0][:-1], edges[0][1:]]).T
        ybins = np.array([edges[1][:-1], edges[1][1:]]).T
        zbins = np.array([edges[2][:-1], edges[2][1:]]).T
        results = np.zeros((len(xbins)*len(ybins)*len(zbins), 4))
        results[:,[0,1,2]] = np.array([[i,j,k] for i in np.mean(xbins, axis=1)
                for j in np.mean(ybins, axis=1)
                for k in np.mean(zbins, axis=1)])
        if self.style == 'count':
            results[:,3] = np.ravel(hists)
        if self.style != 'count':
            tmp_mean = []
            tmp_sum = []
            tmp_std = []
            for i in xbins:
                for j in ybins:
                    for k in zbins:
                        xbools = np.logical_and(
                                self.systems.atoms.pos[:,0] > i[0],
                                self.systems.atoms.pos[:,0] <= i[1])
                        ybools = np.logical_and(
                                self.systems.atoms.pos[:,1] > j[0],
                                self.systems.atoms.pos[:,1] <= j[1])
                        zbools = np.logical_and(
                                self.systems.atoms.pos[:,2] > k[0],
                                self.systems.atoms.pos[:,2] <= k[1])
                        bools = np.logical_and(xbools, ybools)
                        tmp_mean.append(np.mean(self.systems.atoms.prop(self.prop)[bools]))
                        tmp_sum.append(np.sum(self.systems.atoms.prop(self.prop)[bools]))
                        tmp_std.append(np.std(self.systems.atoms.prop(self.prop)[bools]))
            if self.style == 'mean':
                results[:,3] = np.array(tmp_mean)
            elif self.style == 'sum':
                results[:,3] = np.array(tmp_sum)
            elif self.style == 'std':
                results[:,3] = np.array(tmp_std)
        return results
# }}}

if __name__ == "__main__":
    atoms = am.load('atom_dump', 'atoms.data')
    Mesh = Mesh(systems=atoms, bins=5, style='count', prop='lop')
    results = Mesh.m3d()
    print(len(results))
