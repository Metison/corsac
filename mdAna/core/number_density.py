# Written by Metison Wood <wubqmail(at)163.com> under DWYW license

# Standard Python libraries
# {{{
from __future__ import (absolute_import, print_function, division,
        unicode_literals)
import sys
import os
import numpy as np
import atomman as am
from ..compatibility import iteritems, range, inttype, stringtype
# }}}

def nd_1D(systems=am.System(), bins=10, ranges=None,
        direction='z', atype='', outfile=False):
    # {{{
    """Computes 1D number density.
    
    Parameters
    ----------
    systems: atomman.System
        The underlying system object of atomman.
    bins: int or sequence, optional
        1D: the bins specification:
            int: num of bins.
            sequence: monotonically increasing bin edges.
    ranges: sequence, optional
        1D: [xlo, xhi].
    direction: str.
        the direction of bins.
    outfile: bool, optional
        default: False, not output file
    atype: str or int.
        atoms type

    Returns
    -------
    numpy.ndarray
    file: optional
    """
    # Select atoms based on type
    if isinstance(atype, stringtype) and atype != '':
        atype = int(atype)
    if atype:
        bools = systems.atoms.atype == atype
        systems = am.System(box=systems.box, atoms=systems.atoms[bools])
    # Check direction
    if direction == "x":
        colnum = 0
        areas = systems.box.ly * systems.box.lz
    elif direction == "y":
        colnum = 1
        areas = systems.box.lx * systems.box.lz
    elif direction == "z":
        colnum = 2
        areas = systems.box.lx * systems.box.ly
    else:
        raise ValueError("direction not exist")
    # Count & computes number density
    hists, edges = np.histogram(systems.atoms.pos[:,colnum],
            bins=bins, range=ranges)
    results = np.zeros((len(hists), 2))
    mbins = np.array([edges[:-1], edges[1:]]).T
    results[:,0] = np.mean(mbins, axis=1)
    results[:,1] = hists/(mbins[:,1]-mbins[:,0])/areas
    # Out put
    if outfile:
        filepath = input('Input the file name or path[nd1D.dat]:')
        filepath = 'nd1D.dat' if not filepath else filepath
        header = direction+' ND'+str(atype)
        np.savetxt(filepath, results, header=header, fmt='%.6f')
    return results
    # }}}
def nd_2D(systems=am.System(), bins=10, ranges=None,
        direction='z', dranges=None, atype='', outfile=False):
    # {{{
    """Computes 2D number density.
    
    Parameters
    ----------
    systems: atomman.System
        The underlying system object of atomman.
    bins: int or sequence, optional
        int: num of bins for all dims.
        [int, int]: num of bins for each dim.
        array: bin edges for all dims.
        [array, array]: bin edges for each dim.
    ranges: sequence, optional
        2D: [xlo, xhi, ylo, yhi].
    direction: str.
        the vertical direction of bins.
    drange: sequence, optional
        [dlo, dhi]
    outfile: bool, optional
        default: False, not output file
    atype: str or int.
        atoms type

    Returns
    -------
    numpy.ndarray
    file: optional
    """
    # Select atoms based on type
    if isinstance(atype, str) and atype != '':
        atype = int(atype)
    if atype:
        bools = systems.atoms.atype == atype
        systems = am.System(box=systems.box, atoms=systems.atoms[bools])
    # Check direction
    if direction == "x":
        dcol, xcol, ycol = 0, 1, 2
    elif direction == "y":
        dcol, xcol, ycol = 1, 0, 2
    elif direction == "z":
        dcol, xcol, ycol = 2, 0, 1
    else:
        raise ValueError("direction not exist")
    # Select atoms
    if dranges is None:
        dranges = [np.min(systems.atoms.pos[:,dcol]),
                np.max(systems.atoms.pos[:,dcol])]
    bools = np.logical_and(
        systems.atoms.pos[:,dcol] > dranges[0],
        systems.atoms.pos[:,dcol] <= dranges[1])
    X = systems.atoms.pos[:,xcol][bools]
    Y = systems.atoms.pos[:,ycol][bools]
    # Count & computes number density
    hists, xedges, yedges = np.histogram2d(X, Y, bins=bins, range=ranges)
    xbins = np.array([xedges[:-1], xedges[1:]]).T
    ybins = np.array([yedges[:-1], yedges[1:]]).T
    areas = [(i[1]-i[0])*(j[1]-j[0]) for i in xbins
            for j in ybins]
    results = np.zeros((len(xbins)*len(ybins), 3))
    results[:,[0, 1]] = np.array([[i, j] for i in np.mean(xbins, axis=1)
        for j in np.mean(ybins, axis=1)])
    results[:,2] = np.ravel(hists)/areas
    # Output
    if outfile:
        filepath = input('Input the file name or path[nd2D.dat]:')
        filepath = 'nd2D.dat' if not filepath else filepath
        header = direction+' ND'+str(atype)
        np.savetxt(filepath, results, header=header, fmt='%.6f')
    return results
    # }}}

# Test
if __name__ == "__main__":
    systems = am.load('atom_dump', 'atoms.data')
    results = nd_2D(systems=systems, bins=20, atype=1, outfile=True)
    print(results)
