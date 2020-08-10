#!/usr/bin/env python
# Written by Metison Wood <wubqmail(at)163.com> under DWYW license
# To analyze the Bond Orientational Order at 3D

# Python standard libraries
# {{{
import os, sys
import numpy as np
import atomman as am
import subprocess as sp
# }}}

def BAA(systems=None, symbols=None, face=0.5, minneib=0,
        edge=0.01):
    # {{{
        """Analyzing the csro using dumpana.
        
        Parameters
        ----------
        systems: atomman.System
        symbols: str or a list of str
            elements in systems.
        edge: float, optional.
            edge limit.
        face: float, optional.
            face limit.
        minneib: int, optional.
            the minmum neighbor.
        edge: float, optional.
            edge limit.

        Returns
        -------
        results: numpy.ndarray
            edges, hist
        """
        # Check args
        symbols = symbols if isinstance(symbols, str) else " ".join(symbols)
        systems.dump('atom_dump', f='.baa.data',
                prop_name=['atom_id', 'atype', 'pos'])
        with open('.dumpana', 'w') as f:
            f.write(symbols+'\n15\n1\n2\n'+str(face)+'\n'
                    +str(minneib)+'\n'+str(edge)+'\n\n\n\n\n')
        cmdline = 'dumpana -1 .baa.data < .dumpana'
        sp.run(cmdline, shell=True, stdout=sp.DEVNULL)
        os.remove('.dumpana')
        os.remove('.baa.data')
        tmp_data = np.genfromtxt('bondang.dat', usecols=5)
        bins = np.linspace(0, 180, num=181)
        hist, edges = np.histogram(tmp_data, bins, density=True)
        os.remove('bondang.dat')
        ledges = edges[:-1]
        redges = edges[1:]
        edges = (ledges + redges)/2
        results = np.vstack((edges, hist)).T
        np.histogram
        return results
    # }}}

if __name__ == "__main__":
    import mdAna as ma
    frames = ma.Frame('atom.data')
    timesteps, systems = frames.read_single()
    results = BAA(systems=systems, symbols=['Cu', 'Ag'])
    print(results)
