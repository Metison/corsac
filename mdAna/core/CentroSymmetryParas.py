#!/usr/bin/env python
# Written by Metison Wood <wubqmail(at)163.com> under DWYW license
# To analyze the Centro-Symmetry Parameters at 3D

# Python standard libraries
# {{{
import os, sys
import numpy as np
import atomman as am
import subprocess as sp
# }}}

def CSP(systems=None, symbols=None, face=0.5, minneib=0, ref=6):
    # {{{
        """Analyzing the CSP using dumpana.
        
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
        ref: int, optional
            the nearest neighbors of ref lattice.

        Returns
        -------
        systems: atomman.System
        """
        # Check args
        symbols = symbols if isinstance(symbols, str) else " ".join(symbols)
        systems.dump('atom_dump', f='.csp.data',
                prop_name=['atom_id', 'atype', 'pos'])
        with open('.dumpana', 'w') as f:
            f.write(symbols+'\n4\n1\n3\n'+str(face)+'\n'
                    +str(minneib)+'\n\n\n'+str(ref)+'\n')
        cmdline = 'dumpana -1 .csp.data < .dumpana'
        sp.run(cmdline, shell=True)
        os.remove('.dumpana')
        os.remove('.csp.data')
        tmp_data = np.genfromtxt('csp.dat')
        systems.atoms.view['csp']= tmp_data[:, 5]
        os.remove('csp.dat')
        return systems
    # }}}

if __name__ == "__main__":
    import mdAna as ma
    frames = ma.Frame('atom.data')
    timesteps, systems = frames.read_single()
    results = CSP(systems=systems, symbols=['Cu', 'Ag'], ref=12)
    print(results.atoms.csp)
