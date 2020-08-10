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

def BOO(systems=None, symbols=None, face=0.5, minneib=0, l=6):
    # {{{
        """Analyzing the BOO using dumpana.
        
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
        l: int, optional
            the l of qml.

        Returns
        -------
        systems: atomman.System
        """
        # Check args
        symbols = symbols if isinstance(symbols, str) else " ".join(symbols)
        l = str(l)
        filename = 'q'+l+'q'+l+'_1.dat'
        systems.dump('atom_dump', f='.boo.data',
                prop_name=['atom_id', 'atype', 'pos'])
        with open('.dumpana', 'w') as f:
            f.write(symbols+'\n9\n1\n'+l+'\nall\n\n0\n'+str(face)
                    +'\n'+str(minneib))
        cmdline = 'dumpana -1 .boo.data < .dumpana'
        sp.run(cmdline, shell=True, stdout=sp.DEVNULL)
        os.remove('.dumpana')
        os.remove('.boo.data')
        tmp_data = np.genfromtxt(filename)
        systems.atoms.view['q'+l]= tmp_data[:, 5]
        systems.atoms.view['w'+l]= tmp_data[:, 6]
        systems.atoms.view['q'+l+'q'+l]= tmp_data[:, 7]
        os.remove(filename)
        return systems
    # }}}

if __name__ == "__main__":
    import mdAna as ma
    frames = ma.Frame('atom.data')
    timesteps, systems = frames.read_single()
    results = BOO(systems=systems, symbols=['Cu', 'Ag'], l=4)
    print(results.atoms.q4)
