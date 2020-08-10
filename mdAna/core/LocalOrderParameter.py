# Written by Metison Wood <wubqmail(at)163.com> under DWYW license 
# Standard Python libraries
# {{{
from __future__ import (absolute_import, print_function,
        division, unicode_literals)
import os
import sys
import numpy as np
import atomman as am
import subprocess as sp
# }}}

class LOP(object):
    """A class for computing the lop value of an atomman.System."""
    def __init__(self, systems=None, elements=None, style="-f"):
    # {{{
        """Initial a lop class.

        Parameters
        ----------
        systems: atommman.System
        elements: str, a list of str
            the elements of systems.
        style: str
            -f[FCC], -hcp[HCP], -b[BCC], -d[diamond]
        """
        # Check args
        if not isinstance(systems, am.System):
            raise ValueError("an atomman.System is needed.")
            sys.exit(1)
        if isinstance(elements, str):
            elements = [elements,]
        elements = " ".join(elements)
        if style not in ['-f', '-hcp', '-b', '-d']:
            raise ValueError('Lattice style not supported')
            sys.exit(1)
        self.systems = systems
        self.elements = elements
        self.style = style
    # }}}
    def dumpana(self, face=0.5, edge=0.01, minnei=0):
    # {{{
        """Computing the lop value by using dumpana.
        
        Parameters
        ----------
        face: str, float, optional
            criterion for tiny surface, 0 to keep all.
        edge: str, float, optional
            criterion for short edge, 0 to keep all.
        minnei: str, int, optional
            keep at least num of neighbors, no matter how tiny the suface is.

        Returns
        -------
        atomman.System
            contain lop prop
        """
        print('Analyzing the lop ...')
        # Check args
        face, edge, minnei = str(face), str(edge), str(minnei)
        # Dump a lammpstrj file
        prop_name = ['atom_id', 'atype', 'pos']
        self.systems.dump('atom_dump', f='.tmp.data',
                prop_name=prop_name, return_prop_info=True)
        # Dumpana 
        dumpana_options = "\n".join([self.elements, '1', '1',
            face, minnei, edge, '.voro'])
        with open('.dumpana', 'w') as f:
            f.write(dumpana_options)
        sp.run('dumpana -1 .tmp.data < .dumpana', shell=True,
                stdout=sp.DEVNULL)
        # lop
        sp.run('lop -i 2 -s -o .lop.data -c 0 '+self.style+
                ' .voro_1.dat', shell=True, stdout=sp.DEVNULL)
        lop_value =np.genfromtxt('.lop.data', usecols=(4))
        self.systems.atoms.lop = lop_value
        os.remove('.dumpana')
        os.remove('.tmp.data')
        os.remove('.voro_1.dat')
        os.remove('.lop.data')
        return self.systems
    # }}}

if __name__ == "__main__":
    from mdAna import Frame
    from mdAna import Dump
    frames = Frame('msd1.lammpstrj')
    timesteps, systems = frames.read_single(num=1)
    lops = LOP(systems=systems, elements='Cu')
    results = lops.dumpana()
    dumps = Dump(timesteps=timesteps, systems=results)
    dumps.dump()
