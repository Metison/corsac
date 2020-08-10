# Written by Metison Wood <wubqmail(at)163.com> under DWYW license
# Standard Python libraries
# {{{
import os
import sys
import atomman as am
import numpy as np
import random
from math import sqrt
# }}}

class Model(object):
    """a class for handling model."""
    def __init__(self, outpath='', outstyle='atom_data'):
    # {{{
        """Initial a model class.
        
        Parameters
        ----------
        outpath: str, optional
            the path of output.
        outstyle: str, optional
            the style of output.
        """
        self.outpath = outpath
        self.outstyle = outstyle
    # }}}
    def centered(self, systems=None):
    # {{{
        """Set the origin as geometric center.

        Parameters
        ----------
        systems: atomman.System

        Returns
        -------
        atomman.System
        """
        pos = systems.atoms.pos
        box = systems.box
        systems.atoms.pos[:, 0] = pos[:, 0] - (box.xlo+box.xhi)/2
        systems.atoms.pos[:, 1] = pos[:, 1] - (box.ylo+box.yhi)/2
        systems.atoms.pos[:, 2] = pos[:, 2] - (box.zlo+box.zhi)/2
        systems.box.set(xlo=-(box.a)/2, xhi=(box.a)/2,
                ylo=-(box.b)/2, yhi=(box.b)/2,
                zlo=-(box.c)/2, zhi=(box.c)/2)
        return systems
    # }}}
    def mismatch(self, cell_a=None, cell_b=None, ranges=100,
            direction='y'):
    # {{{
        """Compute the minmum mismatch.
        Note: minimize the s where s = m*x-n*y
        
        Parameters
        ----------
        cell_a: atomman.System
            the cell of lattice A.
        cell_b: atomman.System
            the cell of lattice B.
        ranges: int or [int, int], optional
            the range of units num
        direction: str, optional
            the surface used for coupling
        
        Returns
        -------
        m, n, k, j
        """
        # Check args
        ranges = [ranges, ranges] if isinstance(ranges, int) else ranges
        boxa = [cell_a.box.lx, cell_a.box.ly, cell_a.box.lz]
        boxb = [cell_b.box.lx, cell_b.box.ly, cell_b.box.lz]
        if direction == 'z':
            a, b = 0, 1
        elif direction == 'y':
            a, b = 0, 2
        elif direction == 'x':
            a, b = 1, 2
        s1 = []
        s2 = []
        for i in range(1, ranges[0]+1):
            for j in range(1, ranges[1]+1):
                s1.append([i, j, abs(i*boxa[a]-j*boxb[a])])
                s2.append([i, j, abs(i*boxa[b]-j*boxb[b])])
        s1 = np.array(s1)
        s2 = np.array(s2)
        mins1 = s1[:,2].min()
        mins2 = s2[:,2].min()
        m = s1[:,0][s1[:,2] == mins1]
        n = s1[:,1][s1[:,2] == mins1]
        k = s2[:,0][s2[:,2] == mins2]
        j = s2[:,1][s2[:,2] == mins2]
        return m[0], n[0], k[0], j[0], mins1, mins2
    # }}}
    def misfit(self, a=None, b=None, ranges=100):
    # {{{
        """Compute the minmum misfit and return m,n
        Note: minimize the s where: s = 2*(m*a-n*b)/(m*a+n*b)
        
        Parameters
        ----------
        a: float
            the constant of lattice A.
        b: float
            the constant of lattice B.
        ranges: int, optional
            the range of m, n
            
        Returns
        -------
        m, n (int)
        """
        s = []
        for m in range(1, ranges+1):
            for n in range(1, ranges+1):
                s.append([m, n, abs(2*(m*a-n*b)/(m*a+n*b))])
        s = np.array(s)
        mins = s[:,2].min()
        m = s[:,0][s[:,2] == mins]
        n = s[:,1][s[:,2] == mins]
        return m[0], n[0], mins
    # }}}
    def closest(self, cell=None, boxsize=100):
    # {{{
        """Get the closest box of a specified boxsize.
        
        Parameters
        ----------
        cell: atomman.System
            cell to supersize
        boxsize: int or [int, int, int], optional
            the size of box
        
        Returns
        -------
        atomman.System
        supersize: [int, int, int]
        """
        # Check args
        boxsize = [boxsize, boxsize, boxsize]\
                if isinstance(boxsize, int) else boxsize
        if len(boxsize) != 3:
            raise ValueError('boxsize must be int or [int, int, int]')
        x, y, z = cell.box.lx, cell.box.ly, cell.box.lz
        x_size = int(round(boxsize[0]/x))
        y_size = int(round(boxsize[1]/y))
        z_size = int(round(boxsize[2]/z))
        systems = cell.supersize(x_size, y_size, z_size)
        supersize = [x_size, y_size, z_size]
        return systems, supersize
    # }}}
    def alloy(self, systems=None, fracs=0.5, seeds=952770):
    # {{{
        """Random addition of alloying elements.
        
        Parameters
        ----------
        systems: atomman.System
            system of elements.
        fracs: float or [float, float], optional
            the proportion of alloying element A or A, B
            
        Returns
        -------
        atomman.System
        """
        # Check args
        natoms = systems.natoms
        id_all = [i for i in range(0, natoms)]
        if isinstance(fracs, float):
            random.seed(a=seeds)
            sel_natoms = round(natoms * fracs)
            id_atoms = random.sample(range(0, natoms), sel_natoms)
            systems.atoms.atype[id_atoms] = 2
            return systems
        if len(fracs) == 2:
            seeds = [seeds, seeds] if isinstance(seeds, int) else seeds
            sel_a = round(natoms * fracs[0])
            sel_b = round(natoms * fracs[1])
            random.seed(a=seeds[0])
            id_a = random.sample(range(0, natoms), sel_a)
            systems.atoms.atype[id_a] = 2
            tmp_b = np.setdiff1d(id_all, id_a)
            random.seed(a=seeds[1])
            id_b = tmp_b[random.sample(range(0, len(tmp_b)), sel_b)]
            systems.atoms.atype[id_b] = 3
            return systems
    # }}}
    def AB(self, systems_a=None, systems_b=None, interval=2.5,
            symbols_a=None, symbols_b=None, direction='z'):
    # {{{
        """Create a two layer lattice.
        
        Parameters
        ----------
        systems_a: atomman.System
            the upper lattice.
        systems_b: atomman.System
            the under lattice.
        interval: float, optional
            the interval between the two lattice.
        symbols_a: a list of int.
            the elements list.
        symbols_b: a list of int.
            the elements list.
        direction: str, optional
            the direction of stack.

        Returns
        -------
        atomman.System
        """
        # Check args
        if direction == 'z':
            sel = 2
        elif direction == 'y':
            sel = 1
        elif direction == 'x':
            sel = 0
        if symbols_a is None:
            symbols_a = systems_a.atoms.atypes
        if symbols_b is None:
            symbols_b = systems_b.atoms.atypes
        sys_a = self.centered(systems=systems_a)
        sys_b = self.centered(systems=systems_b)
        # Atoms
        # {{{
        # atype
        a_atypes = sys_a.atypes
        b_atypes = sys_b.atypes
        a_atype = sys_a.atoms.atype
        b_atype = sys_b.atoms.atype
        for i in zip(a_atypes, symbols_a):
            a_atype[a_atype==i[0]] = i[1]
        for i in zip(b_atypes, symbols_b):
            b_atype[b_atype==i[0]] = i[1]
        atype = np.hstack((a_atype, b_atype))
        # pos
        a_pos = sys_a.atoms.pos
        b_pos = sys_b.atoms.pos
        b_pos[:, sel] = b_pos[:, sel] + a_pos[:, sel].max() -\
                b_pos[:, sel].min() + interval
        pos = np.vstack((a_pos, b_pos))
        atoms = am.Atoms(atype=atype, pos=pos)
        # }}}
        # Box
        # {{{
        # box avect, bvect, cvect
        a_box, b_box = sys_a.box, sys_b.box
        tmp_l = pos[:, sel].max() - pos[:, sel].min() + interval
        box_size = [max([a_box.lx, b_box.lx]),
                max([a_box.ly, b_box.ly]),
                max([a_box.lz, b_box.lz])]
        box_size[sel] = tmp_l
        avect = [box_size[0], 0, 0]
        bvect = [0, box_size[1], 0]
        cvect = [0, 0, box_size[2]]
        origin = sys_a.box.origin
        box = am.Box()
        # }}}
        box.set(vects=[avect, bvect, cvect])
        box.set(origin=origin)
        systems = am.System(atoms=atoms, box=box)
        #return self.centered(systems)
        return systems
    # }}}
    def dump(self, systems=None, fname='model.data'):
    # {{{
        """Dump a model.

        Parameters
        ----------
        systems: atomman.System
            the stystem to output.
        
        Returns
        -------
        files
        """
        systems.dump(self.outstyle, f=os.path.join(self.outpath, fname))
        return 0
    # }}}

def interplanar_spacing(constant=3.615, miller=[0, 0, 1], style='fcc'):
    # {{{
    """Computing the interplanar spacing of fcc, bcc, sc
    
    Parameters
    ----------
    constant: float
        the lattice constant.
    miller: [int, int, int], optional
        miller index.
    style: str, optional
        sc, fcc, bcc.

    Returns
    -------
    float
    """
    a, b, c = miller[0], miller[1], miller[2]
    if style == 'sc':
        return constant/sqrt(a**2 + b**2 + c**2)
    elif style == 'fcc':
        if a == 1 and b == 1 and c == 1:
            return constant/sqrt(a**2 + b**2 + c**2)
        else:
            return constant/sqrt(a**2 + b**2 + c**2)/2
    elif style == 'bcc':
        if (a+b+c)%2 == 0:
            return constant/sqrt(a**2 + b**2 + c**2)
        else:
            return constant/sqrt(a**2 + b**2 + c**2)/2

    # }}}

if __name__ == "__main__":
    m = Model()
    ca = am.load('atom_dump', '111.cell')
    #cb = am.load('atom_dump', '111.cell')
    #results = m.AB(systems_a=ca, systems_b=cb)
    systems = m.closest(cell=ca)
    #print(systems)
    #results = m.alloy(systems=systems, fracs=[0.5, 0.2])
    #results = m.misfit(a=3.625, b=4.065, ranges=100)
    #ca = am.load('atom_data', 'a.cell')
    #cb = am.load('atom_data', 'b.cell')
    #results = m.mismatch(cell_a=ca, cell_b=cb, ranges=100)
    #results = interplanar_spacing()/40
    results = m.dump(systems=systems)
    print(results)
