# Written by Metison Wood <wubqmail(at)163.com> under DWYW license
# To analyze the voronoi index of an atomman.System()

# Python standard libraries
# {{{
from __future__ import (absolute_import, print_function, division,
        unicode_literals)
import os,sys,shutil
import subprocess as sp
import numpy as np
import pandas as pd
from ..tools import elements
# }}}

class Voronoi(object):
    """A class for analyzing the voronoi index of an atomman.System.
    There are two method to analyze the voronoi index:
        origvoro: using voro++, but less limit edge and face.
        revoro: using dumpana, with limit edge and face limited.
    """
    def __init__(self, systems=None, symbols=None, radius=None, ndim=3):
    # {{{
        """Init a voronoi class.
        
        Parameters
        ----------
        systems: atomman.System
        symbols: str, or a list of str
            the elements of systems.
        radius: float, or a list of float, optional
            the radius of elements, default from ptables.
        ndim: int, optional
            the number of dimension.
        """
        # Check args
        if symbols is None:
            raise ValueError('symbols is needed.')
            sys.exit(1)
        symbols = [symbols] if isinstance(symbols, str) else symbols
        if radius is not None:
            radius = radius
            radius = {1: radius[0], 2: radius[1]}
        else:
            table = elements()
            radius = table[symbols].loc['radius'].values
            mradius = {}
            for i in range(1, len(radius)+1):
                mradius[i] = radius[i-1]
        self.radius = mradius
        self.symbols = symbols
        atoms = systems.atoms
        atomRadius = np.array(pd.Series(atoms.atype).map(mradius))
        posRadius = np.column_stack((atoms.pos, atomRadius))
        self.voroin = np.column_stack((np.arange(atoms.natoms)+1, posRadius))
        boxs = systems.box
        self.bounds = [boxs.xlo, boxs.xhi, boxs.ylo, boxs.yhi,
                boxs.zlo, boxs.zhi]
        self.ndim = ndim
        self.systems = systems
    # }}}
    def origvoro(self, outfile=True, outpath=''):
    # {{{
        """Radical voronoi tessellation using voro++, for Period.
        
        Parameters
        ----------
        outfile: bool, optional
            True for saving files.
        outpath: str, optional
            the path for saving.

        Returns
        -------
        systems: atomman.System
            systems contain: CN, voroV, voroS, voro5,
                voroIndex(str), voroFace(str), voroNeib(str)
        files: voroinfo.dat, optional.

        Notes
        -----
        file contain:
            # id radii x y z vol areas numface face_edge neib faceareas
        and the atomman.System's voroinfo can't output to file.
        """
        formats =  '%d '+ '%.8f ' * self.ndim + ' %.4f'
        np.savetxt('.voroin', self.voroin, fmt=formats)
        cmdline = 'voro++ -p -r -o -c "%i %r %x %y %z %v %F %s %a %n %f" '\
                + ('%f ' * 6 % tuple(self.bounds)) + '.voroin'
        sp.run(cmdline, shell=True)
        os.remove('.voroin')
        CN, voroV, voroS, voro5 = [], [], [], []
        v_index, v_face, v_neib = [], [], []
        with open('.voroin.vol', 'r') as f:
            for line in f:
                tmp_data = line.split()
                tmp_cn = int(tmp_data[7])
                CN.append(tmp_cn)
                voroV.append(float(tmp_data[5]))
                voroS.append(float(tmp_data[6]))
                tmp_voro = tmp_data[8:]
                tmp_index = tmp_voro[:tmp_cn]
                voro5.append(tmp_index.count('5'))
                v_index.append(' '.join(tmp_index))
                v_neib.append(' '.join(tmp_voro[tmp_cn:2*tmp_cn]))
                v_face.append(' '.join(tmp_voro[2*tmp_cn:]))
            self.systems.atoms.CN = CN
            self.systems.atoms.voroV = voroV
            self.systems.atoms.voroS = voroS
            self.systems.atoms.voro5 = voro5
            self.systems.atoms.voroIndex = v_index
            self.systems.atoms.voroFace = v_face
            self.systems.atoms.voroNeib = v_neib
        if outfile:
            os.rename('.voroin.vol', 'voroinfo.dat')
            shutil.move('voroinfo.dat', os.path.join(outpath,
                'voroinfo.dat'))
        else:
            os.remove('.voroin.vol')
        return self.systems
    # }}}
    def revoro(self, edge=0.01, face=0.5, minneib=0, outfile=True, outpath=''):
    # {{{
        """Radical voronoi tessellation using dumpana.
        
        Parameters
        ----------
        edge: float, optional.
            edge limit.
        face: float, optional.
            face limit.
        minneib: int, optional.
            the minmum neighbor.
        outfile: bool, optional
            True for saving files.
        outpath: str, optional
            the path for saving.

        Returns
        -------
        systems: atomman.System
            systems contain: CN, voroV, voro5,
                voroIndex(str), voroFace(str), voroNeib(str)
        files: voroinfo.dat, optional.

        Notes
        -----
        file('voroinfo.dat') contain:
            # id type x y z vol index numneib neib faceareas
        and the atomman.System's voroinfo can't output to file.
        """
        print("Analyzing the voronoi now...")
        self.systems.dump('atom_dump', f='.voroin',
                prop_name=['atom_id', 'atype', 'pos'])
        with open('.dumpana', 'w') as f:
            f.write(' '.join(self.symbols)+'\n1\n1\n'+str(face)
                    +'\n'+str(minneib)+'\n'+str(edge)+'\n.voro')
        cmdline = 'dumpana -1 .voroin < .dumpana'
        sp.run(cmdline, shell=True, stdout=sp.DEVNULL)
        os.remove('.dumpana')
        os.remove('.voroin')
        INDEX, CN, voroV, voro5 = [], [], [], []
        v_index, v_face, v_neib = [], [], []
        with open('.voro_1.dat', 'r') as f:
            j = 0
            for line in f:
                if j < 4:
                    j += 1
                    continue
                tmp_data = line.split()
                INDEX.append(int(tmp_data[0]))
                tmp_cn = int(tmp_data[8])
                CN.append(tmp_cn)
                voroV.append(float(tmp_data[5]))
                voro5.append(int(tmp_data[7]))
                v_index.append(' '.join(tmp_data[6].split(',')))
                tmp_voro = tmp_data[9:]
                v_neib.append(' '.join(tmp_voro[:tmp_cn]))
                v_face.append(' '.join(tmp_voro[tmp_cn:2*tmp_cn]))
            all_data = {'INDEX': INDEX,
                    'ATYPE': self.systems.atoms.atype,
                    'CN': CN, 'voroV': voroV,
                    'voro5': voro5, 'voroINDEX': v_index,
                    'voroFACE': v_face, 'voroNEIBS': v_neib}
            DATA = pd.DataFrame(data=all_data).sort_values('INDEX')
            self.systems.atoms.CN = DATA.CN
            self.systems.atoms.voroV = DATA.voroV
            self.systems.atoms.voro5 = DATA.voro5
            self.systems.atoms.voroINDEX = DATA.voroINDEX
            self.systems.atoms.voroFACE = DATA.voroFACE
            self.systems.atoms.voroNEIBS = DATA.voroNEIBS
        if outfile:
            DATA.to_csv(os.path.join(outpath, 'voroinfo.dat'), index=False)
        os.remove('.voro_1.dat')
        return self.systems
    # }}}
    def vorocna(self, edge=0.01, face=0.5, minneib=0, outfile=True, outpath=''):
    # {{{
        """Common neighbor analysis using dumpana.
        
        Parameters
        ----------
        edge: float, optional.
            edge limit.
        face: float, optional.
            face limit.
        minneib: int, optional.
            the minmum neighbor.
        outfile: bool, optional
            True for saving files.
        outpath: str, optional
            the path for saving.

        Returns
        -------
        systems: atomman.System
            systems contain: CN, N5, CNA(str), Neib(str)
        files: cnainfo.dat, optional.

        Notes
        -----
        file contain:
            # id type x y z CN CNA Neib
        and the atomman.System's cnainfo can't output to file.
        """
        self.systems.dump('atom_dump', f='.cnain',
                prop_name=['atom_id', 'atype', 'pos'])
        with open('.dumpana', 'w') as f:
            f.write(' '.join(self.symbols)+'\n3\n1\n'+str(face)
                    +'\n'+str(minneib)+'\nn\nn\nn\n.cna.dat')
        cmdline = 'dumpana -1 .cnain < .dumpana'
        sp.run(cmdline, shell=True, stdout=sp.DEVNULL)
        os.remove('.dumpana')
        os.remove('.cnain')
        tmp_data = np.genfromtxt('.cna.dat', dtype=np.int32)
        INDEX, CN, CNA, NEIBS = [], [], [], []
        for i in range(1, self.systems.natoms+1):
            print(i)
            INDEX.append(i)
            tmp_res = tmp_data[np.logical_or(
                tmp_data[:,0] == i, tmp_data[:,1] == i)]
            CN.append(len(tmp_res))
            CNA.append(" ".join(tmp_res[:,2].astype(np.str)))
            tmp_neib = list(tmp_res[:,0][tmp_res[:,0] != i]) +\
                    list(tmp_res[:,1][tmp_res[:,1] != i])
            #Neib.append(" ".join([str(k) for k in tmp_neib]))
            NEIBS.append(" ".join(np.array(tmp_neib).astype(np.str)))
        all_data = {'INDEX': INDEX, 'ATYPE': self.systems.atoms.atype,
                'CN': CN, 'CNA': CNA, 'NEIBS': NEIBS}
        DATA = pd.DataFrame(data=all_data)
        self.systems.atoms.CN = DATA.CN
        self.systems.atoms.CNA = DATA.CNA
        self.systems.atoms.NEIBS = DATA.NEIBS
        os.remove('.cna.dat')
        if outfile:
            DATA.to_csv(os.path.join(outpath, 'vorocna.dat'), index=False)
        return self.systems
    # }}}

if __name__ == "__main__":
    import atomman as am
    systems = am.load('atom_dump', 'test.data')
    v = Voronoi(systems=systems, symbols=['Cu', 'Ag'])
    #v.origvoro()
    results = v.revoro()
    results = v.vorocna()
    print(results)
