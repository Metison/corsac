# Written by Metison Wood <wubqmail(at)163.com> under DWYW license
# Create a Common Neighor Analysis

# Python standard libraries
# {{{
import os,sys
import atomman as am
import numpy as np
import pandas as pd
import subprocess as sp
import glob
from ..atoms import dump_modify
# }}}

class CNA(object):
    """A structure analysis methods."""
    def __init__(self, systems=None):
    # {{{
        """Initial a class for Common Neighbor Analysis.
        
        Parameters
        ----------
        systems: atomman.System().
        """
        self.systems = systems
        self.pos = systems.atoms.pos
    # }}}
    def mcna(self, rcutoff=2.8, dr=0.01, initialsize=20,
            deltasize=5, outfile=True, outpath=''):
    # {{{
        """A modified method of CNA. Not stable.
        
        Parameters
        ----------
        rcutoff: float, optional.
            the initial rcutoff.
        dr: float, optional.
            the search inc of rcutoff
        intialsize: int, optional
            the initialsize of neighbor.
        deltasize: int, optional
            atomman's doc
        outfile: bool, optional
            default(True) for output file.
        outpath: str, optional
            the path for output. default is current workdir.
        """
        neighbors = am.NeighborList(system=self.systems,
                cutoff=rcutoff+3.0, initialsize=initialsize,
                deltasize=deltasize)
        r_atom_id, r_atype, r_CN, r_N5, r_Neibs, r_Bondtypes = \
                [], [], [], [], [], []
        j = 0
        for i in zip(range(self.systems.natoms), neighbors):
            atom_id, atype, CN, N5, Neibs, Bondtypes = self.lsc(
                    atom=i[0], neibs=i[1], dr=dr, rcutoff=rcutoff)
            r_atom_id.append(atom_id)
            r_atype.append(atype)
            r_CN.append(CN)
            r_N5.append(N5)
            r_Neibs.append(Neibs)
            r_Bondtypes.append(Bondtypes)
            if j > 100:
                break
            j = j+1
        results = pd.DataFrame({'INDEX': r_atom_id,
            'ATYPE': r_atype, 'CN': r_CN, 'N5': r_N5,
            'NEIBS': r_Neibs, 'BTYPES': r_Bondtypes})
        if outfile:
            results.to_csv(os.path.join(outpath, 'lscinfo.dat'),
                    sep=',', index=False)
        return results
    # }}}
    def lsc(self, atom=None, neibs=None, rcutoff=None, dr=None):
    # {{{
        """Find the largest standard cluster.
        
        Parameters
        ---------
        atom: int
            the id of atom in center.
        neibs: a list of int
            the neighbors of center atom.
        rcutoff: float
            the initial rcut.
        dr: float
            delta rcutoff.
        
        Returns
        -------
        atom_id, atype, CN, N5, Neibs, Bondtypes: int, int, int, int, str, str.
        """
        # Set variables
        rcut = rcutoff
        dvects = self.systems.dvect(self.pos[atom], self.pos[neibs])
        dvects = np.sqrt(np.square(dvects[:,0]) +
                np.square(dvects[:,1]) +
                np.square(dvects[:,2]))
        j = 0
        while True:
            j += 1
            if j > 100:
                raise ValueError('loop too much')
                sys.exit(1)
            # Find the tmporary cluster
            t_neibs = neibs[dvects < rcut]
            if len(t_neibs) <= 8:
                rcut = rcut + dr
                continue
            # Check the non-lsc
            c_flag = self.checklsc(atom, t_neibs, rcut)
            if not c_flag:
                rcut  = rcut + dr
            # Remove the longest bond to lsc
            # **Note**: May not delete atoms
            else:
                # This part need modify
                HAres, Neibs, N5res = self.tolsc(atom, t_neibs, rcut)
                Neibs = " ".join([str(i) for i in Neibs])
                N5 = N5res
                Atom_id = atom
                CN = len(HAres)
                Atype = self.systems.atoms.atype[atom]
                Bondtypes = " ".join(HAres)
                break
        print(Atom_id, Atype, CN, N5, Neibs, Bondtypes)
        return Atom_id, Atype, CN, N5, Neibs, Bondtypes
    # }}}
    def checklsc(self, atom=None, neibs=None, rcut=None):
    # {{{
        """A function for checking whether is a non-lsc.

        Parameters
        ----------
        atom: int
            center atom.
        neibs: a list of int.
            neighbors
        rcut: float
            the rcutoff

        Returns
        -------
        bool: True for non-lsc, False for lsc.
        """
        for i in neibs:
            rlist = neibs[neibs != i]
            dvects = self.systems.dvect(self.pos[i],
                    self.pos[rlist])
            dvects = np.sqrt(np.square(dvects[:,0]) +
                    np.square(dvects[:,1]) +
                    np.square(dvects[:,2]))
            comneibs = np.array(rlist[dvects <= rcut])
            if len(comneibs) < 3:
                return False
            haflag, haindex = self.checkring(comneibs, rcut)
            if not haflag:
                return True
        return False
    # }}}
    def checkring(self, comneibs=None, rcut=None):
    # {{{
        """A algorithm for checking common neibs.
        
        Parameters
        ----------
        comneibs: a list of int
            common neighbors of two atom.

        Returns
        -------
        haflag, haindex: bool, str
            flag and 'x' or ['555']
        """
        dlists = []
        lneibs = len(comneibs)
        # Compute the dvects between any two points.
        for i in range(lneibs-1):
            for j in range(i+1, lneibs):
                a = comneibs[i]
                b = comneibs[j]
                c = self.systems.dvect(self.pos[a], self.pos[b])
                c = np.sqrt(np.square(c[0]) +
                        np.square(c[1]) + np.square(c[2]))
                dlists.append([a, b, round(c, 5)+0.00001])
        dlists = np.array(dlists)
        cdatas = dlists[:,:2][dlists[:, 2] <= rcut].astype('int')
        cdatas = [list(i) for i in cdatas]
        # Only 0 or 1 bond
        if len(cdatas) <= 1:
            haindex = str(len(comneibs))+str(len(cdatas))*2
        # More than 1 bond
        elif len(cdatas) > 1:
            data = pd.Series(np.array(cdatas).ravel())
            # Check multipoints
            # =================
            if data.value_counts().iloc[0] > 2:
                haindex = 'x'
                haflag = False
                return haflag, haindex
            numb = len(cdatas)
            # Find the largest bond.
            for i in range(numb):
                for j in range(numb):
                    x = list(set(list(cdatas[i])+list(cdatas[j])))
                    y = len(list(cdatas[i])) + len(list(cdatas[j]))
                    if i == j:
                        break
                    if len(x) < y:
                        cdatas[i] = x
                        cdatas[j] = ['x']
            rings = np.array([len(i)-1 for i in cdatas if i != ['x']])
            # Only one ring
            if len(rings) == 1:
                if rings[0] < numb:
                    haindex = str(lneibs)+str(rings[0]+1)*2
                elif rings[0] == numb:
                    haindex = str(lneibs)+str(numb)*2
            # Multi ring  && Check subring.
            elif len(rings) > 1:
                # Check subring
                # =============
                if np.sum(rings) < numb:
                    haindex = 'x'
                    haflag = False
                    return haflag, haindex
                elif np.sum(rings) == numb:
                    haindex = str(lneibs)+str(numb)+str(np.max(rings))
        haflag = True
        return  haflag, haindex
    # }}}
    def tolsc(self, atom=None, neibs=None, rcut=None):
    # {{{
        """Remove the longest bond until to lsc.

        Parameters
        ----------
        atom: int
            center atom.
        neibs: a list of int.
            neighbors
        rcut: float
            the rcutoff

        Returns
        HAres, Neibs, N5res: a list of int, a list of int, int
        """

        # Obtain all sub-clusters.
        subs = []
        for i in neibs:
            rlist = neibs[neibs != i]
            dvects = self.systems.dvect(self.pos[i],
                    self.pos[rlist])
            dvects = np.sqrt(np.square(dvects[:,0]) +
                    np.square(dvects[:,1]) +
                    np.square(dvects[:,2]))
            bools = dvects <= rcut
            comneibs = np.array(rlist[bools])
            subs.append([[atom, i], comneibs])
        # Obtain all bond-length under rcut.
        lengths = []
        l = len(neibs)
        allatom = list(neibs) + [atom]
        for i in range(l-1):
            for j in range(i+1, l):
                a = allatom[i]
                b = allatom[j]
                c = self.systems.dvect(self.pos[a], self.pos[b])
                c = np.sqrt(np.square(c[0]) +
                        np.square(c[1]) + np.square(c[2]))
                if c <= rcut:
                    lengths.append(round(c, 5)+0.00001)
        lengths = list(set(lengths))
        lengths.sort(reverse=True)
        l = len(lengths)
        for i in range(l-1):
            maxbond = lengths[i]
            mrcut = lengths[i+1]
            loopflag = 0
            # If center atom in longest bond, remove the tempneib.
            # Find center-longest
            clongest = []
            for j in subs:
                c = self.systems.dvect(self.pos[j[0][0]],
                        self.pos[j[0][1]])
                c = np.sqrt(np.square(c[0]) +
                        np.square(c[1]) + np.square(c[2]))
                if round(c, 5)+0.00001 == maxbond:
                    clongest.append(j[0][1])
            # Analyze all subclusters
            HAres = []
            Neibs = []
            N5res = []
            for j in subs:
                if j[0][1] not in clongest:
                    comneibs = []
                    for k in j[1]:
                        c = self.systems.dvect(self.pos[j[0][1]],
                                self.pos[k])
                        c = np.sqrt(np.square(c[0]) +
                                np.square(c[1]) + np.square(c[2]))
                        if round(c, 5)+0.00001 < maxbond \
                                and k not in clongest:
                            comneibs.append(k)
                    haflag, haindex = self.checkring(comneibs, mrcut)
                    if not haflag:
                        loopflag = 1
                        break
                    else:
                        if len(comneibs) == 5:
                            N5res.append(5)
                        HAres.append(haindex)
                        Neibs.append(j[0][1])
            if loopflag == 0:
                break
        return HAres, Neibs, len(N5res)
    # }}}
    def lsc(self, timestep=None, length='1 1 4.9',
            ru_gr='1 45 20', gr_para='0.08 3.0 5.0',
            sca_para='0.85 1 0', outfile=True, outpath=''):
        # {{{
        """Analyze the Common Neighbors.

        Parameters
        ----------
        timestep: int
            the timestep of dump file.
        length, ru_gr, gr_para, sca_para: all str.
            some paras of MDSAS.
        outfile: bool, optional
            default(True) for output file.
        outpath: str, optional
            the path for output. default is current workdir.

        Returns
        -------
        atomman.System contain prop of lsc
        files:
            contain "atom_id, CN, CNA, Neibs"
        """
        # Dump atom data in lsc style.
        prop_info=[
                {'prop_name': 'atom_id', 'table_name': ['id']},
                {'prop_name': 'atype', 'table_name': ['type']},
                {'prop_name': 'pos', 'table_name': ['x','y','z']}]
        tmp_sys = self.systems.dump('atom_dump', prop_info=prop_info)
        lscin = dump_modify(systems=tmp_sys, timestep=timestep)
        # Input file
        timestep = '1' if timestep == 0 else str(timestep)
        # atom file
        with open('lsca.'+timestep+'.txt', 'w') as f:
            f.write(lscin)
        # num of atom
        num_atoms = str(self.systems.natoms)
        # MDA.INI
        MDA = "pretext lsca\n"+\
                "approach SCA "+num_atoms+"\n"+\
                "period 1 1 1\n"+\
                "datafile 5 1 1 1\n"+\
                "Base-Step "+timestep+" 0 "+timestep+" -1\n"+\
                "length "+length+"\n"+\
                "ru-gr "+ru_gr+"\n"+\
                "gr-para "+gr_para+"\n"+\
                "SCA-para "+sca_para+"\n"+\
                "7 iSite 2\n"+\
                "-1 end 0"
        with open('MDA.INI', 'w') as f:
            f.write(MDA)
        sp.run('MDSAS', shell=True, stdout=sp.DEVNULL)
        # Handling pos file from MDSAS
        for f in glob.glob('lsca*.txt'):
            os.remove(f)
        os.remove('MDSAS.log')
        os.remove('MDA.INI')
        with open('poslsca.'+timestep+'.txt', 'r') as f:
            i = 0
            datas = {'INDEX':[], 'ATYPE':[], 'CN':[], 'LSC':[], 'NEIBS':[]}
            for line in f:
                if i>=8:
                    lsc = line.split('[')[1].split(']')[0].split()
                    lscindex = ''
                    for j in zip(range(0,len(lsc),2),
                            range(1,len(lsc),2)):
                        lscindex += (lsc[j[1]]+' ')*int(lsc[j[0]])
                    datas['LSC'].append(' '.join(lscindex.split()))
                    datas['CN'].append(len(lscindex.split()))
                    allres = line.split()
                    tmp_neibs = ' '.join(allres[5:int(allres[4])+5])
                    datas['NEIBS'].append(tmp_neibs)
                    datas['INDEX'].append(allres[0])
                    datas['ATYPE'].append(self.systems.atoms.atype[
                        int(allres[0])-1])
                i += 1
        datas = pd.DataFrame(datas)
        if outfile:
            datas.to_csv(os.path.join(outpath, 'lscinfo.dat'), index=False)
        # Post processing
        os.remove('poslsca.'+timestep+'.txt')
        self.systems.atoms.LSC = datas['LSC']
        self.systems.atoms.CN = datas['CN']
        self.systems.atoms.NEIBS = datas['NEIBS']
        return self.systems
        # }}}

if __name__=="__main__":
    import mdAna as ma
    frames = ma.atoms.Frame('msd1.lammpstrj')
    timestep, systems = frames.read_single()
    cna = CNA(systems=systems)
    systems = cna.lsc(timestep=timestep)
    print(np.mean(np.array(systems.atoms.CN)))

