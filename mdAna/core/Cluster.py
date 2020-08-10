#!/usr/bin/env python
# Written by Metison Wood <wubqmail(at)163.com> under DWYW license
# To analyze the cluster info a atomman.system or neighbor list

# Python standard libraries
# {{{
import os, sys
import numpy as np
import pandas as pd
# }}}

class Cluster(object):
# {{{
    def __init__(self, systems=None, resfile=None, style='voro',
            selc='all', atype='all'):
    # {{{
        """Initial a cluster class.
        
        Parameters
        ----------
        systems: atomman.System(), optional
            a system contain Voronoi or CNA info.
        resfile: str, optional
            the file contains analyzed info.
        style: str, optional
            the style of input data.
        selc: a list of list.
            the selected cluster types. default: all
        """
        self.voroV = None
        # Check args
        if systems is None and resfile is None:
            raise ValueError('Input data is needed')
            sys.exit(1)
        elif systems is not None and resfile is not None:
            raise ValueError('Only one input data is needed')
            sys.exit(1)
        elif systems is not None and resfile is None:
            # Get the data from systems
            data = pd.DataFrame(systems.atoms.atom_id,
                    columns=['INDEX'])
            data['ATYPE'] = systems.atoms.atype
            data['CN'] = systems.atoms.atype
            if style == 'voro':
                self.voroV = systems.atoms.voroV
                data['FLAG'] = systems.atoms.voroINDEX
                data['NEIBS'] = systems.atoms.voroNEIBS
            elif style == 'cna':
                CNA = self.sorted(systems.atoms.CNA)
                data['FLAG'] = CNA
                data['NEIBS'] = systems.atoms.NEIBS
            elif style == 'lsc':
                data['FLAG'] = systems.atoms.LSC
                data['NEIBS'] = systems.atoms.NEIBS
        elif systems is None and resfile is not None:
            tmp_data = pd.read_csv(resfile, index_col=False)
            data = pd.DataFrame(tmp_data.INDEX, columns=['INDEX'])
            data['ATYPE'] = tmp_data.ATYPE
            data['CN'] = tmp_data.CN
            # Get the data from resfile
            if style == 'voro':
                data['FLAG'] = tmp_data.voroINDEX
                data['NEIBS'] = tmp_data.voroNEIBS
                self.voroV = tmp_data.voroV
            elif style == 'cna':
                CNA = self.sorted(tmp_data.CNA.values)
                data['FLAG'] = CNA
                data['NEIBS'] = tmp_data.NEIBS
            elif style == 'lsc':
                data['FLAG'] = tmp_data.LSC
                data['NEIBS'] = tmp_data.NEIBS
        # Select the desired clusters
        self.bool = np.zeros(data['INDEX'].shape) == 0
        selc = [selc] if isinstance(selc, str)\
                and selc != 'all' else selc
        if isinstance(atype, int):
            if selc == 'all':
                self.bool = (data['ATYPE'] == atype)
            else:
                self.b = np.zeros(data['INDEX'].shape) == 1
                for i in selc:
                    self.b = self.b | (data['FLAG'] == i)
                self.bool = self.b & (data['ATYPE'] == atype)
        elif atype == 'all':
            if selc != 'all':
                self.b = np.zeros(data['INDEX'].shape) == 1
                for i in selc:
                    self.b = self.b | (data['FLAG'] == i)
                self.bool = self.b
        # If no clusters are selected.
        if True not in self.bool:
            raise ValueError('no cluster is selected')
            sys.exit(1)
        # Get all datas
        self.data = data
    # }}}
    def neiblist(self, selall=False):
    # {{{
        """Obtain the selected neiblist.
        
        Returns
        -------
        results: list
            indexs, neibs
        """
        if not selall:
            indexs = list(self.data['INDEX'][self.bool])
            tmp_neibs = self.data['NEIBS'][self.bool]
        else:
            indexs = list(self.data['INDEX'])
            tmp_neibs = self.data['NEIBS']
        neibs = []
        for i in tmp_neibs:
            neibs.append([int(j) for j in i.split(' ')])
        print(len(indexs))
        return indexs, neibs
    # }}}
    def clusterhist(self):
    # {{{
        """Analyze the cluster info.

        Returns
        -------
        results: pd.DataFrame
            the histogram of cluster type.
        """
        results = self.data['FLAG'].value_counts()
        return results
    # }}}
    def CN(self):
    # {{{
        """Obtain the CN.
        
        Returns
        -------
        results: list
        """
        results = self.data['CN'][self.bool]
        return results
    # }}}
    def volume(self):
    # {{{
        """Obtain the volume of voro.
        
        Returns
        -------
        results: list
        """
        results = self.voroV[self.bool]
        return results
    # }}}
    def connectivity(self):
    # {{{
        """Analyze the connectivity of selected cluster.
        
        Returns
        -------
        results: list
            vertex, edge, face, penetrate
        """
        print('Analyze the connectivity of cluster now...')
        vertex, edge, face, penetrate = 0, 0, 0, 0
        INDEX, NEIBS = self.neiblist()
        length = len(INDEX)
        # Loop all atoms in a spec type
        for i in range(length):
        # atoms set to compare
            CA = [INDEX[i]] + NEIBS[i]
        # another atoms set
            tmp_neibs = NEIBS[i][:]
            # Obtain the 1th & 2th neibs
            for j in NEIBS[i]:
                tmp_neibs += [int(h) for h in
                        self.data['NEIBS'][j-1].split(' ')]
            # Filter data
            tmp_neibs = list(set(tmp_neibs).intersection(set(INDEX)))
            tmp_neibs = [h for h in tmp_neibs if h > INDEX[i]]
            for k in tmp_neibs:
                l = INDEX.index(k)
                CB = [INDEX[l]] + NEIBS[l]
                # Compare code
                x = len(set(CA)) + len(set(CB))
                y = len(set(CA + CB))
                if x - y == 1:
                    vertex += 1
                elif x - y == 2:
                    edge += 1
                elif x - y == 3:
                    face += 1
                elif x - y >= 4:
                    penetrate += 1
        results = [vertex, edge, face, penetrate]
        return results
    # }}}
    def super(self):
    # {{{
        """Obtain the super clusters."""
        print('Analyze the super cluster now...')
        # Get the neiblist
        tmp_data = []
        INDEX, NEIBS = self.neiblist()
        sindex = set(INDEX)
        for i in zip(INDEX, NEIBS):
            # Only get the same type neibs
            stype = list(set(i[1]).intersection(sindex))
            tmp_data.append([i[0]] + stype)
        # Combine
        b = len(tmp_data)
        for i in range(b):
            for j in range(b):
                x = list(set(tmp_data[i]+tmp_data[j]))
                y = len(list(set(tmp_data[i]))) +\
                        len(list(set(tmp_data[j])))
                if i == j or tmp_data[i] == 0 or tmp_data[j] == 0:
                        break
                elif len(x) < y:
                        tmp_data[i] = x
                        tmp_data[j] = ['x']
        results = [i for i in tmp_data if i != ['x']]
        return results
    # }}}
    def sorted(self, data=None):
    # {{{
        """Sorted the CNA.
        
        Parameters
        ----------
        data: numpy.ndarray.
        """
        results = []
        for i in data:
            tmp_data = [int(j) for j in i.split(' ')]
            tmp_data = sorted(tmp_data)
            results.append(' '.join([str(k) for k in tmp_data]))
        results = np.array(results)
        return results
    # }}}
# }}}

if __name__ == "__main__":
    import mdAna as ma
    frames = ma.Frame('sel.lammpstrj')
    t, s = frames.read_single()
    voro = ma.Voronoi(s, ['Cu', 'Ag'])
    #resvoro = voro.revoro(outfile=False)
    #rescna = voro.vorocna(outfile=False)
    lsc = ma.CNA(s)
    reslsc = lsc.lsc(t, outfile=False)
    selc2 = '1311 1311 1421 1422 1422 1431 1431 1431 1431 1551 1551'
    selc1 = '1431 1431 1541 1541 1551 1551 1551 1551 1551 1551 1551 1551'
    selc = '1551 1551 1551 1551 1551 1551 1551 1551 1551 1551 1551 1551'
    #cluster = Cluster(resfile='vorocna.dat', style='cna',
    #        atype='all', selc=[selc, selc1])
    cluster = Cluster(systems=reslsc, style='lsc',
            atype='all', selc='all')
    i, n = cluster.neiblist()
    res = cluster.clusterhist()
    print(res)
    res = cluster.CN()
    #res = cluster.volume()
    #res = cluster.connectivity()
    #res = cluster.super()
    print('Good Job')
