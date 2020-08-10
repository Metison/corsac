# Written by Metison Wood <wubqmail(at)163.com> under DWYW license

# Standard Python libraries
# {{{
import os, sys
import numpy  as np
import pandas as pd
import atomman as am
import itertools
from math import pi
import subprocess as sp
# }}}
def wavevector3d(qnum=30):
    # {{{
    """Set wave vector for three dimension.

    Parameters
    ----------
    qnum: int, optional
        the q point of sq.

    Returns
    -------
    numpy.ndarray
    """
    wavenumber = np.square(np.arange(1, qnum))
    wavevector = []
    for i in itertools.product(range(qnum), repeat=3):
        d = i[0]**2 + i[1]**2 + i[2]**2
        if d in wavenumber:
            wavevector.append(np.array([d, i[0], i[1], i[2]]))
    wavevector = np.array(wavevector)
    wavevector = wavevector[wavevector[:, 0].argsort()]
    return wavevector
    # }}}
def wavevector2d(qnum=30):
    # {{{
    """ Set wave vector for two dimension.

    Parameters
    ----------
    qnum: int, optional
        the q point of sq.

    Returns
    -------
    numpy.ndarray
    """
    wavenumber = np.square(np.arange(1, qnum))
    wavevector = []
    for i in itertools.product(range(qnum), repeat=2):
        d = i[0]**2 + i[1]**2
        if d in wavenumber:
            wavevector.append(np.array([d, i[0], i[1]]))
    wavevector = np.array(wavevector)
    wavevector = wavevector[wavevector[:, 0].argsort()]
    return wavevector
    # }}}
class StaticSQ(object):
    """A class of static structure factors."""
    def __init__(self, systems=None, ndim=3, qnum=None, rMax=15.,
            bins=None, partial=True, direction='z', dranges=None,
            outfile=True, outpath=''):
    # {{{
        """
        Initial a static structure factor class.

        Parameters
        ----------
        systems: atomman.System()
            atomic systems.
        ndim: int, optional
            the dimension of gr.
        qnum: int, optional
            q point of wavevector.
        rMax: float, optional
            the maximum radii of sq.
        bins: int, optional
            the bins of radii of sq.
        partial: bool, optional
            True indicates return the partial gr(default).
        direction: str, optional
            'x', 'y', 'z', only valid when the ndim is 2.
        dranges: list of float, optional
            dlo, dhi, only valid when the ndim is 2.
        outfile: bool, optional
            True indicates output the results to file.
        outpath: str, optional
            the path to output, default to currently.
        """
        # Commonly params
        self.ndim, self.rMax, self.bins = ndim, rMax, bins
        self.partial = partial
        # System params
        self.natoms = systems.atoms.natoms
        self.atype = systems.atoms.atype
        self.pos = systems.atoms.pos
        self.edges = np.array([systems.box.lx, systems.box.ly, systems.box.lz])
        boxbounds = [[systems.box.xlo, systems.box.xhi],
                [systems.box.ylo, systems.box.yhi],
                [systems.box.zlo, systems.box.zhi]]
        # Set parameters for 2D
        if self.ndim == 2:
            if direction == "z":
                dcol, poscol = 2, [0,1]
            elif direction == 'y':
                dcol, poscol = 1, [0,2]
            elif direction == 'x':
                dcol, poscol = 0, [1,2]
            # Set dranges
            dranges = boxbounds[dcol] if dranges == None else dranges
            bools = np.logical_and(
                    systems.atoms.pos[:, dcol] > dranges[0],
                    systems.atoms.pos[:, dcol] <= dranges[1])
            self.natoms = len(systems.atoms.pos[bools])
            self.atype = systems.atoms.atype[bools]
            self.pos = systems.atoms.pos[:, poscol][bools]
            self.edges = self.edges[poscol]
        self.waveinc = 2 * pi / self.edges[0]
        # Set qnum && rMax
        if qnum:
            qnum = qnum+1
            if self.rMax is None:
                self.rMax = qnum * self.waveinc
            if self.rMax is not None and int(self.rMax/self.waveinc+1) < qnum:
                qnum = int(self.rMax/self.waveinc+1)
        elif self.rMax is not None and qnum is None:
            qnum = int(self.rMax/self.waveinc+1)
        else:
            raise ValueError('qnum or rMax needed')
        self.rdelta = self.waveinc if self.bins is None else\
                self.rMax/self.bins
        # Check args
        if len(np.unique(self.edges)) != 1:
            raise ValueError('Box is not Cubic')
            sys.exit(1)
        if self.rMax > self.edges.min():
            raise ValueError('rMax must be less than the L')
            sys.exit(1)
        if qnum > int(self.edges.min()/self.waveinc+1):
            raise ValueError('the r of sq must be less than L')
            sys.exit(1)
        if self.rdelta < self.waveinc:
            raise ValueError('rdelta must not less than 2*pi/L')
            sys.exit(1)
        if qnum * self.waveinc < self.rMax:
            raise ValueError('qnum not enough to rMax')
            sys.exit(1)
        # Set post params
        self.wavevector = wavevector3d(qnum=qnum) if ndim == 3\
                else wavevector2d(qnum)
        self.atypes, self.typecounts= np.unique(self.atype,
                return_counts = True)
        # Output params
        self.outfile = outfile
        self.outpath = outpath
    # }}}
    def unary(self):
# {{{
        """Compute the sq of unary system.
        
        Parameters
        ----------
        
        Returns
        -------
        results: numpy.ndarray
            r sq
        file: sq.dat, optional
        """
        qvalue, qcount = np.unique(self.wavevector[:,0], return_counts=True)
        results = np.zeros((len(self.wavevector[:,0]), 2))
        sqres = np.zeros((len(self.wavevector[:,0]), 2))
        for i in range(self.natoms):
            medium = self.waveinc * (self.pos[i] *\
                    self.wavevector[:,1:]).sum(axis=1)
            sqres += np.column_stack((np.sin(medium), np.cos(medium)))
        results[:,1] = np.square(sqres).sum(axis=1) / self.natoms
        results[:,0]  = self.wavevector[:, 0]
        results = pd.DataFrame(results)
        results = np.array(results.groupby(results[0]).mean())
        qvalue = self.waveinc * np.sqrt(qvalue)
        results = np.column_stack((qvalue, results))
        results = self.setbins(results)
        if self.outfile:
            names = 'q  S(q)'
            np.savetxt(os.path.join(self.outpath, 'sq.dat'),
                    results, fmt='%.6f', header=names)
        return results
# }}}
    def binary(self):
    # {{{
        """Compute the sq of binary system.
        
        Parameters
        ----------
        
        Returns
        -------
        results: numpy.ndarray
            r sq s11 s22
        file: sqbinary.dat, optional
        """
        if not self.partial:
            return self.unary()
        qvalue, qcount = np.unique(self.wavevector[:,0], return_counts=True)
        results = np.zeros((len(self.wavevector[:,0]), 4))
        sqres = np.zeros((len(self.wavevector[:,0]), 2))
        sq11 = np.zeros((len(self.wavevector[:,0]), 2))
        sq22 = np.zeros((len(self.wavevector[:,0]), 2))
        for i in range(self.natoms):
            medium = self.waveinc * (self.pos[i] * self.wavevector[:, 1:]).sum(axis = 1)
            sqres += np.column_stack((np.sin(medium), np.cos(medium)))
            if self.atype[i] == 1:
                sq11 += np.column_stack((np.sin(medium), np.cos(medium)))
            if self.atype[i] == 2:
                sq22 += np.column_stack((np.sin(medium), np.cos(medium)))
        results[:,1] += np.square(sqres).sum(axis=1) / self.natoms
        results[:,2] += np.square(sq11).sum(axis=1) / self.typecounts[0]
        results[:,3] += np.square(sq22).sum(axis=1) / self.typecounts[1]
        results[:, 0]  = self.wavevector[:, 0]
        results = pd.DataFrame(results)
        results = np.array(results.groupby(results[0]).mean())
        qvalue = self.waveinc * np.sqrt(qvalue)
        results = np.column_stack((qvalue, results))
        results = self.setbins(results)
        if self.outfile:
            names = 'q  S(q)  S11(q)  S22(q)'
            np.savetxt(os.path.join(self.outpath, 'sqbinary.dat'),
                    results, fmt='%.6f', header = names)
        return results
    # }}}
    def ternary(self):
    # {{{
        """Compute the sq of ternary system.
        
        Parameters
        ----------
        
        Returns
        -------
        results: numpy.ndarray
            r sq s11 s22 s33
        file: sqternary.dat, optional
        """
        if not self.partial:
            return self.unary()
        qvalue, qcount = np.unique(self.wavevector[:,0], return_counts = True)
        results = np.zeros((len(self.wavevector[:,0]), 5))
        sqres = np.zeros((len(self.wavevector[:,0]), 2))
        sq11 = np.zeros((len(self.wavevector[:,0]), 2))
        sq22 = np.zeros((len(self.wavevector[:,0]), 2))
        sq33 = np.zeros((len(self.wavevector[:,0]), 2))
        for i in range(self.natoms):
            medium = self.waveinc * (self.pos[i] *\
                    self.wavevector[:,1:]).sum(axis=1)
            sqres += np.column_stack((np.sin(medium), np.cos(medium)))
            if self.atype[i] == 1:
                sq11 += np.column_stack((np.sin(medium), np.cos(medium)))
            if self.atype[i] == 2:
                sq22 += np.column_stack((np.sin(medium), np.cos(medium)))
            if self.atype[i] == 3:
                sq33 += np.column_stack((np.sin(medium), np.cos(medium)))
        results[:,1] += np.square(sqres).sum(axis=1) / self.natoms
        results[:,2] += np.square(sq11).sum(axis=1) / self.typecounts[0]
        results[:,3] += np.square(sq22).sum(axis=1) / self.typecounts[1]
        results[:,4] += np.square(sq33).sum(axis=1) / self.typecounts[2]
        results[:,0]  = self.wavevector[:,0]
        results = pd.DataFrame(results)
        results = np.array(results.groupby(results[0]).mean())
        qvalue = self.waveinc * np.sqrt(qvalue)
        results = np.column_stack((qvalue, results))
        results = self.setbins(results)
        if self.outfile:
            names = 'q  S(q)  S11(q)  S22(q)  S33(q)'
            np.savetxt(os.path.join(self.outpath,'sqternary.dat'),
                    results, fmt='%.6f', header=names)
        return results
    # }}}
    def quarternary(self):
    # {{{
        """Compute the sq of quarternary system.
        
        Parameters
        ----------
        
        Returns
        -------
        results: numpy.ndarray
            r sq s11 s22 s33 s44
        file: sqquart.dat, optional
        """
        if not self.partial:
            return self.unary()
        qvalue, qcount = np.unique(self.wavevector[:, 0], return_counts = True)
        results = np.zeros((len(self.wavevector[:, 0]), 6))
        sqres = np.zeros((len(self.wavevector[:, 0]), 2))
        sq11    = np.zeros((len(self.wavevector[:, 0]), 2))
        sq22    = np.zeros((len(self.wavevector[:, 0]), 2))
        sq33    = np.zeros((len(self.wavevector[:, 0]), 2))
        sq44    = np.zeros((len(self.wavevector[:, 0]), 2))
        for i in range(self.natoms):
            medium = self.waveinc * (self.pos[i] * self.wavevector[:,1:]).sum(axis = 1)
            sqres += np.column_stack((np.sin(medium), np.cos(medium)))
            if self.atype[i] == 1:
                sq11 += np.column_stack((np.sin(medium), np.cos(medium)))
            if self.atype[i] == 2:
                sq22 += np.column_stack((np.sin(medium), np.cos(medium)))
            if self.atype[i] == 3:
                sq33 += np.column_stack((np.sin(medium), np.cos(medium)))
            if self.atype[i] == 4:
                sq44 += np.column_stack((np.sin(medium), np.cos(medium)))
        results[:,1] += np.square(sqres).sum(axis=1) / self.natoms
        results[:,2] += np.square(sq11).sum(axis=1) / self.typecounts[0]
        results[:,3] += np.square(sq22).sum(axis=1) / self.typecounts[1]
        results[:,4] += np.square(sq33).sum(axis=1) / self.typecounts[2]
        results[:,5] += np.square(sq44).sum(axis=1) / self.typecounts[3]

        results[:, 0]  = self.wavevector[:, 0]
        results = pd.DataFrame(results)
        results = np.array(results.groupby(results[0]).mean())
        qvalue = self.waveinc * np.sqrt(qvalue)
        results = np.column_stack((qvalue, results))
        results = self.setbins(results)
        if self.outfile:
            names = 'q  S(q)  S11(q)  S22(q)  S33(q)  S44(q)'
            np.savetxt(os.path.join(self.outpath, 'sqquart.dat'),
                    results, fmt='%.6f', header=names)
        return results
    # }}}
    def setbins(self, results=None):
    # {{{
        """Set data to bins."""
        if self.bins is None:
            return results[results[:,0]<self.rMax]
        edges = np.linspace(0, self.rMax, num=self.bins+1)
        ledges, redges = edges[:-1], edges[1:]
        indexs = []
        datas = []
        for i in zip(ledges, redges):
            bools = np.logical_and(results[:,0]>i[0], results[:,0]<i[1])
            indexs.append((i[0]+i[1])/2)
            datas.append(np.mean(results[:,1:][bools], axis=0))
        indexs = np.array([indexs]).T
        datas = np.array(datas)
        results = np.concatenate((indexs, datas), axis=1)
        return results
    # }}}

def BTSQ(systems=None, symbols=['Cu', 'Ag'], qmax=6, qpoint=41,
        bins=200, outfile=True, outpath=''):
# {{{
    """Analyzing the Bhatia-Thornton sq.
    
    Parameters
    ----------
    systems: atomman.System
    symbols: [str, str]
        elementsA, elementsB
    qmax: int
    qpoint: int
    outfile: bool, optional
        True indicates output the results to file.
    outpath: str, optional
        the path to output, default to currently.

    Returns
    -------
    results: pd.DataFrame
        columns: q, SNN, SNC, SCC
    file: btsq.dat
    """
    symbols = " ".join(symbols)
    systems.dump('atom_dump', f='.sq.data',
            prop_name=['atom_id', 'atype', 'pos'])
    with open('.dumpana', 'w') as f:
        f.write(symbols + '\n19\n1\nall\n2\n1\n' + str(qmax) +
                '\n' + str(qpoint) +'\n\n' + str(bins) + '\nsq.dat')
    cmdline = 'dumpana -1 .sq.data < .dumpana'
    sp.run(cmdline, shell=True)
    os.remove('.dumpana')
    os.remove('.sq.data')
    results = np.genfromtxt('sq.dat')[:, :-1]
    os.remove('sq.dat')
    columns = ['q', 'SNN', 'SNC', 'SCC']
    results = pd.DataFrame(results, columns=columns)
    if outfile:
        results.to_csv(os.path.join(outpath, 'btsq.dat'),
                index=False, sep=' ', float_format='%.6f')
    return results
# }}}
if __name__ == "__main__":
    systems = am.load('atom_dump', 'atoms.data')
    #sq = StaticSQ(systems=systems, ndim=3, rMax=None, qnum=350,
    #        bins=100, partial=True, direction='z', dranges=None,
    #        outpath='')
    #sq.unary()
    #sq.binary()
    #sq.ternary()
    #sq.quarternary()
    res = BTSQ(systems)
    print(res)
