# Written by Metison Wood <wubqmail(at)163.com> under DWYW license

# Standard Python libraries
# {{{
import os, sys
import numpy as np
import atomman as am
import subprocess as sp
# }}}

class PairCorFunc(object):
    """ Compute the Pair Correlation Functions of an atomman.System."""

    def __init__(self, systems=None, ndim=3, bc=[1,1,1], rMax=15.,
            bins=100, partial=True, direction='z', dranges=None,
            outfile=True, outpath=''):
    # {{{
        """
        Initial a pair correlation function class.

        Parameters
        ----------
        systems: atomman.System()
            atomic systems.
        ndim: int, optional
            the dimension of gr.
        bc: list of int, optional
            boundary conditions of box, 0: no period, 1: period.
        rMax: float, optional
            the maximum radii of gr.
        bins: int, optional
            the bins of radii of gr.
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
        if not isinstance(systems, am.System):
            raise ValueError('systems must be atommman.System')
            sys.exit(1)
        if ndim != 3 and ndim != 2:
            raise ValueError('dimension must be 2 or 3')
            sys.exit(1)
        # Commonly Params
        self.ndim = ndim
        self.dimfac = 4.0/3 if ndim == 3 else 1.0
        self.rdelta = rMax/bins
        self.bc = bc
        self.rMax = rMax
        self.bins = bins
        self.partial = partial
        # System Params
        self.natoms = systems.atoms.natoms
        self.atype = systems.atoms.atype
        self.pos = systems.atoms.pos
        self.edges = np.array([systems.box.lx, systems.box.ly, systems.box.lz])
        self.vects = systems.box.vects
        # Set parameters for 2D
        if self.ndim == 2:
            if direction == "z":
                dcol, poscol = 2, [0,1]
            elif direction == 'y':
                dcol, poscol = 1, [0,2]
            elif direction == 'x':
                dcol, poscol = 0, [1,2]
            bools = np.logical_and(
                    systems.atoms.pos[:, dcol] > dranges[0],
                    systems.atoms.pos[:, dcol] <= dranges[1])
            self.natoms = len(systems.atoms.pos[bools])
            self.atype = systems.atoms.atype[bools]
            self.pos = systems.atoms.pos[:, poscol][bools]
            self.edges = self.edges[poscol]
            self.vects = self.vects[poscol][:, poscol]
            self.bc = [1, 1]
        self.volume = np.prod(np.array(self.edges))
        self.ND = self.natoms/self.volume
        self.atypes, self.typecounts= np.unique(self.atype,
                return_counts = True)
        self.NDtype = self.typecounts/self.volume
        # Output Params
        self.outfile = outfile
        self.outpath = outpath
        # Check args
        if rMax > self.edges.min()/2.:
            raise ValueError('rMax must be less than L/2')
            sys.exit(1)
    # }}}
    def unary(self):
    # {{{
        """Compute the gr of unary system."""
        results = np.zeros(self.bins)
        vectsinv = np.linalg.inv(self.vects)
        for i in range(self.natoms - 1):
            RIJ = self.pos[i+1:] - self.pos[i]
            matrixRIJ = np.dot(RIJ, vectsinv)
            RIJ = np.dot(matrixRIJ - np.rint(matrixRIJ) * self.bc,
                    self.vects) #remove PBC
            distance = np.sqrt(np.square(RIJ).sum(axis=1))
            counts, edges = np.histogram(distance, bins=self.bins,
                    range=(0, self.rMax))
            results += counts

        ledges, redges = edges[:-1], edges[1:]
        Params = self.dimfac * np.pi * (redges**self.ndim - ledges**self.ndim)
        results = results * 2 / self.natoms / (Params * self.ND)
        medges = np.mean(np.vstack((ledges, redges)), axis=0)
        results = np.column_stack((medges, results))
        if self.outfile:
            np.savetxt(os.path.join(self.outpath, 'gr.dat'),
                    results, fmt='%.6f', header='r g(r)')
        return results
    # }}}
    def binary(self):
    # {{{
        """Compute the gr of binary system."""
        if not self.partial:
            return self.unary()
        results = np.zeros((self.bins, 4))
        vectsinv = np.linalg.inv(self.vects)
        for i in range(self.natoms - 1):
            RIJ = self.pos[i+1:] - self.pos[i]
            TIJ = np.c_[self.atype[i+1:], np.zeros_like(
                self.atype[i+1:]) + self.atype[i]]
            matrixRIJ = np.dot(RIJ, vectsinv)
            RIJ = np.dot(matrixRIJ - np.rint(matrixRIJ)*self.bc, self.vects) #remove PBC
            distance = np.sqrt(np.square(RIJ).sum(axis=1))
            counts, edges = np.histogram(distance, bins=self.bins,
                    range=(0, self.rMax))
            results[:, 0] += counts
            countTIJ = TIJ.sum(axis = 1)
            counts, edges = np.histogram(distance[countTIJ == 2],
                bins=self.bins, range=(0, self.rMax)) # 11
            results[:, 1] += counts
            counts, edges = np.histogram(distance[countTIJ == 3],
                bins=self.bins, range=(0, self.rMax)) # 12
            results[:, 2] += counts
            counts, edges = np.histogram(distance[countTIJ == 4],
                bins=self.bins, range=(0, self.rMax)) # 22
            results[:, 3] += counts
        ledges, redges = edges[:-1], edges[1:]
        Params = self.dimfac * np.pi * (redges**self.ndim - ledges**self.ndim)
        results[:,0] = results[:,0]*2/ self.natoms/\
                (Params * self.ND)
        results[:,1] = results[:,1]*2/ self.typecounts[0]/\
                (Params * self.NDtype[0])
        results[:,2] = results[:,2]*2/ Params*self.volume /\
                self.typecounts[0]/ self.typecounts[1]/ 2.0
        results[:,3] = results[:, 3]*2/\
                self.typecounts[1] / (Params * self.NDtype[1])
        medges = np.mean(np.vstack((ledges, redges)), axis=0)
        results = np.column_stack((redges, results))
        if self.outfile:
            names = 'r  g(r)  g11(r)  g12(r)  g22(r)'
            np.savetxt(os.path.join(self.outpath, 'grbinary.dat'),
                    results, fmt='%.6f', header=names)
        return results
    # }}}
    def ternary(self):
    # {{{
        """Compute the gr of ternary system."""
        if not self.partial:
            return self.unary()
        results = np.zeros((self.bins, 7))
        vectsinv = np.linalg.inv(self.vects)
        for i in range(self.natoms - 1):
            RIJ = self.pos[i+1:] - self.pos[i]
            TIJ = np.c_[self.atype[i+1:], np.zeros_like(
                self.atype[i+1:]) + self.atype[i]]
            matrixRIJ = np.dot(RIJ, vectsinv)
            RIJ = np.dot(matrixRIJ - np.rint(matrixRIJ)*self.bc, self.vects) #remove PBC
            distance = np.sqrt(np.square(RIJ).sum(axis = 1))
            counts, edges = np.histogram(distance, bins=self.bins,
                    range=(0, self.rMax))
            results[:, 0] += counts
            countTIJ = TIJ.sum(axis=1)
            Countsub = np.abs(TIJ[:,0] - TIJ[:,1])
            counts, edges = np.histogram(distance[countTIJ==2],
                    bins = self.bins, range=(0, self.rMax))
            results[:, 1] += counts #11
            counts, edges = np.histogram(distance[(countTIJ==4) &
                (Countsub==0)], bins=self.bins, range=(0, self.rMax))
            results[:, 2] += counts #22
            counts, edges = np.histogram(distance[countTIJ==6],
                    bins=self.bins, range=(0, self.rMax))
            results[:, 3] += counts #33
            counts, edges = np.histogram(distance[countTIJ==3],
                    bins=self.bins, range = (0, self.rMax))
            results[:, 4] += counts #12
            counts, edges = np.histogram(distance[countTIJ==5],
                    bins=self.bins, range=(0, self.rMax))
            results[:, 5] += counts #23
            counts, edges = np.histogram(distance[(countTIJ==4) &
                (Countsub==2)], bins=self.bins, range=(0, self.rMax))
            results[:, 6] += counts #13

        ledges, redges = edges[:-1], edges[1:]
        Params = self.dimfac * np.pi * (redges**self.ndim - ledges**self.ndim)
        results[:, 0] = results[:,0]*2/ self.natoms/ (Params * self.ND)
        results[:, 1] = results[:,1]*2/ self.typecounts[0]/ (Params * self.NDtype[0])
        results[:, 2] = results[:,2]*2/ self.typecounts[1]/ (Params * self.NDtype[1])
        results[:, 3] = results[:,3]*2/ self.typecounts[2]/ (Params * self.NDtype[2])
        results[:, 4] = results[:,4]*2/ Params * self.volume/\
                self.typecounts[0] / self.typecounts[1] /2.0
        results[:, 5] = results[:,5]*2/ Params * self.volume/\
                self.typecounts[1] / self.typecounts[2] /2.0
        results[:, 6] = results[:,6]*2/ Params * self.volume/\
                self.typecounts[0] / self.typecounts[2] /2.0
        medges = np.mean(np.vstack((ledges, redges)), axis=0)
        results  = np.column_stack((medges, results))
        if self.outfile:
            names = 'r g(r) g11(r) g22(r) g33(r) g12(r) g23(r) g13(r)'
            np.savetxt(os.path.join(self.outpath,'grternary.dat'),
                    results, fmt='%.6f', header=names)
        return results
    # }}}
    def quarternary(self):
    # {{{
        """Compute the gr of quarternary system."""
        if not self.partial:
            return self.unary()
        results = np.zeros((self.bins, 11))
        vectsinv = np.linalg.inv(self.vects)
        for i in range(self.natoms - 1):
            RIJ = self.pos[i+1:] - self.pos[i]
            TIJ = np.c_[self.atype[i+1:], np.zeros_like(
                self.atype[i+1:]) + self.atype[i]]
            matrixRIJ = np.dot(RIJ, vectsinv)
            RIJ = np.dot(matrixRIJ - np.rint(matrixRIJ)*self.bc, self.vects) #remove PBC
            distance = np.sqrt(np.square(RIJ).sum(axis = 1))
            counts, edges = np.histogram(distance, bins=self.bins,
                    range=(0, self.rMax))
            results[:,0] += counts
            countTIJ = TIJ.sum(axis = 1)
            Countsub = np.abs(TIJ[:, 0] - TIJ[:, 1])
            counts, edges = np.histogram(distance[countTIJ==2],
                    bins=self.bins, range=(0, self.rMax))
            results[:,1] += counts #11
            counts, edges = np.histogram(distance[(countTIJ==4) &
                (Countsub==0)], bins=self.bins, range=(0, self.rMax))
            results[:,2] += counts #22
            counts, edges = np.histogram(distance[(countTIJ==6) &
                (Countsub==0)], bins=self.bins, range=(0, self.rMax))
            results[:,3] += counts #33
            counts, edges = np.histogram(distance[countTIJ==8],
                    bins=self.bins, range=(0, self.rMax))
            results[:,4] += counts #44
            counts, edges = np.histogram(distance[countTIJ==3],
                    bins=self.bins, range=(0, self.rMax))
            results[:,5] += counts #12
            counts, edges = np.histogram(distance[(countTIJ==4) &
                (Countsub==2)], bins=self.bins, range=(0, self.rMax))
            results[:,6] += counts #13
            counts, edges = np.histogram(distance[Countsub==3],
                    bins=self.bins, range=(0, self.rMax))
            results[:,7] += counts #14
            counts, edges = np.histogram(distance[(countTIJ==5) &
                (Countsub==1)], bins=self.bins, range=(0, self.rMax))
            results[:,8] += counts #23
            counts, edges = np.histogram(distance[(countTIJ==6) &
                (Countsub==2)], bins=self.bins, range=(0, self.rMax))
            results[:,9] += counts #24
            counts, edges = np.histogram(distance[countTIJ==7],
                    bins=self.bins, range=(0, self.rMax))
            results[:,10] += counts #34
        ledges, redges = edges[:-1], edges[1:]
        Params = self.dimfac * np.pi * (redges**self.ndim - ledges**self.ndim)
        results[:,0] = results[:,0]*2/ self.natoms/ (Params * self.ND)
        results[:,1] = results[:,1]*2/ self.typecounts[0]/ (Params*self.NDtype[0])
        results[:,2] = results[:,2]*2/ self.typecounts[1]/ (Params*self.NDtype[1])
        results[:,3] = results[:,3]*2/ self.typecounts[2]/ (Params*self.NDtype[2])
        results[:,4] = results[:,4]*2/ self.typecounts[3]/ (Params*self.NDtype[3])
        results[:,5] = results[:,5]*2/ Params * self.volume /\
                self.typecounts[0] / self.typecounts[1] /2.0
        results[:,6] = results[:,6]*2/ Params * self.volume /\
                self.typecounts[0] / self.typecounts[2] /2.0
        results[:,7] = results[:,7]*2/ Params * self.volume /\
                self.typecounts[0] / self.typecounts[3] /2.0
        results[:,8] = results[:,8]*2/ Params * self.volume /\
                self.typecounts[1] / self.typecounts[2] /2.0
        results[:,9] = results[:,9]*2/ Params * self.volume /\
                self.typecounts[1] / self.typecounts[3] /2.0
        results[:,10] = results[:,10]*2/ Params * self.volume /\
                self.typecounts[2] / self.typecounts[3] /2.0
        medges = np.mean(np.vstack((ledges, redges)), axis=0)
        results = np.column_stack((medges, results))
        if self.outfile:
            names = 'r g(r) g11(r) g22(r) g33(r) g44(r) g12(r)\
                    g13(r) g14(r) g23(r) g24(r) g34(r)'
            np.savetxt(os.path.join(self.outpath,'grquart.dat'),
                    results, fmt='%.6f', header=names)
        return results
    # }}}

def gr(systems=None, symbols=None, minv=0, maxv=15, bins=100,
        outfile=True, outpath=''):
    # {{{
        """Analyzing the gr using dumpana.
        
        Parameters
        ----------
        systems: atomman.System
        symbols: str or a list of str.
            eg. 'Cu' or ['Cu', 'Ag'].
        minv: float, optional
            lower bound of gr.
        maxv: float, optional
            upper bound of gr.
        bins: int, optional
            the num of data.
        outfile: bool, optional
            True for saving files.
        outpath: str, optional
            the path for saving.

        Returns
        -------
        results: numpy.ndarray
            r g(r)
        files: gr.dat, optional.

        """
        # Check args
        symbols = symbols if isinstance(symbols, str) else " ".join(symbols)
        # Analyzing gr
        systems.dump('atom_dump', f='.gr.data',
                prop_name=['atom_id', 'atype', 'pos'])
        with open('.dumpana', 'w') as f:
            f.write(symbols + '\n13\n1\n1\nall\n'
                    + str(minv) + '\n' + str(maxv) + '\n'
                    + str(bins) + '\n1\n' + 'gr.dat')
        cmdline = 'dumpana -1 .gr.data < .dumpana'
        sp.run(cmdline, shell=True, stdout=sp.DEVNULL)
        os.remove('.dumpana')
        os.remove('.gr.data')
        results = np.genfromtxt('gr.dat')[:, :2]
        os.remove('gr.dat')
        if outfile:
            np.savetxt(os.path.join(outpath, 'gr.dat'), results,
                    fmt="%.6f", header='r g(r)')
        return results
    # }}}
def gr_partial(systems=None, symbols=None, type1=1, type2=1,
        minv=0, maxv=15, bins=100, outfile=True, outpath=''):
    # {{{
        """Analyzing the gr using dumpana.
        
        Parameters
        ----------
        systems: atomman.System
        symbols: str or a list of str.
            eg. 'Cu' or ['Cu', 'Ag']...
        type1: int or a list of int, optional
            source atoms of gr.
        type2: int or a list of int, optional
            neighbor atoms of gr.
        minv: float, optional
            lower bound of gr.
        maxv: float, optional
            upper bound of gr.
        bins: int, optional
            the num of data.
        outfile: bool, optional
            True for saving files.
        outpath: str, optional
            the path for saving.

        Returns
        -------
        results: numpy.ndarray
            r g(r)
        files: gr.dat, optional.

        """
        # Check args
        symbols = symbols if isinstance(symbols, str) else " ".join(symbols)
        if isinstance(type1, int):
            type1 = 'type = '+str(type1)
        elif isinstance(type1, list):
            tmp_type = []
            for i in type1:
                tmp_type.append('type = '+str(i))
            type1 = " | ".join(tmp_type)
        if isinstance(type2, int):
            type2 = 'type = '+str(type2)
        elif isinstance(type2, list):
            tmp_type = []
            for i in type2:
                tmp_type.append('type = '+str(i))
            type2 = " | ".join(tmp_type)
        # Analyzing gr
        systems.dump('atom_dump', f='.gr.data',
                prop_name=['atom_id', 'atype', 'pos'])
        with open('.dumpana', 'w') as f:
            f.write(symbols + '\n13\n1\n2\n'
                    + type1 + '\n' + type2 + '\n'
                    + str(minv) + '\n' + str(maxv) + '\n'
                    + str(bins) + '\n1\n' + 'gr.dat')
        cmdline = 'dumpana -1 .gr.data < .dumpana'
        sp.run(cmdline, shell=True, stdout=sp.DEVNULL)
        os.remove('.dumpana')
        os.remove('.gr.data')
        results = np.genfromtxt('gr.dat')[:, :2]
        os.remove('gr.dat')
        if outfile:
            np.savetxt(os.path.join(outpath, 'gr.dat'), results,
                    fmt="%.6f", header='r g(r)')
        return results
    # }}}

if __name__ == "__main__":
    systems = am.load('atom_dump', 'atoms.data')
    #results = gr(systems=systems, symbols='Cu', maxv=20, bins=300)
    results = gr_partial(systems=systems, type1=[2, 3],
            symbols='Cu', maxv=20, bins=300, outfile=False)
    print(results)
    #pcf = PairCorFunc(systems=systems, ndim=2, bc=[1,1], rMax=24, bins=100,
    #        partial=True, direction='z', dranges=[-49, 49])
    #results = pcf.ternary()
