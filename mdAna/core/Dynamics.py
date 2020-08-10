# Written by Metison Wood <wubqmail(at)163.com> under DWYW license

# Standard Python libraries
# {{{
import os
import numpy  as np
import pandas as pd
from math import pi
from .StaticStructureFactor import wavevector3d
from ..atoms import Frame
# }}}

class Dynamics(object):
    """ A class for computing the dynamic properties.
    Note: only 3D supported"""
    def __init__(self, filepath=None, ndim=3, qmax=2.5, a=1.0, dt=0.001):
    # {{{
        """Initial a class of dynamics.

        Parameters
        ----------
        filepath: str
            file with path.
        ndim: int, optional
            only support 3D now.
        qmax: float or list of float(partial)
            the wavenumber corresponding to the first peak of structure factor.
        a: float, optional
            the cutoff for the overlap function, 1.0(EAM) and 0.3(LJ).
        dt: float, optional
            the timestep of MD simulations, default: ps
        """
        # Input paras
        self.qmax = qmax
        self.a = a
        self.dt = dt
        self.ndim = ndim
        # System paras
        frames = Frame(filepath=filepath)
        self.frames = frames
        timesteps, systems = frames.read_single()
        self.atype = systems.atoms.atype
        self.nframes = frames.nframes
        self.natoms = systems.atoms.natoms
        b = systems.box
        self.boxsize = [b.lx, b.ly, b.lz]
        self.boxbounds = [b.xlo, b.xhi, b.ylo, b.yhi, b.zlo, b.zhi]
        self.atypes = systems.atypes
# }}}
    def single(self, frame1=None, frame2=None, atype=None):
    # {{{
        """Analyzing the dynamics of two frames.
        
        Parameters
        ----------
        frame1: atomman.System
            the ref system.
        frame2: atomman.System
            the target system.
        atype: int, optional
            the atom type.
        
        Returns
        -------
        results: list
            ISF, QT, MSD, NGP
        """
        # Set atype args
        if atype is None:
            TYPESET = np.where(np.array(self.atype) == 1, 1, 1)
            TYPEfrac = 1
        else:
            TYPESET = np.where(np.array(self.atype) == atype, 1, 0)
            TYPEfrac = len(self.atype[self.atype == atype])/self.natoms
        # Get the msd
        msdcol = ['c_Disp[1]', 'c_Disp[2]', 'c_Disp[3]', 'c_Disp[4]']
        RII = np.zeros((self.natoms, 4))
        for i in range(len(msdcol)):
            RII[:, i] = frame2.atoms.prop(msdcol[i]) -\
                    frame1.atoms.prop(msdcol[i])
        # Core code
        cal_ISF = (np.cos(RII[:, :3] * self.qmax).mean(axis = 1)\
                * TYPESET ).sum(axis = 0)/(self.natoms*TYPEfrac)
        cal_QT = ((RII[:, 3] <= self.a) * TYPESET).sum()/(self.natoms*TYPEfrac)
        cal_MSD = (RII[:, 3] * TYPESET).sum()/(self.natoms*TYPEfrac)
        distance = np.square(RII[:, 3]) * TYPESET
        cal_NGP = 3/5.0 * (np.square(distance).sum()\
                / (self.natoms*TYPEfrac))\
                / ((distance.sum()/(self.natoms*TYPEfrac))**2) - 1
        # Output
        results = [abs(cal_ISF), cal_QT, cal_MSD, cal_NGP]
        return results
    # }}}
    def analysis(self, n=1, atype=None, outfile=False, outpath=''):
    # {{{
        """Compute the
            self-intermediate scattering functions ISF
            Overlap function QT
            Mean-square displacements MSD
            non-Gaussian parameter NGP

        Parameters
        ----------
        n: int, optional
            the num of ref frames
        atype: int, optional
            the atom type.
        outfile: bool, optional
            whether output file, default True.
        outpath: str, optional
            the path of output.

        Returns
        -------
        results: np.ndarray
            col: t, ISF, QT, MSD, NPG
        """
        print("Computing now...")
        t1, frame1 = self.frames.read_single(num = n)
        results = []
        for i in range(n+1, self.nframes+1):
            t2, frame2 = self.frames.read_single(num = i)
            datas = self.single(frame1, frame2, atype)
            results.append([(int(t2) - int(t1)) * self.dt] + datas)
        results = np.array(results)
        if outfile:
            header = "t ISF QT MSD NGP"
            np.savetxt(os.path.join(outpath, 'dynamic.dat'),
                    results, fmt="%.6f", header=header)
        return results
    # }}}
    def anaiso(self, atype=None, outfile=True, outpath=''):
    # {{{
        """Compute the
            self-intermediate scattering functions ISF
            dynamic susceptibility ISFX4 based on ISF
            Overlap function QT
            dynamic susceptibility QTX4
            Mean-square displacements MSD
            non-Gaussian parameter NGP

        Parameters
        ----------
        atype: int, optional
            the atom type.
        outfile: bool, optional
            whether output file, default True.
        outpath: str, optional
            the path of output.

        Returns
        -------
        results: np.ndarray
            col: t, ISF, ISFX4, QT, QTX4, MSD, NPG

        Notes
        -----
        This functions only valid in isometric frames
        """
        # Initial
        TYPEfrac = len(self.atype[self.atype==atype])/self.natoms\
                if atype is not None else 1
        results = np.zeros((self.nframes-1, 7))
        cal_ISF = pd.DataFrame(np.zeros((self.nframes-1))[np.newaxis, :])
        cal_QT = pd.DataFrame(np.zeros(self.nframes-1)[np.newaxis, :])
        cal_MSD = pd.DataFrame(np.zeros(self.nframes-1)[np.newaxis, :])
        cal_NGP = pd.DataFrame(np.zeros(self.nframes-1)[np.newaxis, :])
        header = "t ISF ISFX4 QT QTX4 MSD NGP"
        # core code
        print("Computing iso now...")
        for i in range(1, self.nframes):
            datas = self.analysis(n=i, atype=atype)
            if i == 1:
                results[:, 0] = datas[:, 0]
            cal_ISF = pd.concat([cal_ISF,
                pd.DataFrame(datas[:, 1][np.newaxis, :])])
            cal_QT = pd.concat([cal_QT,
                pd.DataFrame(datas[:, 2][np.newaxis, :])])
            cal_MSD = pd.concat([cal_MSD,
                pd.DataFrame(datas[:, 3][np.newaxis, :])])
            cal_NGP = pd.concat([cal_NGP,
                pd.DataFrame(datas[:, 4][np.newaxis, :])])
        cal_ISF = cal_ISF.iloc[1:]
        cal_QT = cal_QT.iloc[1:]
        cal_MSD = cal_MSD.iloc[1:]
        cal_NGP = cal_NGP.iloc[1:]
        results[:, 1] = cal_ISF.mean()
        results[:, 2] = ((cal_ISF**2).mean() -\
                (cal_ISF.mean())**2) * self.natoms * TYPEfrac
        results[:, 3] = cal_QT.mean()
        results[:, 4] = ((cal_QT**2).mean() -\
                (cal_QT.mean())**2) * self.natoms *TYPEfrac
        results[:, 5] = cal_MSD.mean()
        results[:, 6] = cal_NGP.mean()
        if outfile:
            np.savetxt(os.path.join(outpath, 'dynamics.dat'),
                    results, fmt="%.6f", header=header)
        return results
    # }}}

if __name__ == "__main__":
    a = Dynamics(filepath='dumpnvt.lammpstrj')
    #results = a.analysis(atype=1, outfile=True)
    #results = a.analysis(outfile=True)
    #results = a.anaiso(atype=1)
    results = a.anaiso()
    #frames = Frame('test.data')
    #t, systems = frames.read_single()
    #print(systems.atoms.prop('c_Disp[1]'))
    #print(results)
