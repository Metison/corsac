# Written by Metison Wood <wubqmail(at)163.com> under DWYW license
# -*- coding:utf-8 -*-
__author__ = "Metison Wood"

# Standard Python libraries
# {{{
import os, sys
import numpy  as np
import pandas as pd
from math import pi
from .StaticStructureFactor import wavevector3d
from ..atoms import Frame
# }}}

class TrajectoryDynamic(object):
    """A class for analyzing the trajectory dynamics.
    Note: The atomic data of imagebox is needed!
          Only 3-dimension is supported!"""
    def __init__(self, frames=None, qmax=2.5, QTdr=1.0, dt=0.001,
            seltype=None, postype="image"):
    # {{{
        """Initial a class of trajectory dynamics.

        Parameters
        ----------
        frames: frames object
            the frames object from mdAna.
        qmax: float or list of float(partial)
            the wavenumber corresponding to the first peak of structure factor.
        QTdr: float, optional
            the cutoff for the overlap function, 1.0(EAM) and 0.3(LJ).
        dt: float, optional
            the timestep of MD simulations, default: ps
        seltype: int, optional
            the selected atom type of analysis.
        postype: str, optional
            the type of position:
            image(ix, iy, iz);
            disp(c_Disp[1], c_Disp[2], c_Disp[3], c_Disp[4])
        """
        # Input paras
        self.qmax = qmax
        self.QTdr = QTdr
        self.dt = dt
        # System paras
        self.postype = postype
        self.frames = frames
        timesteps, systems = frames.read_single()
        self.atype = systems.atoms.atype
        self.nframes = frames.nframes
        self.natoms = systems.atoms.natoms
        b = systems.box
        self.boxsize = [b.lx, b.ly, b.lz]
        self.boxbounds = [b.xlo, b.xhi, b.ylo, b.yhi, b.zlo, b.zhi]
        self.atypes = systems.atypes
        # Get the selected atom type 
        if seltype is None:
            self.TYPESET = np.where(np.array(self.atype) == 1, 1, 1)
            self.TYPEfrac = 1
        else:
            self.TYPESET = np.where(np.array(self.atype) == seltype, 1, 0)
            self.TYPEfrac = len(self.atype[self.atype == seltype])/self.natoms
    # }}}
    def single_disp(self, numA=1, numB=2):
    # {{{
        """Calculated the pos of two frames.

        Parameters
        ----------
        numA: int
            the number of frameA(ref).
        numB: int
            the nubmer of frameB.

        Returns
        -------
        RII: numpy
            DispX, DispY, DispZ, DispAll
        """
        tA, frameA = self.frames.read_single(num = numA)
        tB, frameB = self.frames.read_single(num = numB)
        atomsA = frameA.atoms
        atomsB = frameB.atoms
        # Check the position type
        if self.postype == "image":
            RII = np.zeros((self.natoms, 3))
            sA = atomsA.pos + atomsA.boximage
            sB = atomsB.pos + atomsB.boximage
            RII = sB - sA
        elif self.postype == "disp":
            dispcol = ['c_Disp[1]', 'c_Disp[2]', 'c_Disp[3]']
            RII = np.zeros((self.natoms, 3))
            for i in range(len(dispcol)):
                RII[:, i] = atomsA.prop(dispcol[i]) - atomsB.prop(dispcol[i])
        else:
            print("The style of position is not supoorted!")
            sys.exit(1)
        tAB = (int(tB) - int(tA)) * self.dt
        return round(tAB, 4), RII
    # }}}
    def multi_disp(self, numRef=1, numSel=None):
    # {{{
        """Obtained the sum displace info from multi frames.

        Parameters
        ----------
        numRef: int, optional
            the ref frames to calculate the displacements.
        numSel: int, list, optional
            the selected frames to calculate the displacements.

        Returns
        -------
        results: pandas.DataFrame
            five cols: times, sumSD, sumCosQD, sumFD, WD
        """
        # Check the Selected frames.
        if numSel is None:
            selFrames = range(numRef+1, self.nframes+1)
        elif isinstance(numSel, int) and numSel > numRef:
            selFrames = range(numRef+1, numSel+1)
        elif isinstance(numSel, list):
            selFrames = numSel
        else:
            print("Multi-frames is needed!")
            sys.exit(1)
        results = np.zeros((len(selFrames), 5))
        for i in range(len(selFrames)):
            results[i, 0], RII = self.single_disp(numA=numRef,
                    numB=selFrames[i])
            results[i, 1] = (np.square(RII).sum(axis=1)\
                    * self.TYPESET).sum()
            results[i, 2] = (np.cos(RII * self.qmax).mean(axis=1)\
                    * self.TYPESET).sum()
            results[i, 3] = (np.square(np.square(RII).sum(axis=1))\
                    * self.TYPESET).sum()
            results[i, 4] = ((np.sqrt(np.square(RII).sum(axis=1))\
                    <= self.QTdr) * self.TYPESET).sum()
        columns = ['times', 'sumSD', 'sumCosQD', 'sumFD', 'WD']
        results = pd.DataFrame(results, columns=columns)
        return results
    # }}}
    def dynamics(self, numRef=1, numSel=None):
    # {{{
        """Analyze the MSD, ISF, NGP, QT.

        Parameters
        ----------
        numRef: int, optional
            the ref frames to calculate the displacements.
        numSel: int, list, optional
            the selected frames to calculate the displacements.

        Returns
        -------
        results: pandas.DataFrame
            five cols: times, MSD, ISF, NGP, WD
        """
        sumDisp = self.multi_disp(numRef=numRef, numSel=numSel)
        Natoms = (self.natoms * self.TYPEfrac)
        cal_MSD = sumDisp['sumSD'] / Natoms
        cal_ISF = sumDisp['sumCosQD'] / Natoms
        cal_NGP = 3/5 * sumDisp['sumFD'] / Natoms /\
                np.square((sumDisp['sumSD'] / Natoms)) -1
        cal_QT = sumDisp['WD'] / Natoms
        columns = ['times', 'MSD', 'ISF', 'NGP', 'QT']
        results = pd.concat([sumDisp['times'], cal_MSD,
            cal_ISF, cal_NGP, cal_QT], axis=1)
        results.columns = columns
        return results
    # }}}
    def avedynamics(self, numRef=1, numSel=None):
    # {{{
        """Analyze the averaged MSD, ISF, NGP, QT.
        Note: only for isometric.

        Parameters
        ----------
        numRef: int, optional
            the ref frames to calculate the displacements.
        numSel: int, optional
            the selected frames to calculate the displacements.

        Returns
        -------
        results: pandas.DataFrame
            five cols: times, MSD, ISF, NGP, WD
        """
        print("Calculate the dynamical properties now...")
        # Handling the numSel
        if numSel is None:
            selFramesNum= self.nframes - 1
            numSel = self.nframes
        elif isinstance(numSel, int) and numSel > numRef:
            selFramesNum= numSel - numRef
        else:
            print("iso-Multi-frames is needed!")
            sys.exit(1)
        results = np.zeros((selFramesNum, 7))
        ave_MSD = pd.DataFrame(np.zeros(selFramesNum))
        ave_ISF = pd.DataFrame(np.zeros((selFramesNum)))
        ave_NGP = pd.DataFrame(np.zeros(selFramesNum))
        ave_QT = pd.DataFrame(np.zeros(selFramesNum))
        columns = ["times", "MSD", "ISF", "ISFX4", "NGP", "QT", "QTX4"]
        Natoms = (self.natoms * self.TYPEfrac)
        for i in range(numRef, numSel):
            sumDisp = self.multi_disp(numRef=i, numSel=numSel)
            if i == 1:
                results[:, 0] = sumDisp['times']
            ave_MSD = pd.concat([ave_MSD, sumDisp['sumSD']], axis=1)
            ave_ISF = pd.concat([ave_ISF, sumDisp['sumCosQD']], axis=1)
            ave_NGP = pd.concat([ave_NGP, sumDisp['sumFD']], axis=1)
            ave_QT = pd.concat([ave_QT, sumDisp['WD']], axis=1)
        ave_MSD = ave_MSD.iloc[:, 1:]
        ave_ISF = ave_ISF.iloc[:, 1:]
        ave_NGP = ave_NGP.iloc[:, 1:]
        ave_QT = ave_QT.iloc[:, 1:]
        results[:, 1] = ave_MSD.mean(axis=1) / Natoms
        results[:, 2] = ave_ISF.mean(axis=1) / Natoms
        results[:, 3] = ((ave_ISF**2).mean(axis=1)\
                - (ave_ISF.mean(axis=1))**2) / Natoms
        results[:, 4] = 3.0/5.0 *  (ave_NGP.mean(axis=1)/Natoms)\
                / np.square(results[:, 1]) - 1.0
        results[:, 5] = ave_QT.mean(axis=1) / Natoms
        results[:, 6] = ((ave_QT**2).mean(axis=1)\
                - (ave_QT.mean(axis=1))**2) / Natoms
        results = pd.DataFrame(results, index=None, columns=columns)
        return results
    # }}}
    def trajectory(self, numRef=1, numSel=None):
    # {{{
        """Analyze the trajectory of per atom.

        Parameters
        ----------
        numRef: int, optional
            the ref frames to calculate the displacements.
        numSel: int, list, optional
            the selected frames to calculate the displacements.

        Returns
        -------
        times, RII
        """
        print("Calculate the trajectory now...")
        # Check the Selected frames.
        if numSel is None:
            selFrames = range(numRef, self.nframes+1)
        elif isinstance(numSel, int) and numSel > numRef:
            selFrames = range(numRef, numSel+1)
        elif isinstance(numSel, list):
            selFrames = [numRef] + numSel
        else:
            print("Multi-frames is needed!")
            sys.exit(1)
        Rpath = np.zeros((self.natoms, 4))
        times = 0
        for i in range(len(selFrames)-1):
            t, RII = self.single_disp(numA=selFrames[i],
                    numB=selFrames[i+1])
            times += t
            displace = np.sqrt(np.square(RII).sum(axis=1))
            RII = np.vstack((np.abs(RII).T, displace)).T
            Rpath += RII
        ts, systems = self.frames.read_single()
        systems.atoms.Tx = Rpath[:, 0]
        systems.atoms.Ty = Rpath[:, 1]
        systems.atoms.Tz = Rpath[:, 2]
        systems.atoms.Tall = Rpath[:, 3]
        return times, systems
    # }}}

if __name__ == "__main__":
    #frames = Frame(filepath='./msd1.lammpstrj')
    frames = Frame(filepath='./dumpnvt.lammpstrj')
    dynamics = TrajectoryDynamic(frames=frames, postype='disp')
    #t, rii = dynamics.single_disp()
    #results = dynamics.multi_disp(numSel=5)
    #results = dynamics.dynamics(numSel=5)
    #results = dynamics.avedynamics(numSel=5)
    times, systems= dynamics.trajectory()
    print(times, systems.atoms.Tx)
    #frames = Frame(filepath=filepath)
