# Written by Metison Wood <wubqmail(at)163.com> under DWYW license
# To analyze the csro of atomman.System

# Python standard libraries
# {{{
import os, sys
import numpy as np
import atomman as am
import subprocess as sp
# }}}

def CSRO(systems=None, symbols=None, face=0.5, minneib=0,
        peratoms=False):
    # {{{
        """Analyzing the csro using dumpana.
        
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
        peratoms: bool, optional
            whether output the peratom csro.

        Returns
        -------
        results: a list of float
            csro11, csro12, csro21, csro22
        systems: atomman.System
            only if the peratoms is True, default(False)
        """
        # Check args
        print("Analyzing the CSRO now...")
        symbols = symbols if isinstance(symbols, str) else " ".join(symbols)
        systems.dump('atom_dump', f='.csro.data',
                prop_name=['atom_id', 'atype', 'pos'])
        if not peratoms:
            with open('.dumpana', 'w') as f:
                f.write(symbols+'\n2\n1\n1\n'+str(face)
                        +'\n'+str(minneib)+'\ncsro.dat')
        else:
            with open('.dumpana', 'w') as f:
                f.write(symbols+'\n2\n1\n2\n'+str(face)
                        +'\n'+str(minneib)+'\ncsro.dat')
        cmdline = 'dumpana -1 .csro.data < .dumpana'
        sp.run(cmdline, shell=True, stdout=sp.DEVNULL)
        os.remove('.dumpana')
        os.remove('.csro.data')
        if not peratoms:
            with open('csro.dat', 'r') as f:
                for i in range(7):
                    f.readline()
                data1 = f.readline().split(':')[1].split()
                data2 = f.readline().split(':')[1].split()
                csro11, csro12 = float(data1[0]), float(data1[1])
                csro21, csro22 = float(data2[0]), float(data2[1])
            os.remove('csro.dat')
            return [csro11, csro12, csro21, csro22]
        else:
            tmp_data = np.genfromtxt('csro.dat')[:, 5:]
            systems.atoms.csro1 = tmp_data[:, 0]
            systems.atoms.csro2 = tmp_data[:, 1]
            os.remove('csro.dat')
            return systems
    # }}}

if __name__ == "__main__":
    #outfile: bool, optional
    #    True for saving files.
    #outpath: str, optional
    #    the path for saving.
    import mdAna as ma
    frames = ma.Frame('atoms.data')
    timesteps, systems = frames.read_single()
    results = CSRO(systems=systems, symbols=['Cu', 'Ag'], peratoms=True)
    print(results.atoms.csro1)
    print(results.atoms.csro2)
