# Written by Metison Wood <wubqmail(at)163.com> under DWYW license

# Standard Python libraries
# {{{
from __future__ import (absolute_import, print_function, division,
        unicode_literals)
from copy import deepcopy
import sys
import os
import numpy as np
import atomman as am
from itertools import islice
# }}}
def dump_modify(systems=None, timestep=None):
# {{{
    """dump the file with timestep.

    Parameters
    ----------
    system: str
        system that need to be modified.
    timestep: int
        timestep

    Returns
    -------
    system: str
        file-object
    """
    # Check args
    if not isinstance(systems, str):
        raise ValueError('systems improper')
        sys.exit(1)
    if not isinstance(timestep, str):
        timestep = str(timestep)
    # Handling the system
    tmp_sys = systems.split('\n')
    tmp_sys[1] = timestep
    results = '\n'.join(tmp_sys)
    return results
# }}}

class Frame(object):
    """A class of get frames in atoms data."""

    def __init__(self, filepath=None):
    # {{{
        """Initilizes a Frame class.
        
        Parameters
        ----------
        filepath: file path contain atoms data.
        """
        # Check argvs
        if filepath is None or not os.path.isfile(filepath):
            raise ValueError('File is not exist')
            sys.exit(1)
        def blocks(files, size=100*1024*1024):
            while True:
                b = files.read(size)
                if not b: break
                yield b
        # Get info of file
        with open(filepath, 'r') as f:
            for a in islice(f, 3, 4):
                self.num_lines = int(a.strip()) + 9
                self.natoms = int(a.strip())
            f.seek(0)
            self.nlines = sum(bl.count("\n") for bl in blocks(f))
            self.nframes = int(self.nlines/(self.num_lines))
            f.seek(0)
            tmp_sys = ''
            for a in islice(f, 0, self.num_lines):
                tmp_sys += a
            systems = am.load('atom_dump', tmp_sys)
            self.prop = systems.atoms.prop()
        self.filename = os.path.basename(filepath)
        self.filepath = filepath
    # }}}
    def read_single(self, num=1):
        # {{{
        """
        Get a single frame.

        Parameters
        ----------
        num: the number of specified frame.

        Returns
        -------
        timestep(int), systems(atomman.System)
        """

        # Check argvs
        if not isinstance(num, int):
            raise ValueError('num of frame must be int')
            sys.exit(1)
        results = {}
        # Get the specified frame
        if num > self.nframes or num <= 0:
            raise ValueError('num isn\'t in range')
            sys.exit(1)
        else:
            with open(self.filepath, 'r') as f:
                start = (num-1) * self.num_lines if num == 1\
                        else (num-1) * (self.num_lines)-1
                stop = num * self.num_lines
                if num == 1:
                    for a in islice(f, start+1, start+2):
                        timestep = int(a.strip())
                else:
                    for a in islice(f, start+2, start+3):
                        timestep = int(a.strip())
                f.seek(0)
                tmp_sys = ''
                for a in islice(f, start, stop):
                    tmp_sys += a
                systems = am.load('atom_dump', tmp_sys)
        # Return
        return timestep, systems
        # }}}
    def read_multi(self, ranges=1, specs=None):
        # {{{
        """
        Get multi frames.

        Parameters
        ----------
        ranges: int or [start, stop] or [start, stop, step]
            the ranges or list of specified frame.
        specs: list, optional
            specified frames

        Returns
        -------
        timestep(list), systems(list)
        """

        # Check argvs
        if isinstance(ranges, int):
            ranges = [1, ranges]
        if ranges[1] == 1 and specs is None:
            return self.read_single()
        results = {}
        # Get the specified frame
        if len(ranges) == 2:
            fbins = range(ranges[0], ranges[1]+1)
        elif len(ranges) == 3:
            fbins = range(ranges[0], ranges[1]+1, ranges[2])
        elif len(ranges) > 3:
            raise ValueError('ranges larger than 3')
        if specs is not None:
            fbins = specs
        if max(fbins) >self.nframes:
            raise ValueError('Greater than maximum')
            sys.exit(1)
        timestep = []
        systems = []
        for i in fbins:
            tmp_step, tmp_sys = self.read_single(num=i)
            timestep.append(tmp_step)
            systems.append(tmp_sys)
        # Return
        return timestep, systems
        # }}}
    def dump_singe(self, num=1, scale=False, prop=True, outpath=''):
    # {{{
        """dump a single frame.

        Parameters
        ----------
        num: the number of specified frame.
        scale: absolute Cartesian values(False, default), True for relative
        prop: True for return all prop(default), False for not.
        outpath: the path of output file.

        Returns
        -------
        file: lammps dump style atom data.
        results: str
            file-object.
        """
        # Check params
        filename = input('Input the file name[frame[num].data]: ')
        filename = 'frame'+str(num)+'.data' if not filename else filename
        if not prop:
            pos = 'spos' if scale else 'pos'
            prop_name = ['atom_id', 'atype', pos]
        elif prop:
            prop_name = np.array(deepcopy(self.prop))
            if scale:
                prop_name[np.where(prop_name=='pos')[0][0]] = 'spos'
        timestep, systems = self.read_single(num=num)
        tmp_sys = systems.dump('atom_dump', scale=scale,
                prop_name=prop_name)
        results = dump_modify(systems=tmp_sys, timestep=timestep)
        with open(os.path.join(outpath, filename), 'w') as f:
            f.write(results)
        return results
    # }}}
    def dump_multi(self, ranges=1, specs=None, scale=False,
            prop=True, onefile=True, outpath=''):
    # {{{
        """dump multi frames.

        Parameters
        ----------
        ranges: int or [start, stop] or [start, stop, step]
            the ranges or list of specified frame.
        specs: list, optional
            specified frames
        scale: bool, optional
            absolute Cartesian values(False, default), True for relative
        prop: True for return all prop(default), False for not.
        onefile: bool, optional
            True for output frames to one file(default), False to separate.
        outpath: str, optional
            the path of output file.

        Returns
        -------
        multi-files if onefile is False, else onefile.
        results: str
            file-object.
        """
        # Check params
        prefix = input('Input the file prefix[frame]: ')
        prefix = 'frame' if not prefix else prefix
        if not prop:
            pos = 'spos' if scale else 'pos'
            prop_name = ['atom_id', 'atype', pos]
        elif prop:
            prop_name = np.array(deepcopy(self.prop))
            if scale:
                prop_name[np.where(prop_name=='pos')[0][0]] = 'spos'
        # Handling systems
        timestep, systems = self.read_multi(ranges=ranges, specs=specs)
        results = []
        for i in zip(timestep, systems):
            tmp_sys = i[1].dump('atom_dump', scale=scale,
                    prop_name=prop_name)
            results.append(dump_modify(systems=tmp_sys, timestep=i[0]))
        # Output
        if onefile:
            with open(os.path.join(outpath, prefix+'-all.data'), 'w') as f:
                for i in results:
                    f.write(i)
        if not onefile:
            for i in range(len(results)):
                with open(os.path.join(outpath,
                    prefix+str(i+1)+'.data'), 'w') as f:
                    f.write(results[i])
        return results
    # }}}

if __name__ == '__main__':
    frame = Frame('test.data')
    print(frame.filename, frame.nlines, frame.natoms,
            frame.nframes, frame.prop, frame.filepath)
    results = frame.read_single(20)
    print(results)
    #results = frame.read_multi(ranges=[1,11])
    #results = frame.dump_multi(specs=[1,2,5,6], prop=False, scale=True, onefile=True)
    results = frame.dump_singe(num=6, prop=False)
    #print(results)
