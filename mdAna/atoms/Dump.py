# Written by Metison Wood <wubqmail(at)163.com> under DWYW license

# Standard Python libraries
# {{{
from __future__ import (absolute_import, print_function, division,
        unicode_literals)
import os
import sys
import atomman as am
import numpy as np
from copy import deepcopy
from .Frame import dump_modify
from .Frame import Frame
# }}}

class Dump(object):
    """A class for dump atomman.Systems into files."""
    def __init__(self, systems=None, timesteps=None):
    # {{{
        """Initial a dump class.
        
        Parameters
        ----------
        systems: atomman.System or a list of atomman.Systems.
        timesteps: int, or a list of ints
        """
        # Check args
        if isinstance(systems, am.System):
            systems = [systems,]
        if isinstance(timesteps, int):
            timesteps = [timesteps, ]
        self.systems = systems
        self.timesteps = timesteps
        self.prop = systems[0].atoms.prop()
    # }}}
    def dump(self, scale=False, prop=True, onefile=True, outpath=''):
    # {{{
        """dump atomman.Systems into files.

        Parameters
        ----------
        scale: bool, optional
            absolute Cartesian values(False, default), True for relative
        prop: True for return all prop(default), False for not.
        onefile: bool, optional
            True for output frames to one file(default), False to separate.
        outpath: str, optional
            the path of output file.

        Returns
        -------
        multi-files if onefile is False, else onfile.
        results:
            file-object contain systems(str).
        """
        # Check params
        prefix = 'frame'
        if not prop:
            pos = 'spos' if scale else 'pos'
            prop_name = ['atom_id', 'atype', pos]
        elif prop:
            prop_name = np.array(deepcopy(self.prop))
            if scale:
                prop_name[np.where(prop_name=='pos')[0][0]] = 'spos'
        # Handling systems
        results = []
        for i in zip(self.timesteps, self.systems):
            tmp_sys = i[1].dump('atom_dump', scale=scale,
                    prop_name=prop_name, float_format="%.8g")
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

if __name__ == "__main__":
    frames = Frame('msd1.lammpstrj')
    timesteps, systems = frames.read_multi(ranges=frames.nframes)
    dumps = Dump(systems=systems, timesteps=timesteps)
    results = dumps.dump(scale=True, prop=False, onefile=False)
