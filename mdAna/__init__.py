"""
Attributes
----------
rootdir: str
    The absolute path to the mdAna package's root directory used to locate
    contained data files.
"""
# Standard Python libraries
from __future__ import (absolute_import, print_function, division,
        unicode_literals)
import os

# Define rootdir
rootdir = os.path.dirname(os.path.abspath(__file__))

# Read version from VERSION file
with open(os.path.join(os.path.dirname(__file__), 'VERSION')) as version_file:
    __version__ = version_file.read().strip()

# mdAna imports
from . import compatibility
from . import tools
from .atoms import *
from .atoms import __all__ as atoms_all
from .core import *
from .core import __all__ as core_all

__all__ = ['__version__', 'rootdir'] + atoms_all + core_all
__all__.sort()
