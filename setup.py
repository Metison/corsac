import os
from setuptools import setup, find_packages

def getversion():
    """Read version from VERSION file."""
    with open(os.path.join(os.path.dirname(__file__),
        'mdAna', 'VERSION')) as f:
        return f.read().strip()

def getreadme():
    """Fetches description from README.rst file."""
    with open('README.rst') as readme_file:
        return readme_file.read()

setup(name = 'mdAna',
      version = getversion(),
      description = 'Molecular Dynamic Analysis Toolkit',
      long_description = getreadme(),
      #ext_modules = extensions,
      classifiers=[
          'Development Status :: 1 - Alpha',
          'Intended Audience :: Science/Research',
          'Natural Language :: English',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.7',
          'Topic :: Scientific/Engineering :: Physics'
          ],
      keywords = [
          'atom',
          'atomic',
          'atomistic',
          'molecular dynamic',
          'interatomic',
          'analysis'
          ],
      url = 'http://github.com/Metison/corsac',
      author = 'Metison Wood',
      author_email = 'wubqmail@163.com',
      packages = find_packages(),
      install_requires = [
          'xmltodict',
          'DataModelDict',
          'numpy',
          'matplotlib',
          'scipy',
          'pandas',
          'numericalunits',
          'atomman>=1.2.0',
          'requests',
          ],
      package_data={'': ['*']},
      zip_safe = False
      )
