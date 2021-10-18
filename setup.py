#!/usr/bin/env python

import os
from setuptools import setup, find_packages

here = os.path.dirname(__file__)
version_ns = {}
# with open(os.path.join(here, 'acc', '_version.py')) as f:
#     exec(f.read(), {}, version_ns)

# define distribution
setup(
    name = "mcvine.acc",
    # version = version_ns['__version__'],
    #packages = find_packages(".", exclude=['tests', 'notebooks', 'jenkins']),
    packages = ['mcvine.acc'],
    package_dir = {'mcvine.acc': "acc"},
    data_files = [],
    install_requires = [
    ],
    dependency_links = [
    ],
    author = "MCViNE team",
    description = "accelerated mcvine",
    license = 'BSD',
    keywords = "instrument, neutron",
    url = "https://github.com/mcvine/acc",
    # download_url = '',
)

# End of file
