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
    packages = [
        'mcvine.acc',
        'mcvine.acc.geometry',
        'mcvine.acc.components',
        'mcvine.acc.components.sources',
        'mcvine.acc.components.optics',
    ],
    package_dir = {
        'mcvine.acc': "acc",
        'mcvine.acc.geometry': "acc/geometry",
        'mcvine.acc.components': "acc/components",
        'mcvine.acc.components.sources': "acc/components/sources",
        'mcvine.acc.components.optics' : "acc/components/optics",
    },
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
