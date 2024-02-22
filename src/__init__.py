""" Project initialization and common objects. """

import logging
import os
from pathlib import Path
import re
import sys
import pickle

logging.basicConfig(
    stream=sys.stdout,
    format='[%(levelname)s][%(module)s] %(message)s',
    level=os.getenv('LOGLEVEL', 'info').upper()
)

AllData = None
workdir = None
cachedir = None
systems = None
keys = None
labels = None
ranges = None
design_array = None
data_list = None
data_list_val = None
exp_data_list = None
exp_cov = None
observables = None

def Initialize():
    global workdir
    workdir = Path(os.getenv('WORKDIR', '.'))

    global cachedir
    cachedir = workdir / 'cache'
    cachedir.mkdir(parents=True, exist_ok=True)

def parse_system(system):
    """
    Parse a system string into a pair of projectiles and a beam energy.

    """
    match = re.fullmatch('([A-Z]?[a-z])([A-Z]?[a-z])([0-9]+)', system)
    return match.group(1, 2), int(match.group(3))


class lazydict(dict):
    """
    A dict that populates itself on demand by calling a unary function.

    """
    def __init__(self, function, *args, **kwargs):
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def __missing__(self, key):
        self[key] = value = self.function(key, *self.args, **self.kwargs)
        return value

Initialize()
