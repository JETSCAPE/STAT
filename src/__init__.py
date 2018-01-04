""" Project initialization and common objects. """

import logging
import os
from pathlib import Path
import re
import sys
import pickle
from sklearn.externals import joblib

logging.basicConfig(
    stream=sys.stdout,
    format='[%(levelname)s][%(module)s] %(message)s',
    level=os.getenv('LOGLEVEL', 'info').upper()
)

workdir = Path(os.getenv('WORKDIR', '.'))

cachedir = workdir / 'cache'
cachedir.mkdir(parents=True, exist_ok=True)

#: Sets the collision systems for the entire project,
#: where each system is a string of the form
#: ``'<projectile 1><projectile 2><beam energy in GeV>'``,
#: such as ``'PbPb2760'``, ``'AuAu200'``, ``'pPb5020'``.
#: Even if the project uses only a single system,
#: this should still be a list of one system string.
systems = ['PbPb5020']


#: Design attribute. This is a list of 
#: strings describing the inputs.
#: The default is for the example data.
keys = ['lambda_jet','alpha_s'] #labels in words

#: Design attribute. This is a list of input
#: labels in LaTeX for plotting.
#: The default is for the example data. 
labels = [r'\Lambda_{jet}',r'\alpha_s}'] #labels in LaTeX

#: Design attribute. This is list of tuples of 
#: (min,max) for each design input.
#: The default is for the example data.
ranges = [(0.01,0.3),(0.05,0.35)]

#: Design array to use - should be a numpy array.
#: Keep at None generate a Latin Hypercube with above (specified) range.
#: Design array for example is commented under default.
design_array = None
#design_array = pickle.load((cachedir / 'lhs/design_s.p').open('rb'))

#: Dictionary of the model output.
#: Form MUST be data_list[system][observable][subobservable][{'Y': ,'x': }].
#:     'Y' is an (n x p) numpy array of the output.
#:
#:     'x' is a (1 x p) numpy array of numeric index of columns of Y (if exists). In the example data, x is p_T. 
#: This MUST be changed from None - no built-in default exists. Uncomment the line below default for example.
data_list = None
#data_list = pickle.load((cachedir / 'model/main/full_data_dict.p').open('rb'))

#: Dictionary for the model validation output
#: Must be the same for as the model output dictionary
#data_list_val = pickle.load((cachedir / 'model/validation/data_dict_val.p').open('rb'))
data_list_val = None

#: Dictionary of the experimental data.
#: Form MUST be exp_data_list[system][observable][subobservable][{'y':,'x':,'yerr':{'stat':,'sys'}}].
#:      'y' is a (1 x p) numpy array of experimental data.
#:
#:      'x' is a (1 x p) numpy array of numeric index of columns of Y (if exists). In the example data, x is p_T.
#:
#:      'yerr' is a dictionary with keys 'stat' and 'sys'.
#:
#:      'stat' is a (1 x p) array of statistical errors.
#:
#:      'sys' is a (1 x p) array of systematic errors.
#: This MUST be changed from None - no built-in default exists. Uncomment the line below default for example.
exp_data_list = None
#exp_data_list = pickle.load((cachedir / 'hepdata/data_list_exp.p').open('rb'))

#: Experimental covariance matrix.
#: Set exp_cov = None to have the script estimate the covariance matrix.
#: Example commented below default.
exp_cov = None
#exp_cov = pickle.load((cachedir / 'hepdata/cov_exp_pbpb5020_30_50.p').open('rb'))


#: Observables to emulate as a list of 2-tuples
#: ``(obs, [list of subobs])``.
observables = [('R_AA',[None])]

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
