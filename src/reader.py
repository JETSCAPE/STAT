'''
Read data according to the JetScape 1.0 stat specification
'''

import numpy as np

import os
import pickle
from pathlib import Path

data_list = []
observables = []
systems = []
labels = []
keys = []
ranges = []
design_array = []
exp_data_list = []
exp_cov = []

def Initialize(picklefile):
    global workdir
    workdir = Path(os.getenv('WORKDIR', '.'))

    global AllData
    AllData = pickle.load((workdir / picklefile).open('rb'))

    #: Sets the collision systems for the entire project,
    #: where each system is a string of the form
    #: ``'<projectile 1><projectile 2><beam energy in GeV>'``,
    #: such as ``'PbPb2760'``, ``'AuAu200'``, ``'pPb5020'``.
    #: Even if the project uses only a single system,
    #: this should still be a list of one system string.
    global systems
    systems = AllData["systems"]

    #: Design attribute. This is a list of
    #: strings describing the inputs.
    #: The default is for the example data.
    global keys
    keys = AllData["keys"]

    #: Design attribute. This is a list of input
    #: labels in LaTeX for plotting.
    #: The default is for the example data.
    global labels
    labels = AllData["labels"]

    #: Design attribute. This is list of tuples of
    #: (min,max) for each design input.
    #: The default is for the example data.
    global ranges
    ranges = AllData["ranges"]

    #: Design array to use - should be a numpy array.
    #: Keep at None generate a Latin Hypercube with above (specified) range.
    #: Design array for example is commented under default.
    global design_array
    design_array = AllData["design"]

    #: Dictionary of the model output.
    #: Form MUST be data_list[system][observable][subobservable][{'Y': ,'x': }].
    #:     'Y' is an (n x p) numpy array of the output.
    #:
    #:     'x' is a (1 x p) numpy array of numeric index of columns of Y (if exists). In the example data, x is p_T.
    #: This MUST be changed from None - no built-in default exists. Uncomment the line below default for example.
    global data_list
    data_list = AllData["model"]

    #: Dictionary for the model validation output
    #: Must be the same for as the model output dictionary
    #data_list_val = pickle.load((cachedir / 'model/validation/data_dict_val.p').open('rb'))
    global data_list_val
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
    global exp_data_list
    exp_data_list = AllData["data"]

    #: Experimental covariance matrix.
    #: Set exp_cov = None to have the script estimate the covariance matrix.
    #: Example commented below default.
    global exp_cov
    exp_cov = AllData["cov"]


    #: Observables to emulate as a list of 2-tuples
    #: ``(obs, [list of subobs])``.
    global observables
    observables = AllData["observables"]

def ReadDesign(FileName):
    # This is the output object
    Result = {}
    Version = ''

    Result["FileName"] = FileName

    # First read all the header information
    for Line in open(FileName):
        Items = Line.split()
        if (len(Items) < 2): continue
        if Items[0] != '#': continue

        if(Items[1] == 'Version'):
            Version = Items[2]
        elif(Items[1] == 'Parameter'):
            Result["Parameter"] = Items[2:]

    if(Version != '1.0'):
        raise AssertionError('Bad file version number while reading design points')

    # Then read the actual design parameters
    Result["Design"] = np.loadtxt(FileName)
    return Result

def ReadData(FileName):
    # Initialize objects
    Result = {}
    Version = ''

    Result["FileName"] = FileName

    # First read all the header information
    for Line in open(FileName):
        Items = Line.split()
        if (len(Items) < 2): continue
        if Items[0] != '#': continue

        if(Items[1] == 'Version'):
            Version = Items[2]
        elif(Items[1] == 'DOI'):
            Result["DOI"] = Items[2:]
        elif(Items[1] == 'Source'):
            Result["Source"] = Items[2:]
        elif(Items[1] == 'System'):
            Result["System"] = Items[2]
        elif(Items[1] == 'Centrality'):
            Result["Centrality"] = Items[2:4]
        elif(Items[1] == 'XY'):
            Result["XY"] = Items[2:4]
        elif(Items[1] == 'Label'):
            Result["Label"] = Items[2:]

    if(Version != '1.0'):
        raise AssertionError('Bad file version number while reading design points')

    XMode = ''
    if(Result["Label"][0:4] == ['x', 'y', 'stat,low', 'stat,high']):
        XMode = 'x'
    elif(Result["Label"][0:5] == ['xmin', 'xmax', 'y', 'stat,low', 'stat,high']):
        XMode = 'xminmax'
    else:
        raise AssertionError('Invalid list of initial columns!  Should be ("x", "y", "stat,low", "stat,high"), or ("xmin", "xmax", "y", "stat,low", "stat,high")')

    # Then read the actual data
    RawData = np.loadtxt(FileName)

    Result["Data"] = {}
    if(XMode == 'x'):
        Result["Data"]["x"] = RawData[:, 0]
        Result["Data"]["y"] = RawData[:, 1]
        Result["Data"]["yerr"] = {}
        Result["Data"]["yerr"]["stat"] = RawData[:, 2:4]
        Result["Data"]["yerr"]["sys"] = RawData[:, 4:]
        Result["SysLabel"] = Result["Label"][4:]
    elif(XMode == 'xminmax'):
        Result["Data"]["x"] = (RawData[:, 0] + RawData[:, 1]) / 2
        Result["Data"]["xerr"] = (RawData[:, 1] - RawData[:, 0]) / 2
        Result["Data"]["y"] = RawData[:, 2]
        Result["Data"]["yerr"] = {}
        Result["Data"]["yerr"]["stat"] = RawData[:, 3:5]
        Result["Data"]["yerr"]["sys"] = RawData[:, 5:]
        Result["SysLabel"] = Result["Label"][5:]

    return Result

def ReadCovariance(FileName):
    # Initialize objects
    Result = {}
    Version = ''

    Result["FileName"] = FileName

    # First read all the header information
    for Line in open(FileName):
        Items = Line.split()
        if (len(Items) < 2): continue
        if Items[0] != '#': continue

        if(Items[1] == 'Version'):
            Version = Items[2]
        elif(Items[1] == 'Data1'):
            Result["Data1"] = Items[2]
        elif(Items[1] == 'Data2'):
            Result["Data2"] = Items[2]

    if(Version != '1.0'):
        raise AssertionError('Bad file version number while reading design points')

    # Then read the actual covariance matrix
    Result["Matrix"] = np.loadtxt(FileName)
    return Result

def ReadPrediction(FileName):
    # Initialize objects
    Result = {}
    Version = ''

    Result["FileName"] = FileName

    # First read all the header information
    for Line in open(FileName):
        Items = Line.split()
        if (len(Items) < 2): continue
        if Items[0] != '#': continue

        if(Items[1] == 'Version'):
            Version = Items[2]
        elif(Items[1] == 'Data'):
            Result["Data"] = Items[2]
        elif(Items[1] == 'Design'):
            Result["Design"] = Items[2]

    if(Version != '1.0'):
        raise AssertionError('Bad file version number while reading design points')

    # Then read the actual model predictions
    Result["Prediction"] = np.loadtxt(FileName).T
    return Result

def InitializeCovariance(data):
    Result = {}
    for system, content in data.items():
        Result[system] = {}
        Combination = []
        for obs in content:
            for subobs in content[obs]:
                Combination.append((obs, subobs))
        for item1 in Combination:
            Result[system][item1] = {}
            for item2 in Combination:
                Result[system][item1][item2] = None
    return Result

"""
def EstimateCovariance(DataX, DataY, SysLength = {}, SysStrength = {}, ScaleX = True, IgnoreMissing = False)
    DataX          data used for the first index of the output matrix
    DataY          data used for the second index of the output matrix
    SysLength      correlation length, source by source.  One can specify a "default" for sources not listed
                   Negative value indicates that the source is treated as uncorrelated.
                   If "default" is not specified, it is assumed to be -1 (ie., uncorrelated)
    SysStrength    correlation strength, source by source.  Again there is a "default" one can specify
                   If "default" is not specified, it is assumed to be 1.0
    ScaleX         whether correlation length is in units of "range of x axis", or units of "x"
    IgnoreMissing  set to true to ignore sources not explicitly listed in SysLength dictionary

Estimates covariance matrix for a block

The function returns a 2D matrix with size (DataX points, DataY points)
If we pass the same thing to DataX and DataY, we can calculate the diagonal blocks.

The formula used to populate the matrix is
   Strength * Sigma_x * Sigma_y * exp(pow(-|x - y| / Length, 1.9))
for each source, and the summed up for all considered sources

Currently only symmetric uncertainties are supported.  If a source ends with ",low" it is ignored and assumed to be covered by a corresponding one ending with ",high"
"""
def EstimateCovariance(DataX, DataY, SysLength = {}, SysStrength = {}, ScaleX = True, IgnoreMissing = False):
    # Number of entries in each data
    NX = len(DataX["Data"]["x"])
    NY = len(DataY["Data"]["x"])

    # Scale of x
    # If ScaleX is true, the correlation length is in units of "x range"
    #    otherwise it's the same unit as "x"
    DX = 1
    DY = 1
    if ScaleX == True:
        DX = 1 / (max(DataX["Data"]["x"]) - min(DataX["Data"]["x"]))
        DY = 1 / (max(DataY["Data"]["x"]) - min(DataY["Data"]["x"]))

    # Initialize empty matrix
    Matrix = np.zeros([NX, NY])

    # Add statistical uncertainty here, if this is diagonal block
    DiagonalBlock = False
    if DataX["FileName"] == DataY["FileName"]:
        DiagonalBlock = True
        for i in range(0, NX):
            Matrix[i, i] = Matrix[i, i] + DataX["Data"]["yerr"]["stat"][i][0]**2

    # Add a default behavior if not supplied already
    if "default" not in SysLength:
        SysLength["default"] = -1
    if "default" not in SysStrength:
        SysStrength["default"] = 1.0

    # Now loop over systematic source in dataX, and check if the same thing exist in dataY
    for Source in DataX["SysLabel"]:
        if ",low" in Source:
            continue
        if Source not in DataY["SysLabel"]:
            continue

        if (IgnoreMissing == True) and (Source not in SysLength):
            continue

        IX = DataX["SysLabel"].index(Source)
        IY = DataY["SysLabel"].index(Source)

        Length = SysLength.get(Source, SysLength["default"])
        Strength = SysStrength.get(Source, SysStrength["default"])

        for x in range(0, NX):
            for y in range(0, NY):
                Factor = 0
                if(Length > 0):   # Correlated
                    Diff = DataX["Data"]["x"][x] * DX - DataY["Data"]["x"][y] * DY
                    Factor = np.exp(-np.power(np.absolute(Diff) / Length, 1.9));
                else:             # Non-correlated
                    if DiagonalBlock == True:
                        Factor = (x == y)
                Factor = Factor * Strength
                Matrix[x, y] = Matrix[x, y] + DataX["Data"]["yerr"]["sys"][x][IX] * DataY["Data"]["yerr"]["sys"][y][IY] * Factor

    return Matrix


class TuneData:
    def __init__(self, picklefile):
        self.workdir = Path(os.getenv('WORKDIR', '.'))
        self.AllData = pickle.load((self.workdir / picklefile).open('rb'))
        self.systems = self.AllData["systems"]
        self.keys = self.AllData["keys"]
        self.labels = self.AllData["labels"]
        self.ranges = self.AllData["ranges"]
        self.design_array = self.AllData["design"]
        self.data_list = self.AllData["model"]
        self.data_list_val = None
        self.exp_data_list = self.AllData["data"]
        self.exp_cov = self.AllData["cov"]
        self.observables = self.AllData["observables"]

    def dumpData(self):
        return self.workdir, self.AllData, self.systems, self.keys, self.labels, self.ranges, self.design_array, self.data_list, self.data_list_val, self.exp_data_list, self.exp_cov, self.observables