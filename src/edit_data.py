import sys
import os
import math
import argparse
import numpy as np

# -------------------------- ARGUMENT PARSING ------------------------ #

prog_description = 'Edit or otherwise modify JETSCAPE formatted data files.'
prog_epilogue    = 'Original files saved with .orig suffix'

# Initialize argument parser
parser = argparse.ArgumentParser(
    description=prog_description,
    usage='%(prog)s [options]',
    epilog=prog_epilogue,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Add arguments
parser.add_argument('-d', '--dir', type=str,
                    default='../input/MATTERLBT1',
                    help='Data file directory')
parser.add_argument('-f', '--file', type=str,
                    default='Data_ATLAS_PbPb2760_RAACharged_30to40_2015.dat',
                    help='Data file name')
parser.add_argument('-e', '--error',type=str,
                    default='sys_lo',
                    help='Label of error from which portions will be extracted')
parser.add_argument('-n', '--new',type=dict,
                    default={'sys,TAA':0.04,'sys,lumi':0.03},
                    help='Dictionary for new error labels and values.')
parser.add_argument('-s', '--suffix',type=str,
                    default='.orig',
                    help='Suffix to append to original file.')
parser.add_argument('-v', '--verbose',action='store_true', help='Standard verbose option.')

# Parse arguments
args = parser.parse_args()

# Check for file
FileName = os.path.join(args.dir,args.file)
if not os.path.isfile(FileName):
        raise ValueError('File {} not found.'.format(FileName))

# Read header and rebuild as list of lines to allow for easy insertion
HeaderLines = []
for Line in open(FileName):
    Items = Line.split()
    if (len(Items) < 2): continue
    if Items[0] != '#':
        continue

    HeaderLines.append(Line)
    if(Items[1] == 'Version'):
        Version = Items[2]
        if(Version != '1.0'):
            raise AssertionError('Bad file version number while reading data')
        
    if(Items[1] == 'Label'):
        LabelIndex = len(HeaderLines)-1
        try:
            ErrorIndex = Items.index(args.error)-2
        except:
            raise ValueError('Error Label {} not found.'.format(args.error))           
        if (args.verbose):
            print('ErrorIndex = ',ErrorIndex)

# Insert new header information, replace Label with EditRecord add New Label to end
EditLine = '# EditRecord Extract '
LabelLine = HeaderLines[LabelIndex].rstrip('\n')
for Key, Error in args.new.items():
    EditLine += str(Key) + ':' + str(Error)
    LabelLine += ' ' + str(Key) + '_lo ' + str(Key) + '_hi'
HeaderLines[LabelIndex] = EditLine + ' from ' + args.error + '\n'
HeaderLines.append(LabelLine)

NewHeader = ''.join(HeaderLines) + '\n'
if (args.verbose):
    print(NewHeader)

# Fetch data, convert fractional error to NewError and subtract in quadriture from OldError
Data = np.loadtxt(FileName)
NewData = ''
for Entry in Data:
    Yval = Entry[2]
    ModError = Entry[ErrorIndex]
#    print(Yval,OldError)
    for Key, Error in args.new.items():
        # Use string format to truncate
        NewError = float('{0:.02e}'.format(Yval*Error))
        NewEntry = np.array([NewError,NewError])
        if (abs(ModError)>abs(NewError)):
            ModError = (ModError**2-NewError**2)**0.5
        else:
            print('NewError: ',NewError,' > ModError: ',ModError)
            raise AssertionError('New Error too large to subtract')
        Entry[ErrorIndex] = ModError
        Entry[ErrorIndex+1] = ModError
        Entry = np.append(Entry,NewEntry)
    for e in Entry:
        NewData += ' {0:.02e}'.format(e)
    NewData += '\n'
if (args.verbose):
    print(NewData)

# Rename old data file and replace with new one
os.rename(FileName,FileName+args.suffix)
with open(FileName,'w+') as f:
    f.write(NewHeader)
    f.write(NewData)
