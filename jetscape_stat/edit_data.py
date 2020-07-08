'''
Edit or otherwise modify JETSCAPE formatted data files.
'''

import sys
import os
import math
import argparse
import numpy as np

# -------------------------- ARGUMENT PARSING ------------------------ #

def edit_data(configFileEntry = None):

  # Check that config file entry exists
  if not configFileEntry:
    print('configFileEntry is empty!')
    return

  # Load configuration variables
  dir = configFileEntry['dir']
  file = configFileEntry['file']
  error = configFileEntry['error']
  new = configFileEntry['new']
  suffix = configFileEntry['suffix']
  verbose = configFileEntry['verbose']

  # Check for file
  FileName = os.path.join(dir,file)
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
        ErrorIndex = Items.index(error)-2
      except:
        raise ValueError('Error Label {} not found.'.format(error))
      if (verbose):
        print('ErrorIndex = ',ErrorIndex)

  # Insert new header information, replace Label with EditRecord add New Label to end
  EditLine = '# EditRecord Extract '
  LabelLine = HeaderLines[LabelIndex].rstrip('\n')
  for Key, Error in new.items():
    EditLine += str(Key) + ':' + str(Error)
    LabelLine += ' ' + str(Key) + ',low ' + str(Key) + ',high'
  HeaderLines[LabelIndex] = EditLine + ' from ' + error + '\n'
  HeaderLines.append(LabelLine)

  NewHeader = ''.join(HeaderLines) + '\n'
  if (verbose):
    print(NewHeader)

  # Fetch data, convert fractional error to NewError and subtract in quadriture from OldError
  Data = np.loadtxt(FileName)
  NewData = ''
  for Entry in Data:
    Yval = Entry[2]
    ModError = Entry[ErrorIndex]
    # print(Yval,OldError)
    for Key, Error in new.items():
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
  if (verbose):
    print(NewData)

  # Rename old data file and replace with new one
  os.rename(FileName,FileName+suffix)
  with open(FileName,'w+') as f:
    f.write(NewHeader)
    f.write(NewData)

#----------------------------------------------------------------------------------------------
if __name__ == '__main__':
  edit_data()
