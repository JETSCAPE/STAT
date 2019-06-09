'''
Download experimental data (from hepdata and/or phenix) and write output files in jetscape-stat
format documented [location of format specification file goes here], and/or edit its format.

All relevant entries in a user-specified configuration file will be processed:
    python get_expdata.py -c expdata_config.yaml
'''
    
import os
import sys
import yaml
import argparse
import get_hepdata
import get_phenix
import edit_data

#----------------------------------------------------------------------------------------------
def get_expdata(configFile):

  # Specifiy Inputs from configFile
  with open(configFile, 'r') as stream:
    config = yaml.safe_load(stream)
  
  # Loop through configFile, and process each download entry with appropriate script
  for entry in config:
    if 'hepdata' in entry:
      print('Getting hepdata for entry "{}"...'.format(entry))
      get_hepdata.get_hepdata(config[entry])
    elif 'phenix' in entry:
      print('Getting phenix data for entry "{}"...'.format(entry))
      get_phenix.get_phenix(config[entry])

  # Loop through configFile, and process each extractError entry with edit_data.py
  for entry in config:
    if 'extractError' in entry:
      print('Editing data for entry "{}"...'.format(entry))
      edit_data.edit_data(config[entry])

#----------------------------------------------------------------------------------------------
if __name__ == '__main__':
  # Define arguments
  parser = argparse.ArgumentParser(description='Script to download experimental data')
  parser.add_argument('-c', '--configFile', action='store',
                      type=str, metavar='configFile',
                      default='expdata_config.yaml',
                      help='Path of config file for downloading data')
                      
  # Parse the arguments
  args = parser.parse_args()
  
  print('Downloading Data using configFile: \"{0}\"'.format(args.configFile))
  
  # If invalid configFile is given, exit
  if not os.path.exists(args.configFile):
    print('File \"{0}\" does not exist! Exiting!'.format(args.configFile))
    sys.exit(0)

  get_expdata(configFile = args.configFile)


