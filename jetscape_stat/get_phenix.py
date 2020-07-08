"""
Fetch data from phenix url and write output files in jetscape-stat format documented
  [location of format specification file goes here]
"""

from urllib.request import urlopen
import re
import ssl
import yaml
import os
import argparse

#----------------------------------------------------------------------------------------------
def get_phenix(configFileEntry = None):

  # Check that config file entry exists
  if not configFileEntry:
    print('configFileEntry is empty!')
    return

  # Load configuration variables
  filepath = configFileEntry['filepath']
  data_vrs = configFileEntry['data_vrs']
  data_str = configFileEntry['data_str']
  data_url = configFileEntry['data_url']
  data_doi = configFileEntry['data_doi']
  data_fig = configFileEntry['data_fig']
  data_exp = configFileEntry['data_exp']
  data_sys = configFileEntry['data_sys']
  data_meas = configFileEntry['data_meas']
  data_errs = configFileEntry['data_errs']
  data_cent = configFileEntry['data_cent']
  data_year = configFileEntry['data_year']

  # Prepare generic header portion
  generic_header = '# Version 1.0\n'
  generic_header += '# DOI ' + data_doi + '\n'
  generic_header += '# Source ' + data_url + '\n'
  generic_header += '# Experiment ' + data_exp + '\n'
  generic_header += '# System ' + data_sys + '\n'
  print(generic_header)

  # KLUDGE-ALERT
  #   For this url we need to disable certificates -- fixme later
  #   This step is not needed when fetching data from hepdata.net

  context = ssl._create_unverified_context()
  response = urlopen(data_url,context=context)
  content  = response.read().decode('utf-8')
  lines = content.split('\n')

  # Use fcnt to keep track of data files to write, one for each centrailty bin
  # header[fnct] to store header for each centrality file
  # data[fcnt]   to store multiline data for each centrality file
  fcnt   = 0
  files   = []
  headers = []
  entries = []

  for line in lines:
    # print(line)
    # Lines begining with Cent indicate new centrality bin
    m = re.match('Cent(\d+)\-(\d+)',line)
    if (m):
      fcnt += 1
      data_cent = m.group(1) + 'to' + m.group(2)

      # Create filename
      filelist = [data_str,data_exp,data_sys,data_meas,data_cent,data_year]
      # print(filelist)
      filename = filepath + '_'.join(filelist) + '.dat'
      files.append(filename)
      # print (fcnt,filename)

      # Complete Header
      this_header = generic_header
      this_header += '# Centrality ' + data_cent + '\n'
      this_header += '# XY pT RAA \n'
      this_header += '# Label x y stat,low stat,high corr,low corr,high global,low global,high \n'
      headers.append(this_header)
      entries.append('')

    # Lines begnning with a single digit assumed to be XY values for current centrality bin
    m = re.match('^\d',line)
    if (m):
      input = line.split()
      output = (input[0],input[1],input[2],input[2],input[4],input[4],input[6],input[6])
      # KLUDGE-ALERT, remove the two pT=19 entries with zero RAA
      if (output[1]!='0'):
        entries[-1] += ' '.join(output)
        entries[-1] += '\n'

  # Loop over files
  for i in list(range(fcnt)):
    print(i)
    print('Writing output to ',i,files[i])
    with open(files[i],'w+') as f:
      f.write(headers[i])
      f.write(entries[i])

  # print(files[i])
  # print(headers[i])
  # print(entries[i])

#----------------------------------------------------------------------------------------------
if __name__ == '__main__':
  get_phenix()
