"""
Fetch data from hepdata (yaml format) and write output files in jetscape-stat format documented
  [location of format specification file goes here]
"""

from urllib.request import urlopen
import re
import sys
import ssl
import yaml
import os

#----------------------------------------------------------------------------------------------
def get_hepdata(configFileEntry = None):

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
  # May not always be defined, so need to handle a bit more carefully
  data_qualifiers = configFileEntry.get("data_qualifiers", [] * len(data_sys))
  data_meas = configFileEntry['data_meas']
  data_cent = configFileEntry['data_cent']
  data_year = configFileEntry['data_year']

  # Create directory for filepath, if it doesn't exist
  if not filepath.endswith('/'):
    filepath = filepath + '/'
  if not os.path.exists(filepath):
    os.makedirs(filepath)

  # KLUDGE-ALERT
  # For this url we need to disable certificates -- find out why don't Jonah's scripts need this
  context = ssl._create_unverified_context()

  for i in list(range(len(data_url))):

    # Create new header for this entry
    data_header = '# Version 1.0\n'
    data_header += '# DOI ' + data_doi[i] + '\n'
    data_header += '# Source ' + data_url[i] + '\n'
    data_header += '# Experiment ' + data_exp[i] + '\n'
    data_header += '# System ' + data_sys[i] + '\n'
    data_header += '# Centrality ' + data_cent[i] + '\n'
    # Fill the remaining header entries from hepdata

    # Reset data lists
    xlo  = []
    xhi  = []
    yval = []
    errs = []

    # Fetch yaml_data from hepdata url
    response = urlopen(data_url[i],context=context)
    content  = response.read().decode('utf-8')
    yaml_data = yaml.safe_load(content)

    # Determine which dependent variable entry to take
    n_dependent_variable_entries = len(yaml_data['dependent_variables'])
    if n_dependent_variable_entries > 1 and "data_qualifiers" in configFileEntry:
      dependent_variable_index = -1
      # Search the dependent variables qualifiers to find the one that matches
      selected_qualifiers = data_qualifiers[i]
      print(f"Searching for the right dependent variable with qualifiers {selected_qualifiers}")
      for _i, dependent_variable_entry in enumerate(yaml_data["dependent_variables"]):
        qualifiers_list = dependent_variable_entry["qualifiers"]
        for qualifier in qualifiers_list:
          if qualifier["name"] == selected_qualifiers["name"] and qualifier["value"] == selected_qualifiers["value"]:
            dependent_variable_index = _i
            break
      if dependent_variable_index < 0:
        raise ValueError(f"Unable to find qualifiers {selected_qualifiers}")
    elif n_dependent_variable_entries > 1:
      raise ValueError(
        "There are multiple dependent variables entries, and we don't know which one to select."
        " Please specify a qualifier to select the right dependent variable."
      )
    else:
      # Fall back to 0 by default if not specified
      dependent_variable_index = 0
    print(f"Index of dependent variable to extract: {dependent_variable_index}")

    # Sometimes, not all values are provided. If they're not provided, we should skip them.
    # This includes skipping the bin sizes, etc.
    _indices_to_skip = []
    for _i_value, v in enumerate(yaml_data['dependent_variables'][dependent_variable_index]['values']):
      try:
        float(v["value"])
      except ValueError:
        _indices_to_skip.append(_i_value)
    print(f"Skipping indices due to missing values: {_indices_to_skip}")

    # Fetch XY names from headers
    xname = yaml_data['independent_variables'][0]['header']['name']
    yname = yaml_data['dependent_variables'][dependent_variable_index]['header']['name']
    data_header += '# XY ' + xname + ' ' + yname + '\n'

    # Fetch error labels from first entry to create label for header
    error_label = ''
    # The first entry may not have errors (for example, CMS 5 TeV Jet RAA),
    # so we have to search for the first entry with errors.
    for _i_value, _entry in enumerate(yaml_data['dependent_variables'][dependent_variable_index]['values']):
      if _i_value in _indices_to_skip:
        continue
      if "errors" in _entry:
        # We need the statistical errors to be first, so first we do a search to find the stat errors.
        _stat_error_index = None
        for _i_temp, error_entry in enumerate(_entry["errors"]):
          if "stat" in error_entry["label"]:
            _stat_error_index = _i_temp
            break
        else:
          print("\tWARNING: Could not find statistical error. Please check the data source!")
        if _stat_error_index is not None:
          if _stat_error_index != 0:
            print("\tINFO: Heads up, we're rearraning the errors so that the statistical errors is first.")
          _order_to_extract_errors = list(range(len(_entry["errors"])))
          _order_to_extract_errors.remove(_stat_error_index)
          # Insert the stat error as the first entry. Usually, this will be index 0
          _order_to_extract_errors.insert(0, _stat_error_index)
        else:
          # We don't have the stat error - just use as is and hope for the best.
          _order_to_extract_errors = list(range(len(_entry["errors"])))

        all_error_entries = _entry["errors"]
        for _i_error in _order_to_extract_errors:
          err = all_error_entries[_i_error]
          # If there are spaces, it will break reading the data later
          error_label_text = err["label"].replace(" ", "_")
          error_label += error_label_text + ',low ' + error_label_text + ',high '
        # Once we've found one entry, we're done.
        break
    else:
      raise ValueError("Are there errors in this file?")
    data_header +='# Label xmin xmax y ' + error_label + '\n'

    # print(yaml_data['independent_variables'][0]['values'])
    for _i_value, x in enumerate(yaml_data['independent_variables'][0]['values']):
      if _i_value in _indices_to_skip:
        continue
      xlo.append(x['low'])
      xhi.append(x['high'])
      # print(x['low'],x['high'])

    # print(yaml_data['dependent_variables'][0]['values'])
    for _i_value, v in enumerate(yaml_data['dependent_variables'][dependent_variable_index]['values']):
      if _i_value in _indices_to_skip:
        continue
      # print(v['value'])
      yval.append(v['value'])

      errs.append('')
      all_error_entries = v['errors']
      for _i_error in _order_to_extract_errors:
        err = all_error_entries[_i_error]
        # print(err['label'])
        try:
          e = str(err['symerror'])
          # Convert if percentage
          if (e[-1]=='%'):
            e = str(float(e[:-1])*v['value']*0.01)
          errs[-1] += ' ' + e + ' ' + e
          # print ('  sym',e,e)
        except KeyError:
          e = err['asymerror']
          errs[-1] += ' ' + str(abs(float(e['plus']))) + ' ' + str(abs(float(e['minus'])))
          # print ('  asym',e['plus'],e['minus'])

    # Check that we have same number of x and y entries
    try:
      len(xlo)==len(yval)
    except Exception:
      message = ' Unequal x and y entries ' + str(len(xlo)) + ' ' + str(len(yval))
      sys.exit(message)

    data_lines = ''
    for j in list(range(len(xlo))):
      data_lines += str(xlo[j]) + ' ' + str(xhi[j]) + ' ' + str(yval[j]) + errs[j] + '\n'

    # Create filename
    namelist = [ data_str,data_exp[i],data_sys[i],data_meas[i],data_cent[i],data_year[i] ]
    filename = filepath + '_'.join(namelist) + '.dat'

    print('Writing ',filename)
    with open(filename,'w+') as f:
      f.write(data_header)
      f.write(data_lines)

#----------------------------------------------------------------------------------------------
if __name__ == '__main__':
  get_hepdata()
