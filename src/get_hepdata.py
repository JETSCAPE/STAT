"""
Fetch data from hepdata (yaml format) and write output files in jetscape-stat format documented
  [location of format specification file goes here]
"""

from urllib.request import urlopen
import re
import sys
import ssl
import yaml

# Specifiy Inputs
#   For now we specify we place all inputs in variables.
#   In the future we may move these into lists or xml.config files

filepath = '../input/MATTERLBT1/'
data_vrs = 'Version 1.0'
data_str = 'Data'
data_url = ['https://www.hepdata.net/download/table/ins1496050/Table14/yaml',
                'https://www.hepdata.net/download/table/ins1496050/Table11/yaml',
                'https://www.hepdata.net/download/table/ins1360290/Table27/1/yaml',
                'https://www.hepdata.net/download/table/ins1360290/Table29/1/yaml']
data_doi = ['http://dx.doi.org/10.1007/JHEP04(2017)039',
                'http://dx.doi.org/10.1007/JHEP04(2017)039',
                'http://dx.doi.org/10.1007/JHEP09(2015)050',
                'http://dx.doi.org/10.1007/JHEP09(2015)050']
data_fig = ['figure 4a','figure 4b','figure 13','figure 13']
data_exp = ['CMS','CMS','ATLAS','ATLAS']
data_sys = ['PbPb5020','PbPb5020','PbPb2760','PbPb2760']
data_meas = ['RAACharged','RAACharged','RAACharged','RAACharged']
data_cent = ['00to10','30to50','00to05','30to40']
data_year = ['2017','2017','2015','2015']

# KLUDGE-ALERT
#   For this url we need to disable certificates -- find out why don't Jonah's scripts need this
context = ssl._create_unverified_context()

for i in list(range(len(data_url))):
#for i in [2]:

    # Create new header for this entry
    data_header = '# JETSCAPE data entry version 1.0\n'
    data_header += '# DOI ' + data_doi[i] + '\n'
    data_header += '# Source ' + data_url[i] + '\n'
    data_header += '# Experiment ' + data_exp[i] + '\n'
    data_header += '# System ' + data_sys[i] + '\n'
    data_header += '# Centrality ' + data_cent[i] + '\n'
#   Fill the remaining header entries from hepdata

    # Reset data lists
    xlo  = []
    xhi  = []
    yval = []
    errs = []

    # Fetch yaml_data from hepdata url
    response = urlopen(data_url[i],context=context)
    content  = response.read().decode('utf-8')
    yaml_data = yaml.load(content)

    # Fetch XY names from headers
    xname = yaml_data['independent_variables'][0]['header']['name']
    yname = yaml_data['dependent_variables'][0]['header']['name']
    data_header += '# XY ' + xname + ' ' + yname + '\n'

    # Fetch error labels from first entry to create label for header
    error_label = ''
    first_entry = yaml_data['dependent_variables'][0]['values'][0]
    for err in first_entry['errors']:
        error_label += err['label']+'_lo ' + err['label']+'_hi '
    data_header +='# Label xmin xmax y ' + error_label + '\n'
    
#    print(yaml_data['independent_variables'][0]['values'])
    for x in yaml_data['independent_variables'][0]['values']:
        xlo.append(x['low'])
        xhi.append(x['high'])                    
#        print(x['low'],x['high'])

#    print(yaml_data['dependent_variables'][0]['values'])
    for v in yaml_data['dependent_variables'][0]['values']:
#        print(v['value'])
        yval.append(v['value'])

        errs.append('')
        for err in (v['errors']):
#            print(err['label'])
            try:
                e = str(err['symerror'])
                # Convert if percentage
#                if (e[-1]=='%'):
#                    e = str(float(e[:-1])*v['value']*0.01)
                errs[-1] += ' ' + e + ' ' + e
#                print ('  sym',e,e)
            except KeyError:
                e = err['asymerror']
                errs[-1] += ' ' + str(e['plus']) + ' ' + str(e['minus'])
#                print ('  asym',e['plus'],e['minus'])

    # Check that we have same number of x and y entries
    try:
        len(xlo)==len(yval)
    except Error:
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




