""" experimental data """

from collections import defaultdict
from itertools import compress
import logging
import pickle
from urllib.request import urlopen

import numpy as np
import yaml

from . import cachedir, systems


# TODO improve flow cumulant observable keys:
# cumulant v_n{k} should have obs == 'vnk' and subobs == (n, k)


class HEPData:
    """
    Interface to a HEPData yaml file.

    Ignore centrality bins above the specified `maxcent`.

    """
    def __init__(self, inspire_rec, table, version=1, maxcent=80):
        cachefile = (
            cachedir / 'hepdata' /
            'ins{}_table{}.pkl'.format(inspire_rec, table)
        )
        logging.debug('loading hepdata record %s table %s', inspire_rec, table)

        if cachefile.exists():
            logging.debug('reading from cache')
            with cachefile.open('rb') as f:
                self.data = pickle.load(f)
        else:
            logging.debug('not found in cache, downloading from hepdata.net')
            cachefile.parent.mkdir(exist_ok=True)
            with cachefile.open('wb') as f, urlopen(
                    'https://hepdata.net/download/table/'
                    'ins{}/Table{}/{}/yaml'.format(inspire_rec, table, version)
            ) as u:
                self.data = yaml.load(u)
                pickle.dump(self.data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # extract centrality bins
        for x in self.data['independent_variables']:
            if x['header']['name'].lower() == 'centrality':
                try:
                    cent = [(v['low'], v['high']) for v in x['values']]
                except KeyError:
                    # try to guess bins from midpoints
                    mids = [v['value'] for v in x['values']]
                    width = set(a - b for a, b in zip(mids[1:], mids[:-1]))
                    if len(width) > 1:
                        raise ValueError('variable bin widths')
                    d = width.pop() / 2
                    cent = [(m - d, m + d) for m in mids]
                break

        # select bins whose upper edge is <= maxcent
        self._centselectors = [c[1] <= maxcent for c in cent]
        cent = self._filtercent(cent)

        # save centrality bins and midpoints as public attribute
        self.cent = dict(
            cent=cent,
            x=np.array([(a + b)/2 for a, b in cent])
        )

    def _filtercent(self, values):
        """
        Filter `values` by the centrality selectors created in the constructor
        (i.e. ignore bins above maxcent).

        """
        return list(compress(values, self._centselectors))

    def x(self, name):
        """
        Get an independent variable ("x" data) with the given name.

        """
        for x in self.data['independent_variables']:
            if x['header']['name'] == name:
                return self._filtercent(x['values'])

    def y(self, name=None, **quals):
        """
        Get a dependent variable ("y" data) with the given name and qualifiers.

        """
        for y in self.data['dependent_variables']:
            if name is None or y['header']['name'] == name:
                y_quals = {q['name']: q['value'] for q in y['qualifiers']}
                if all(y_quals[k] == v for k, v in quals.items()):
                    return self._filtercent(y['values'])

    def dataset(self, name=None, **quals):
        """
        Return a dict containing y values and errors along with centrality
        data.  Arguments are passed directly to self.y().

        """
        y = []
        yerr = defaultdict(list)

        for v in self.y(name, **quals):
            y.append(v['value'])
            for err in v['errors']:
                try:
                    e = err['symerror']
                except KeyError:
                    e = err['asymerror']
                    if abs(e['plus']) != abs(e['minus']):
                        raise RuntimeError(
                            'asymmetric errors are not implemented'
                        )
                    e = abs(e['plus'])

                yerr[err.get('label', 'sum')].append(e)

        return dict(
            y=np.array(y),
            yerr={k: np.array(v) for k, v in yerr.items()},
            **self.cent
        )


def get_calibration_data():
    """
    Experimental data for model calibration.

    """
    data = {s: {} for s in systems}

    # PbPb2760 and PbPb5020 dNch/deta
    for system, args, name in [
            ('PbPb2760', (880049, 1), 'D(N)/DETARAP'),
            ('PbPb5020', (1410589, 2),
             r'$\mathrm{d}N_\mathrm{ch}/\mathrm{d}\eta$'),
    ]:
        data[system]['dNch_deta'] = {None: HEPData(*args).dataset(name)}

    # PbPb2760 identified dN/dy and mean pT
    system = 'PbPb2760'

    for obs, table, combine_func in [
            ('dN_dy', 31, np.sum),
            ('mean_pT', 32, np.mean),
    ]:
        data[system][obs] = {}
        d = HEPData(1222333, table)
        for key, re_products in [
            ('pion', ['PI+', 'PI-']),
            ('kaon', ['K+', 'K-']),
            ('proton', ['P', 'PBAR']),
        ]:
            dsets = [
                d.dataset(RE='PB PB --> {} X'.format(i))
                for i in re_products
            ]

            data[system][obs][key] = dict(
                y=combine_func([d['y'] for d in dsets], axis=0),
                yerr={
                    e: combine_func([d['yerr'][e] for d in dsets], axis=0)
                    for e in dsets[0]['yerr']
                },
                **d.cent
            )

    # PbPb2760 and PbPb5020 flows
    for system, tables in [
            ('PbPb5020', [1, 2, 2]),
            ('PbPb2760', [3, 4, 4]),
    ]:
        data[system]['vn'] = {}

        for n, t in enumerate(tables, start=2):
            data[system]['vn'][n] = HEPData(1419244, t).dataset(
                'V{}{{2, |DELTAETA|>1}}'.format(n)
            )

    return data


data = get_calibration_data()


def get_extra_data():
    """
    Experimental data for model verification.  These observables require many
    more model events to compute and thus are not useful for calibration.

    """
    data = {s: {} for s in systems}

    # PbPb2760 flow correlations
    for obs, table in [('sc', 1), ('sc_central', 3)]:
        d = HEPData(1452590, table)
        data['PbPb2760'][obs] = {
            mn: d.dataset('SC({},{})'.format(*mn))
            for mn in [(3, 2), (4, 2)]
        }

    # PbPb2760 central flows vn{2}
    system, obs = 'PbPb2760', 'vn_central'
    data[system][obs] = {}

    for n, table in [(2, 11), (3, 12)]:
        dset = HEPData(900651, table).dataset()
        # the (unlabeled) errors in the dataset are actually stat
        dset['yerr']['stat'] = dset['yerr'].pop('sum')
        # estimate sys error fraction
        dset['yerr']['sys'] = {2: .025, 3: .040}[n]*dset['y']
        data[system][obs][n] = dset

    # PbPb2760 and PbPb5020 v2{4}
    for system, table in [('PbPb2760', 3), ('PbPb5020', 1)]:
        cent = []
        x = []
        y = []
        yerr = dict(stat=[], sys=[])

        # discard missing values in these datasets
        # TODO handle this in HEPData class
        d = HEPData(1419244, table)
        for v, x_, cent_ in zip(d.y('V2{4}'), d.cent['x'], d.cent['cent']):
            value = v['value']
            if value == '-':
                continue

            cent.append(cent_)
            x.append(x_)
            y.append(value)
            for e in v['errors']:
                yerr[e['label']].append(e['symerror'])

        # fix incorrect data point
        if system == 'PbPb5020':
            y[0] = .036
            yerr['stat'][0] = .003
            yerr['sys'][0] = .0006

        data[system]['vn4'] = {2: dict(
            cent=cent,
            x=np.array(x),
            y=np.array(y),
            yerr={k: np.array(v) for k, v in yerr.items()}
        )}

    return data


extra_data = get_extra_data()


def cov(x, y, yerr, stat_frac=1e-4, sys_corr_length=100, **kwargs):
    """
    Estimate a covariance matrix from stat and sys errors.

    """
    try:
        stat = yerr['stat']
        sys = yerr['sys']
    except KeyError:
        stat = y * stat_frac
        sys = yerr['sum']

    return np.diag(stat**2) + (
        np.exp(-.5*(np.subtract.outer(x, x)/sys_corr_length)**2) *
        np.outer(sys, sys)
    )


def print_data(d, indent=0):
    """
    Pretty print the nested data dict.

    """
    prefix = indent * '  '
    for k in sorted(d):
        v = d[k]
        k = prefix + str(k)
        if isinstance(v, dict):
            print(k)
            print_data(v, indent + 1)
        else:
            if k.endswith('cent'):
                v = ' '.join(
                    str(tuple(int(j) if j.is_integer() else j for j in i))
                    for i in v
                )
            elif isinstance(v, np.ndarray):
                v = str(v).replace('\n', '')
            print(k, '=', v)


if __name__ == '__main__':
    print_data(data)
    print_data(extra_data)
