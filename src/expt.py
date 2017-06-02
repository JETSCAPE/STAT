""" experimental data """

from collections import defaultdict
import logging
import pickle
from urllib.request import urlopen

import numpy as np
import yaml

from . import cachedir, systems


class HEPData:
    """
    Interface to a HEPData yaml file.

    Downloads and caches the dataset specified by the INSPIRE record and table
    number.  The web UI for `inspire_rec` may be found at:

        https://hepdata.net/record/ins`inspire_rec`

    """
    def __init__(self, inspire_rec, table):
        cachefile = (
            cachedir / 'hepdata' /
            'ins{}_table{}.pkl'.format(inspire_rec, table)
        )
        name = 'record {} table {}'.format(inspire_rec, table)

        if cachefile.exists():
            logging.debug('loading from hepdata cache: %s', name)
            with cachefile.open('rb') as f:
                self._data = pickle.load(f)
        else:
            logging.debug('downloading from hepdata.net: %s', name)
            cachefile.parent.mkdir(exist_ok=True)
            with cachefile.open('wb') as f, urlopen(
                    'https://hepdata.net/download/table/'
                    'ins{}/Table{}/yaml'.format(inspire_rec, table)
            ) as u:
                self._data = yaml.load(u)
                pickle.dump(self._data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def x(self, name, case=True):
        """
        Get an independent variable ("x" data) with the given name.

        If `case` is false, perform case-insensitive matching for the name.

        """
        trans = (lambda x: x) if case else (lambda x: x.casefold())
        name = trans(name)

        for x in self._data['independent_variables']:
            if trans(x['header']['name']) == name:
                return x['values']

        raise LookupError("no x data with name '{}'".format(name))

    @property
    def cent(self):
        """
        Return a list of centrality bins as (low, high) tuples.

        """
        try:
            return self._cent
        except AttributeError:
            pass

        x = self.x('centrality', case=False)

        if x is None:
            raise LookupError('no centrality data')

        try:
            cent = [(v['low'], v['high']) for v in x]
        except KeyError:
            # try to guess bins from midpoints
            mids = [v['value'] for v in x]
            width = set(a - b for a, b in zip(mids[1:], mids[:-1]))
            if len(width) > 1:
                raise RuntimeError('variable bin widths')
            d = width.pop() / 2
            cent = [(m - d, m + d) for m in mids]

        self._cent = cent

        return cent

    def y(self, name=None, **quals):
        """
        Get a dependent variable ("y" data) with the given name and qualifiers.

        """
        for y in self._data['dependent_variables']:
            if name is None or y['header']['name'] == name:
                y_quals = {q['name']: q['value'] for q in y['qualifiers']}
                if all(y_quals[k] == v for k, v in quals.items()):
                    return y['values']

        raise LookupError(
            "no y data with name '{}' and qualifiers '{}'"
            .format(name, quals)
        )

    def dataset(self, name=None, maxcent=80, **quals):
        """
        Return a dict containing:

            cent : list of centrality bins
            x : np.array of centrality bin midpoints
            y : np.array of y values
            yerr : subdict of np.arrays of y errors

        `name` and `quals` are passed to HEPData.y()

        Missing y values are skipped.

        Centrality bins whose upper edge is greater than `maxcent` are skipped.

        """
        cent = []
        y = []
        yerr = defaultdict(list)

        for c, v in zip(self.cent, self.y(name, **quals)):
            # skip missing values
            # skip bins whose upper edge is greater than maxcent
            if v['value'] == '-' or c[1] > maxcent:
                continue

            cent.append(c)
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
            cent=cent,
            x=np.array([(a + b)/2 for a, b in cent]),
            y=np.array(y),
            yerr={k: np.array(v) for k, v in yerr.items()},
        )


def _data():
    """
    Acquire all experimental data.

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
                dsets[0],
                y=combine_func([d['y'] for d in dsets], axis=0),
                yerr={
                    e: combine_func([d['yerr'][e] for d in dsets], axis=0)
                    for e in dsets[0]['yerr']
                }
            )

    # PbPb2760 and PbPb5020 flows
    for system, tables_nk in [
            ('PbPb5020', [
                (1, [(2, 2), (2, 4)]),
                (2, [(3, 2), (4, 2)]),
            ]),
            ('PbPb2760', [
                (3, [(2, 2), (2, 4)]),
                (4, [(3, 2), (4, 2)]),
            ]),
    ]:
        data[system]['vnk'] = {}

        for table, nk in tables_nk:
            d = HEPData(1419244, table)
            for n, k in nk:
                data[system]['vnk'][n, k] = d.dataset(
                    'V{}{{{}{}}}'.format(
                        n, k, ', |DELTAETA|>1' if k == 2 else ''
                    )
                )

    # PbPb2760 central flows vn{2}
    system, obs = 'PbPb2760', 'vnk_central'
    data[system][obs] = {}

    for n, table, sys_err_frac in [(2, 11, .025), (3, 12, .040)]:
        dset = HEPData(900651, table).dataset()
        # the (unlabeled) errors in the dataset are actually stat
        dset['yerr']['stat'] = dset['yerr'].pop('sum')
        # sys error is not provided -- use estimated fractions
        dset['yerr']['sys'] = sys_err_frac * dset['y']
        data[system][obs][n, 2] = dset

    # PbPb2760 flow correlations
    for obs, table in [('sc', 1), ('sc_central', 3)]:
        d = HEPData(1452590, table)
        data['PbPb2760'][obs] = {
            mn: d.dataset('SC({},{})'.format(*mn))
            for mn in [(3, 2), (4, 2)]
        }

    return data


data = _data()


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
