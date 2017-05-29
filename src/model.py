""" model output """

import logging
from pathlib import Path
import pickle

from hic import flow
import numpy as np
from sklearn.externals import joblib

from . import workdir, cachedir, systems, expt
from .design import Design


# TODO move this symmetric cumulant code to hic

def csq(x):
    """
    Return the absolute square |x|^2 of a complex array.

    """
    return (x*x.conj()).real


def corr2(Qn, N):
    """
    Compute the two-particle correlation <v_n^2>.

    """
    return (csq(Qn) - N).sum() / (N*(N - 1)).sum()


def symmetric_cumulant(events, m, n):
    """
    Compute the symmetric cumulant SC(m, n).

    """
    N = np.asarray(events['flow']['N'], dtype=float)
    Q = dict(enumerate(events['flow']['Qn'].T, start=1))

    cm2n2 = (
        csq(Q[m]) * csq(Q[n])
        - 2*(Q[m+n] * Q[m].conj() * Q[n].conj()).real
        - 2*(Q[m] * Q[m-n].conj() * Q[n].conj()).real
        + csq(Q[m+n]) + csq(Q[m-n])
        - (N - 4)*(csq(Q[m]) + csq(Q[n]))
        + N*(N - 6)
    ).sum() / (N*(N - 1)*(N - 2)*(N - 3)).sum()

    cm2 = corr2(Q[m], N)
    cn2 = corr2(Q[n], N)

    return cm2n2 - cm2*cn2


# fully specify numeric data types, including endianness and size, to
# ensure consistency across all machines
float_t = '<f8'
int_t = '<i8'
complex_t = '<c16'


class ModelData:
    """
    Helper class for event-by-event model data.  Reads binary data files and
    computes centrality-binned observables.

    """
    species = ['pion', 'kaon', 'proton', 'Lambda', 'Sigma0', 'Xi', 'Omega']

    dtype = np.dtype([
        ('initial_entropy', float_t),
        ('nsamples', int_t),
        ('dNch_deta', float_t),
        ('dET_deta', float_t),
        ('dN_dy', [(s, float_t) for s in species]),
        ('mean_pT', [(s, float_t) for s in species]),
        ('pT_fluct', [('N', int_t), ('sum_pT', float_t), ('sum_pTsq', float_t)]),
        ('flow', [('N', int_t), ('Qn', complex_t, 8)]),
    ])

    def __init__(self, *files):
        # read each file using the above dtype and treat each as a minimum-bias
        # event sample
        def load_events(f):
            logging.debug('loading %s', f)
            d = np.fromfile(str(f), dtype=self.dtype)
            d.sort(order='dNch_deta')
            return d

        self.events = [load_events(f) for f in files]

    def observables_like(self, data, *keys):
        """
        Compute the same centrality-binned observables as contained in `data`
        with the same nested dict structure.

        This function calls itself recursively, each time prepending to `keys`.

        """
        try:
            x = data['x']
            cent = data['cent']
        except KeyError:
            return {
                k: self.observables_like(v, k, *keys)
                for k, v in data.items()
            }

        def _compute_bin():
            """
            Choose a function to compute the current observable for a single
            centrality bin.

            """
            obs_stack = list(keys)
            obs = obs_stack.pop()

            if obs == 'dNch_deta':
                return lambda events: events[obs].mean()

            if obs == 'dN_dy':
                species = obs_stack.pop()
                return lambda events: events[obs][species].mean()

            if obs == 'mean_pT':
                species = obs_stack.pop()
                return lambda events: np.average(
                    events[obs][species],
                    weights=events['dN_dy'][species]
                )

            if obs.startswith('vn'):
                n = obs_stack.pop()
                k = 4 if obs == 'vn4' else 2
                return lambda events: flow.Cumulant(
                    events['flow']['N'], *events['flow']['Qn'].T[1:]
                ).flow(n, k, imaginary='zero')

            if obs.startswith('sc'):
                mn = obs_stack.pop()
                return lambda events: symmetric_cumulant(events, *mn)

        compute_bin = _compute_bin()

        def compute_all_bins(events):
            n = events.size
            bins = [
                events[int((1 - b/100)*n):int((1 - a/100)*n)]
                for a, b in cent
            ]

            return list(map(compute_bin, bins))

        return dict(
            x=x, cent=cent,
            Y=np.array(list(map(compute_all_bins, self.events))).squeeze()
        )


def observables(system, map_point=False):
    """
    Compute model observables for the given system to match the corresponding
    experimental data.

    """
    if map_point:
        files = [Path('map', system)]
        cachefile = Path(system + '_map')
    else:
        # expected filenames for each design point
        files = [Path(system, p) for p in Design(system).points]
        cachefile = Path(system)

    files = [workdir / 'model_output' / f.with_suffix('.dat') for f in files]
    cachefile = cachedir / 'model' / cachefile.with_suffix('.pkl')

    if cachefile.exists():
        # use the cache unless any of the model data files are newer
        # this DOES NOT check any other logical dependencies, e.g. the
        # experimental data
        # to force recomputation, delete the cache file
        mtime = cachefile.stat().st_mtime
        if all(f.stat().st_mtime < mtime for f in files):
            logging.debug('loading observables cache file %s', cachefile)
            return joblib.load(cachefile)
        else:
            logging.debug('cache file %s is older than event data', cachefile)
    else:
        logging.debug('cache file %s does not exist', cachefile)

    logging.info(
        'loading %s%s event data and computing observables',
        system,
        '_map' if map_point else ''
    )

    data = expt.data[system]

    # identified particle data are not yet available for PbPb5020
    # create dummy entries for these observables so that they are computed for
    # the model
    if system == 'PbPb5020':
        data = dict(
            ((obs, expt.data['PbPb2760'][obs])
             for obs in ['dN_dy', 'mean_pT']),
            **data
        )

    # also compute "extra" data for the MAP point
    if map_point:
        data = dict(expt.extra_data[system], **data)
        # flow correlations and central flow not yet available for PbPb5020
        if system == 'PbPb5020':
            data = dict(
                ((obs, expt.extra_data['PbPb2760'][obs])
                for obs in ['sc', 'sc_central', 'vn_central']),
                **data
            )


    data = ModelData(*files).observables_like(data)

    logging.info('writing cache file %s', cachefile)
    cachefile.parent.mkdir(exist_ok=True)
    joblib.dump(data, cachefile, protocol=pickle.HIGHEST_PROTOCOL)

    return data


data = {s: observables(s) for s in systems}
map_data = {s: observables(s, map_point=True) for s in systems}


if __name__ == '__main__':
    from pprint import pprint
    print('design:')
    pprint(data)
    print('map:')
    pprint(map_data)
