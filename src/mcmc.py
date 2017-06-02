""" Markov chain Monte Carlo model calibration """

import argparse
from contextlib import contextmanager
import logging

import emcee
import h5py
import numpy as np

from . import workdir, systems, expt
from .design import Design
from .emulator import emulators


class LoggingEnsembleSampler(emcee.EnsembleSampler):
    def run_mcmc(self, X0, nsteps, status=None, **kwargs):
        """
        Run MCMC with logging every 'status' steps (default: approx 10% of
        nsteps).

        """
        logging.info('running %d walkers for %d steps', self.k, nsteps)

        if status is None:
            status = nsteps // 10

        for n, result in enumerate(
                self.sample(X0, iterations=nsteps, **kwargs),
                start=1
        ):
            if n % status == 0 or n == nsteps:
                af = self.acceptance_fraction
                logging.info(
                    'step %d: acceptance fraction: '
                    'mean %.4f, std %.4f, min %.4f, max %.4f',
                    n, af.mean(), af.std(), af.min(), af.max()
                )

        return result


class Chain:
    """
    High-level interface for running MCMC calibration and accessing results.

    Currently all design parameters except for the normalizations are required
    to be the same at all beam energies.  It is assumed (NOT checked) that all
    system designs have the same parameters and ranges (except for the norms).

    """
    # calibration observables
    # list of 2-tuples: (obs, [list of subobs])
    # each obs is checked for each system and silently ignored if not found
    observables = [
        ('dNch_deta', [None]),
        ('dN_dy', ['pion', 'kaon', 'proton']),
        ('mean_pT', ['pion', 'kaon', 'proton']),
        ('vnk', [(2, 2), (3, 2), (4, 2)]),
    ]

    def __init__(
            self, model_staterr_frac=.05,
            path=workdir / 'mcmc' / 'chain.hdf'
    ):
        # extra fractional error to add to the experimental uncertainty
        # accounts for model stat fluctuations and predictive uncertainty
        # TODO improve this
        self.model_staterr_frac = model_staterr_frac

        self.path = path
        self.path.parent.mkdir(exist_ok=True)

        def keys_labels_range():
            for sys in systems:
                d = Design(sys)
                klr = zip(d.keys, d.labels, d.range)
                k, l, r = next(klr)
                assert k == 'norm'
                yield (
                    '{} {}'.format(k, sys),
                    '{}\n{:.2f} TeV'.format(l, d.beam_energy/1000),
                    r
                )

            yield from klr

        self.keys, self.labels, self.range = map(
            list, zip(*keys_labels_range())
        )

        self.ndim = len(self.range)
        self.min, self.max = map(np.array, zip(*self.range))

        self.common_indices = list(range(len(systems), self.ndim))

        self.cov_inv = None

    def _predict(self, X):
        """
        Call each system emulator to predict model output at X.

        """
        return {
            sys: emulators[sys].predict(X[:, [n] + self.common_indices])
            for n, sys in enumerate(systems)
        }

    def log_posterior(self, X):
        """
        Evaluate the posterior at X.

        """
        X = np.array(X, copy=False, ndmin=2)

        lp = np.zeros(X.shape[0])

        inside = np.all((X > self.min) & (X < self.max), axis=1)
        lp[~inside] = -np.inf

        if inside.any():
            Y = self._predict(X[inside])

            for sys, sysdata in expt.data.items():
                for obs, subobslist in self.observables:
                    try:
                        obsdata = sysdata[obs]
                    except KeyError:
                        continue

                    for subobs in subobslist:
                        dY = Y[sys][obs][subobs] - obsdata[subobs]['y']
                        lp[inside] += np.einsum(
                            'ij,ki,kj->k',
                            self.cov_inv[sys][obs][subobs],
                            dY, dY
                        )

            lp[inside] *= -.5

        return lp

    def random_pos(self, n=1):
        """
        Generate random positions in parameter space.

        """
        return np.random.uniform(self.min, self.max, (n, self.ndim))

    @staticmethod
    def map(f, args):
        """
        Dummy function so that this object can be used as a 'pool' for
        emcee.EnsembleSampler.

        """
        return f(args)

    def run_mcmc(self, nsteps, nburnsteps=None, nwalkers=None, status=None):
        """
        Run MCMC model calibration.  If the chain already exists, continue from
        the last point, otherwise burn-in and start the chain.

        """
        # compute inverse covariance matrices for each experimental dataset
        if self.cov_inv is None:
            self.cov_inv = {
                sys: {
                    obs: {
                        subobs: np.linalg.inv(
                            expt.cov(**dset) +
                            np.diag((self.model_staterr_frac*dset['y'])**2)
                        )
                        for subobs, dset in obsdata.items()
                    } for obs, obsdata in sysdata.items()
                } for sys, sysdata in expt.data.items()
            }

        with self.open('a') as f:
            try:
                dset = f['chain']
            except KeyError:
                burn = True
                if nburnsteps is None or nwalkers is None:
                    logging.error(
                        'must specify nburnsteps and nwalkers to start chain'
                    )
                    return
                dset = f.create_dataset(
                    'chain', dtype='f8',
                    shape=(nwalkers, 0, self.ndim),
                    chunks=(nwalkers, 1, self.ndim),
                    maxshape=(nwalkers, None, self.ndim),
                    compression='lzf'
                )
            else:
                burn = False
                nwalkers = dset.shape[0]

            sampler = LoggingEnsembleSampler(
                nwalkers, self.ndim, self.log_posterior, pool=self
            )

            if burn:
                logging.info('no existing chain found, burning in')
                X0 = sampler.run_mcmc(
                    self.random_pos(nwalkers),
                    nburnsteps,
                    status=status,
                    storechain=False
                )[0]
                sampler.reset()
            else:
                logging.info('starting from last point of existing chain')
                X0 = dset[:, -1, :]

            sampler.run_mcmc(X0, nsteps, status=status)

            logging.info('writing chain to file')
            dset.resize(dset.shape[1] + nsteps, 1)
            dset[:, -nsteps:, :] = sampler.chain

    def open(self, mode='r'):
        """
        Return a handle to the chain HDF5 file.

        """
        return h5py.File(str(self.path), mode)

    @contextmanager
    def dataset(self, mode='r', name='chain'):
        """
        Return a dataset object in the chain HDF5 file.

        """
        with self.open(mode) as f:
            yield f[name]

    def load(self, *keys, thin=1):
        """
        Read the chain from file.  If 'keys' are given, read only those
        parameters.

        """
        if keys:
            indices = [self.keys.index(k) for k in keys]
            ndim = len(keys)
            if ndim == 1:
                indices = indices[0]
        else:
            ndim = self.ndim
            indices = slice(None)

        with self.dataset() as d:
            return np.array(d[:, ::thin, indices]).reshape(-1, ndim)

    def samples(self, n=1):
        """
        Predict model output at parameter points randomly drawn from the chain.

        """
        with self.dataset() as d:
            X = np.array([
                d[i] for i in zip(*[
                    np.random.randint(s, size=n) for s in d.shape[:2]
                ])
            ])

        return self._predict(X)


def credible_interval(samples, ci=.9):
    """
    Compute the HPD credible interval (default 90%) for an array of samples.

    """
    # number of intervals to compute
    nci = int((1 - ci)*samples.size)

    # find highest posterior density (HPD) credible interval
    # i.e. the one with minimum width
    argp = np.argpartition(samples, [nci, samples.size - nci])
    cil = np.sort(samples[argp[:nci]])   # interval lows
    cih = np.sort(samples[argp[-nci:]])  # interval highs
    ihpd = np.argmin(cih - cil)

    return cil[ihpd], cih[ihpd]


def main():
    parser = argparse.ArgumentParser(description='Markov chain Monte Carlo')

    parser.add_argument(
        'nsteps', type=int,
        help='number of steps'
    )
    parser.add_argument(
        '--nwalkers', type=int,
        help='number of walkers'
    )
    parser.add_argument(
        '--nburnsteps', type=int,
        help='number of burn-in steps'
    )
    parser.add_argument(
        '--status', type=int,
        help='number of steps between logging status'
    )

    Chain().run_mcmc(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
