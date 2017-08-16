""" Markov chain Monte Carlo model calibration """

import argparse
from contextlib import contextmanager
import logging

import emcee
import h5py
import numpy as np
from scipy.linalg import lapack

from . import workdir, systems, expt
from .design import Design
from .emulator import emulators


def mvn_loglike(y, cov):
    """
    Evaluate the multivariate-normal log-likelihood for difference vector `y`
    and covariance matrix `cov`:

        log_p = -1/2*[(y^T).(C^-1).y + log(det(C))] + const.

    The likelihood is NOT NORMALIZED, since this does not affect MCMC.  The
    normalization const = -n/2*log(2*pi), where n is the dimensionality.

    Arguments `y` and `cov` MUST be np.arrays with dtype == float64 and shapes
    (n) and (n, n), respectively.  These requirements are NOT CHECKED.

    The calculation follows algorithm 2.1 in Rasmussen and Williams (Gaussian
    Processes for Machine Learning).

    """
    # Compute the Cholesky decomposition of the covariance.
    # Use bare LAPACK function to avoid scipy.linalg wrapper overhead.
    L, info = lapack.dpotrf(cov, clean=False)

    if info < 0:
        raise ValueError(
            'lapack dpotrf error: '
            'the {}-th argument had an illegal value'.format(-info)
        )
    elif info < 0:
        raise np.linalg.LinAlgError(
            'lapack dpotrf error: '
            'the leading minor of order {} is not positive definite'
            .format(info)
        )

    # Solve for alpha = cov^-1.y using the Cholesky decomp.
    alpha, info = lapack.dpotrs(L, y)

    if info != 0:
        raise ValueError(
            'lapack dpotrs error: '
            'the {}-th argument had an illegal value'.format(-info)
        )

    return -.5*np.dot(y, alpha) - np.log(L.diagonal()).sum()


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
        ('dET_deta', [None]),
        ('dN_dy', ['pion', 'kaon', 'proton']),
        ('mean_pT', ['pion', 'kaon', 'proton']),
        ('vnk', [(2, 2), (3, 2), (4, 2)]),
    ]

    def __init__(self, path=workdir / 'mcmc' / 'chain.hdf'):
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

        self._common_indices = list(range(len(systems), self.ndim))

        self._slices = {}
        self._expt_y = {}
        self._expt_cov = {}

        # pre-compute the experimental data vectors and covariance matrices
        for sys, sysdata in expt.data.items():
            nobs = 0

            self._slices[sys] = []

            for obs, subobslist in self.observables:
                try:
                    obsdata = sysdata[obs]
                except KeyError:
                    continue

                for subobs in subobslist:
                    try:
                        dset = obsdata[subobs]
                    except KeyError:
                        continue

                    n = dset['y'].size
                    self._slices[sys].append(
                        (obs, subobs, slice(nobs, nobs + n))
                    )
                    nobs += n

            self._expt_y[sys] = np.empty(nobs)
            self._expt_cov[sys] = np.empty((nobs, nobs))

            for obs1, subobs1, slc1 in self._slices[sys]:
                self._expt_y[sys][slc1] = expt.data[sys][obs1][subobs1]['y']
                for obs2, subobs2, slc2 in self._slices[sys]:
                    self._expt_cov[sys][slc1, slc2] = expt.cov(
                        sys, obs1, subobs1, obs2, subobs2
                    )

    def _predict(self, X, **kwargs):
        """
        Call each system emulator to predict model output at X.

        """
        return {
            sys: emulators[sys].predict(
                X[:, [n] + self._common_indices],
                **kwargs
            )
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

        nsamples = np.count_nonzero(inside)

        if nsamples > 0:
            pred = self._predict(X[inside], return_cov=True)

            for sys in systems:
                nobs = self._expt_y[sys].size
                # allocate difference (model - expt) and covariance arrays
                dY = np.empty((nsamples, nobs))
                cov = np.empty((nsamples, nobs, nobs))

                model_Y, model_cov = pred[sys]

                # copy predictive mean and covariance into allocated arrays
                for obs1, subobs1, slc1 in self._slices[sys]:
                    dY[:, slc1] = model_Y[obs1][subobs1]
                    for obs2, subobs2, slc2 in self._slices[sys]:
                        cov[:, slc1, slc2] = \
                            model_cov[(obs1, subobs1), (obs2, subobs2)]

                # subtract expt data from model data
                dY -= self._expt_y[sys]

                # add expt cov to model cov
                cov += self._expt_cov[sys]

                # compute log likelihood at each point
                lp[inside] += list(map(mvn_loglike, dY, cov))

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
