""" Gaussian process emulator """

import logging
import pickle

import numpy as np
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process import kernels
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import cachedir, lazydict, model
from .design import Design


class _Covariance:
    """
    Proxy object to extract observable sub-blocks from a covariance array.
    Returned by Emulator.predict().

    """
    def __init__(self, array, slices):
        self.array = array
        self._slices = slices

    def __getitem__(self, key):
        (obs1, subobs1), (obs2, subobs2) = key
        return self.array[
            ...,
            self._slices[obs1][subobs1],
            self._slices[obs2][subobs2]
        ]


class Emulator:
    """
    Multidimensional Gaussian process emulator using principal component
    analysis.

    """
    # observables to emulate
    # list of 2-tuples: (obs, [list of subobs])
    observables = [
        ('dNch_deta', [None]),
        ('dET_deta', [None]),
        ('dN_dy', ['pion', 'kaon', 'proton']),
        ('mean_pT', ['pion', 'kaon', 'proton']),
        ('vnk', [(2, 2), (3, 2), (4, 2)]),
    ]

    def __init__(self, system, npc=10, nrestarts=0):
        logging.info(
            'training emulator for system %s (%d PC, %d restarts)',
            system, npc, nrestarts
        )

        arrays = []
        self._slices = {}

        index = 0
        for obs, subobslist in self.observables:
            self._slices[obs] = {}
            for subobs in subobslist:
                Y = model.data[system][obs][subobs]['Y']
                arrays.append(Y)
                n = Y.shape[1]
                self._slices[obs][subobs] = slice(index, index + n)
                index += n

        # StandardScaler with_mean is unnecessary since PCA removes the mean
        self.scaler = StandardScaler(with_mean=False, copy=False)
        self.pca = PCA(npc, copy=False, whiten=True, svd_solver='full')
        # XXX Although a pipeline is used here for convenience, the scaler and
        # pca objects are accessed directly when inverse transforming
        # predictive covariances.  If the pipeline is modified, self.predict()
        # must be updated accordingly.
        self.pipeline = make_pipeline(self.scaler, self.pca)

        Z = self.pipeline.fit_transform(np.concatenate(arrays, axis=1))

        design = Design(system)
        ptp = design.max - design.min
        kernel = (
            1. * kernels.RBF(
                length_scale=ptp,
                length_scale_bounds=np.outer(ptp, (.1, 10))
            ) +
            kernels.WhiteKernel(
                noise_level=.01,
                noise_level_bounds=(1e-8, 10)
            )
        )

        self.gps = [
            GPR(
                kernel=kernel, alpha=0,
                n_restarts_optimizer=nrestarts,
                copy_X_train=False
            ).fit(design, z)
            for z in Z.T
        ]

    @classmethod
    def from_cache(cls, system, retrain=False, **kwargs):
        """
        Load from the cache if available, otherwise create and cache a new
        instance.

        """
        cachefile = cachedir / 'emulator' / '{}.pkl'.format(system)

        # cache the __dict__ rather than the Emulator instance itself
        # this way the __name__ doesn't matter, e.g. a pickled
        # __main__.Emulator can be unpickled as a src.emulator.Emulator
        if not retrain and cachefile.exists():
            logging.debug('loading emulator for system %s from cache', system)
            emu = cls.__new__(cls)
            emu.__dict__ = joblib.load(cachefile)
            return emu

        emu = cls(system, **kwargs)

        logging.info('writing cache file %s', cachefile)
        cachefile.parent.mkdir(exist_ok=True)
        joblib.dump(emu.__dict__, cachefile, protocol=pickle.HIGHEST_PROTOCOL)

        return emu

    def _results_dict(self, Y):
        """
        Unpack an array of predictions into a nested dict of observables.

        """
        return {
            obs: {
                subobs: Y[..., s]
                for subobs, s in slices.items()
            } for obs, slices in self._slices.items()
        }

    def predict(self, X, return_cov=False):
        """
        Predict model output at X.

        X must be a 2D array-like with shape (nsamples, ndim).  It is passed
        directly to sklearn GaussianProcessRegressor.predict().

        If return_cov is true, return a tuple (mean, cov), otherwise only
        return the mean.

        The mean is returned as a nested dict of observable arrays, each with
        shape (nsamples, n_cent_bins).

        The covariance is returned as a proxy object which extracts observable
        sub-blocks using a dict-like interface:

        >>> mean, cov = emulator.predict(X, return_cov=True)

        >>> mean['dN_dy']['pion']
        <mean prediction of pion dN/dy>

        >>> cov[('dN_dy', 'pion'), ('dN_dy', 'pion')]
        <covariance matrix of pion dN/dy>

        >>> cov[('dN_dy', 'pion'), ('mean_pT', 'kaon')]
        <covariance matrix between pion dN/dy and kaon mean pT>

        The shape of the extracted covariance blocks are
        (nsamples, n_cent_bins_1, n_cent_bins_2).

        NB: the covariance is only computed between observables and centrality
        bins, not between sample points.

        """
        gp_mean = [gp.predict(X, return_cov=return_cov) for gp in self.gps]

        if return_cov:
            gp_mean, gp_cov = zip(*gp_mean)

        # inverse transform the predictive mean
        mean = self._results_dict(
            self.pipeline.inverse_transform(
                np.concatenate([
                    m[:, np.newaxis] for m in gp_mean
                ], axis=1)
            )
        )

        if return_cov:
            # Now transform the predictive variance in PC space to physical
            # space.  Assuming the PCs are uncorrelated, the transformation is
            #
            #   cov_ij = sum_k A_ik var_k A_jk
            #
            # where A is the PC transformation matrix and var_k is the
            # variance of the kth PC.
            # https://en.wikipedia.org/wiki/Propagation_of_uncertainty

            # In the einsum calls below, explicitly disable optimization (which
            # attempts to find the fastest computational path) because the path
            # is already optimized "by hand".

            # Create the linear transformation matrix.
            # shape: (npc, nobs)
            A = np.einsum(
                'ij,i,j->ij',
                self.pca.components_,
                np.sqrt(self.pca.explained_variance_),
                self.scaler.scale_,
                optimize=False
            )

            # At this point, the transformation can be accomplished by
            #
            #   einsum('ki,ka,kj->aij', A, gp_var, A)
            #
            # however it's faster to first expand the ij indices and then sum
            # over k using a dot product.

            # Perform the first part of the transformation.
            # shape: (npc, nobs, nobs)
            A = np.einsum('ki,kj->kij', A, A, optimize=False)

            # Build array of the GP predictive variances at each sample point.
            # shape: (nsamples, npc)
            gp_var = np.concatenate([
                c.diagonal()[:, np.newaxis] for c in gp_cov
            ], axis=1)

            # Compute the covariance at each sample point.
            # (Reshaping A to a 2D array allows np.dot to use BLAS.)
            npc, nobs = self.pca.components_.shape
            cov = np.dot(gp_var, A.reshape(npc, -1)).reshape(-1, nobs, nobs)

            # Add the PCA noise variance (i.e. truncation error) to the
            # diagonals.
            i = np.arange(nobs)
            cov[:, i, i] += self.pca.noise_variance_ * self.scaler.var_

            return mean, _Covariance(cov, self._slices)
        else:
            return mean

    def sample_y(self, X, n_samples=1, random_state=None):
        """
        Sample model output at X.

        Returns a nested dict of observable arrays, each with shape
        (n_samples_X, n_samples, n_cent_bins).

        """
        return self._results_dict(
            self.pipeline.inverse_transform(
                np.concatenate([
                    gp.sample_y(
                        X, n_samples=n_samples, random_state=random_state
                    ).reshape(-1, 1)
                    for gp in self.gps
                ], axis=1)
            ).reshape(X.shape[0], n_samples, -1)
        )


emulators = lazydict(Emulator.from_cache)


if __name__ == '__main__':
    import argparse
    from . import systems

    def arg_to_system(arg):
        if arg not in systems:
            raise argparse.ArgumentTypeError(arg)
        return arg

    parser = argparse.ArgumentParser(
        description='train emulators for each collision system',
        argument_default=argparse.SUPPRESS
    )

    parser.add_argument(
        '--npc', type=int,
        help='number of principal components'
    )
    parser.add_argument(
        '--nrestarts', type=int,
        help='number of optimizer restarts'
    )

    parser.add_argument(
        '--retrain', action='store_true',
        help='retrain even if emulator is cached'
    )
    parser.add_argument(
        'systems', nargs='*', type=arg_to_system,
        default=systems, metavar='SYSTEM',
        help='system(s) to train'
    )

    args = parser.parse_args()
    kwargs = vars(args)

    for s in kwargs.pop('systems'):
        emu = Emulator.from_cache(s, **kwargs)

        print(s)
        print('{} PCs explain {:.5f} of variance'.format(
            emu.pca.n_components_,
            emu.pca.explained_variance_ratio_.sum()
        ))

        for n, (evr, gp) in enumerate(zip(
                emu.pca.explained_variance_ratio_, emu.gps
        )):
            print(
                'GP {}: {:.5f} of variance, kernel: {}'
                .format(n, evr, gp.kernel_)
            )
