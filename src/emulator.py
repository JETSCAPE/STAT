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

    def __init__(self, system, npc=8, nrestarts=0):
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

    def predict(self, X):
        """
        Predict mean model output at X.

        Returns a nested dict of observable arrays, each with shape
        (n_samples_X, n_cent_bins).

        """
        return self._results_dict(
            self.pipeline.inverse_transform(
                np.concatenate([
                    gp.predict(X)[:, np.newaxis] for gp in self.gps
                ], axis=1)
            )
        )

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
