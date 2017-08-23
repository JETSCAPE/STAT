""" Latin-hypercube parameter design """

import itertools
import logging
from pathlib import Path
import re
import subprocess

import numpy as np

from . import cachedir, parse_system


def generate_lhs(npoints, ndim, seed):
    """
    Generate a maximin LHS.

    """
    logging.debug(
        'generating maximin LHS: '
        'npoints = %d, ndim = %d, seed = %d',
        npoints, ndim, seed
    )

    cachefile = (
        cachedir / 'lhs' /
        'npoints{}_ndim{}_seed{}.npy'.format(npoints, ndim, seed)
    )

    if cachefile.exists():
        logging.debug('loading from cache')
        lhs = np.load(cachefile)
    else:
        logging.debug('not found in cache, generating using R')
        proc = subprocess.run(
            ['R', '--slave'],
            input="""
            library('lhs')
            set.seed({})
            write.table(maximinLHS({}, {}), col.names=FALSE, row.names=FALSE)
            """.format(seed, npoints, ndim).encode(),
            stdout=subprocess.PIPE,
            check=True
        )

        lhs = np.array(
            [l.split() for l in proc.stdout.splitlines()],
            dtype=float
        )

        cachefile.parent.mkdir(exist_ok=True)
        np.save(cachefile, lhs)

    return lhs


class Design:
    """
    Latin-hypercube model design.

    Creates a design for the given system with the given number of points.
    Creates the main (training) design if `validation` is false (default);
    creates the validation design if `validation` is true.  If `seed` is not
    given, a default random seed is used (different defaults for the main and
    validation designs).

    Public attributes:

        system: the system string
        projectiles, beam energy: system projectile pair and beam energy
        type: 'main' or 'validation'
        keys: list of parameter keys
        labels: list of parameter display labels (for TeX / matplotlib)
        range: list of parameter (min, max) tuples
        min, max: np.arrays of parameter min and max
        ndim: number of parameters (i.e. dimensions)
        points: list of design point names (formatted numbers)
        array: the actual design array

    The class also implicitly converts to np.array.

    Public methods:

        write_files: creates input files for running events

    This is probably the worst class in this project, and certainly the least
    generic.  It will probably need to be heavily edited for use in any other
    project, if not completely rewritten.

    """
    def __init__(self, system, npoints=500, validation=False, seed=None):
        self.system = system
        self.projectiles, self.beam_energy = parse_system(system)
        self.type = 'validation' if validation else 'main'

        # 5.02 TeV has ~1.2x particle production as 2.76 TeV
        # [https://inspirehep.net/record/1410589]
        norm_range = {
            2760: (8., 20.),
            5020: (10., 25.),
        }[self.beam_energy]

        self.keys, labels, self.range = map(list, zip(*[
            ('norm',          r'{Norm}',                      (norm_range   )),
            ('trento_p',      r'p',                           ( -0.5,    0.5)),
            ('fluct_std',     r'\sigma {fluct}',              (  0.0,    2.0)),
            ('nucleon_width', r'w [{fm}]',                    (  0.4,    1.0)),
            ('dmin3',         r'd_{min}^3 [{fm}^3]',          (  0.0, 1.7**3)),
            ('tau_fs',        r'\tau {fs} [{fm}/c]',          (  0.0,    1.5)),
            ('etas_hrg',      r'\eta/s {hrg}',                (  0.1,    0.5)),
            ('etas_min',      r'\eta/s {min}',                (  0.0,    0.2)),
            ('etas_slope',    r'\eta/s {slope} [{GeV}^{-1}]', (  0.0,    8.0)),
            ('etas_crv',      r'\eta/s {crv}',                ( -1.0,    1.0)),
            ('zetas_max',     r'\zeta/s {max}',               (  0.0,    0.1)),
            ('zetas_width',   r'\zeta/s {width} [{GeV}]',     (  0.0,    0.1)),
            ('zetas_t0',      r'\zeta/s T_0 [{GeV}]',         (0.150,  0.200)),
            ('Tswitch',       r'T {switch} [{GeV}]',          (0.135,  0.165)),
        ]))

        # convert labels into TeX:
        #   - wrap normal text with \mathrm{}
        #   - escape spaces
        #   - surround with $$
        self.labels = [
            re.sub(r'({[A-Za-z]+})', r'\mathrm\1', i)
            .replace(' ', r'\ ')
            .join('$$')
            for i in labels
        ]

        self.ndim = len(self.range)
        self.min, self.max = map(np.array, zip(*self.range))

        # use padded numbers for design point names
        fmt = '{:0' + str(len(str(npoints - 1))) + 'd}'
        self.points = [fmt.format(i) for i in range(npoints)]

        # XXX OK, what the heck is going on here?
        #
        # The original design transformed etas_slope to arctangent space, i.e.,
        # atan(slope) was sampled uniformly in (0, pi/2).  This was intended to
        # help place an upper bound on the slope, but it backfired, instead
        # making it almost impossible to train the GPs, since the model changed
        # so rapidly as atan(slope) approach pi/2.  It also turned out that
        # slope >~ 8 is clearly excluded, since this suppresses flow far too
        # much, regardless of the other parameters.
        #
        # As a result, I decided to re-run with the slope uniform in (0, 8),
        # and use the old design for validation.

        # While working with the original design data, I noticed that very
        # small tau_fs values were problematic, presumably due to numerical
        # issues (dividing by a small number, large initial energy densities,
        # etc).  I also realized that similar things could happen for other
        # parameters.  Thus, for the new design, I have set small but nonzero
        # minima for several parameters (and the emulators can extrapolate to
        # zero).
        lhsmin = self.min.copy()
        if not validation:
            for k, m in [
                    ('fluct_std', 1e-3),
                    ('tau_fs', 1e-3),
                    ('zetas_width', 1e-4),
            ]:
                lhsmin[self.keys.index(k)] = m

        if seed is None:
            seed = 751783496 if validation else 450829120

        self.array = lhsmin + (self.max - lhsmin)*generate_lhs(
            npoints=npoints, ndim=self.ndim, seed=seed
        )

        if validation:
            # Transform etas_slope from arctan space and remove points outside
            # the design range (see above).
            slope_idx = self.keys.index('etas_slope')
            slope_max = self.max[slope_idx]
            self.array[:, slope_idx] = \
                np.tan(np.pi/2/slope_max*self.array[:, slope_idx])
            keep = self.array[:, slope_idx] <= slope_max
            self.array = self.array[keep]
            self.points = list(itertools.compress(self.points, keep))
            logging.debug(
                'removed validation points with etas_slope > %s '
                '(%d points remaining)',
                slope_max, len(self.points)
            )

    def __array__(self):
        return self.array

    _template = ''.join(
        '{} = {}\n'.format(key, ' '.join(args)) for (key, *args) in
        [[
            'trento-args',
            '{projectiles[0]} {projectiles[1]}',
            '--cross-section {cross_section}',
            '--normalization {norm}',
            '--reduced-thickness {trento_p}',
            '--fluctuation {fluct}',
            '--nucleon-min-dist {dmin}',
        ], [
            'nucleon-width', '{nucleon_width}'
        ], [
            'tau-fs', '{tau_fs}'
        ], [
            'hydro-args',
            'etas_hrg={etas_hrg}',
            'etas_min={etas_min}',
            'etas_slope={etas_slope}',
            'etas_curv={etas_crv}',
            'zetas_max={zetas_max}',
            'zetas_width={zetas_width}',
            'zetas_t0={zetas_t0}',
        ], [
            'Tswitch', '{Tswitch}'
        ]]
    )

    def write_files(self, basedir):
        """
        Write input files for each design point to `basedir`.

        """
        outdir = basedir / self.type / self.system
        outdir.mkdir(parents=True, exist_ok=True)

        for point, row in zip(self.points, self.array):
            kwargs = dict(
                zip(self.keys, row),
                projectiles=self.projectiles,
                cross_section={
                    # sqrt(s) [GeV] : sigma_NN [fm^2]
                    200: 4.2,
                    2760: 6.4,
                    5020: 7.0,
                }[self.beam_energy]
            )
            kwargs.update(
                fluct=1/kwargs.pop('fluct_std')**2,
                dmin=kwargs.pop('dmin3')**(1/3),
            )
            filepath = outdir / point
            with filepath.open('w') as f:
                f.write(self._template.format(**kwargs))
                logging.debug('wrote %s', filepath)


def main():
    import argparse
    from . import systems

    parser = argparse.ArgumentParser(description='generate design input files')
    parser.add_argument(
        'inputs_dir', type=Path,
        help='directory to place input files'
    )
    args = parser.parse_args()

    for system, validation in itertools.product(systems, [False, True]):
        Design(system, validation=validation).write_files(args.inputs_dir)

    logging.info('wrote all files to %s', args.inputs_dir)


if __name__ == '__main__':
    main()
