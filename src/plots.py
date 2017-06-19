""" plots / visualizations / figures """

import colorsys
import itertools
import logging
from pathlib import Path
import subprocess
import tempfile
import warnings

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import lines
from matplotlib import patches
from matplotlib import ticker
from scipy.interpolate import PchipInterpolator
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process import kernels
from sklearn.mixture import GaussianMixture

from . import workdir, systems, parse_system, expt, model, mcmc
from .design import Design
from .emulator import emulators


def darken(rgba, amount=.5):
    h, l, s = colorsys.rgb_to_hls(*rgba[:3])
    r, g, b = colorsys.hls_to_rgb(h, l*amount, s)

    try:
        return r, g, b, rgba[3]
    except IndexError:
        return r, g, b


fontsmall, fontnormal, fontlarge = 5, 6, 7
offblack = '#262626'
aspect = 1/1.618
resolution = 72.27
textwidth = 307.28987/resolution
textheight = 261.39864/resolution
fullwidth = 350/resolution
fullheight = 270/resolution

plt.rcdefaults()
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Lato'],
    'mathtext.fontset': 'custom',
    'mathtext.default': 'it',
    'mathtext.rm': 'sans',
    'mathtext.it': 'sans:italic:medium',
    'mathtext.cal': 'sans',
    'font.size': fontnormal,
    'legend.fontsize': fontnormal,
    'axes.labelsize': fontnormal,
    'axes.titlesize': fontlarge,
    'xtick.labelsize': fontsmall,
    'ytick.labelsize': fontsmall,
    'font.weight': 400,
    'axes.labelweight': 400,
    'axes.titleweight': 400,
    'lines.linewidth': .5,
    'lines.markersize': 3,
    'lines.markeredgewidth': 0,
    'patch.linewidth': .5,
    'axes.linewidth': .4,
    'xtick.major.width': .4,
    'ytick.major.width': .4,
    'xtick.minor.width': .4,
    'ytick.minor.width': .4,
    'xtick.major.size': 1.2,
    'ytick.major.size': 1.2,
    'xtick.minor.size': .8,
    'ytick.minor.size': .8,
    'xtick.major.pad': 1.5,
    'ytick.major.pad': 1.5,
    'axes.formatter.limits': (-5, 5),
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelpad': 3,
    'text.color': offblack,
    'axes.edgecolor': offblack,
    'axes.labelcolor': offblack,
    'xtick.color': offblack,
    'ytick.color': offblack,
    'legend.numpoints': 1,
    'legend.scatterpoints': 1,
    'legend.frameon': False,
    'image.cmap': 'Blues',
    'image.interpolation': 'none',
    'pdf.fonttype': 42
})


plotdir = workdir / 'plots'
plotdir.mkdir(exist_ok=True)

plot_functions = {}


def plot(f):
    """
    Plot function decorator.  Calls the function, does several generic tasks,
    and saves the figure as the function name.

    """
    def wrapper(*args, **kwargs):
        logging.info('generating plot: %s', f.__name__)
        f(*args, **kwargs)

        fig = plt.gcf()

        if not fig.get_tight_layout():
            set_tight(fig)

        plotfile = plotdir / '{}.pdf'.format(f.__name__)
        fig.savefig(str(plotfile))
        logging.info('wrote %s', plotfile)
        plt.close(fig)

    plot_functions[f.__name__] = wrapper

    return wrapper


def set_tight(fig=None, **kwargs):
    """
    Set tight_layout with a better default pad.

    """
    if fig is None:
        fig = plt.gcf()

    kwargs.setdefault('pad', .1)
    fig.set_tight_layout(kwargs)


def remove_ticks(*axes):
    """
    Remove all tick marks (but not labels).

    """
    if not axes:
        axes = plt.gcf().axes

    for ax in axes:
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')


def auto_ticks(
        ax, xy=None, nbins=5, steps=[1, 2, 4, 5, 10],
        prune=None, minor=0
):
    """
    Convenient interface to matplotlib.ticker locators.

    """
    if xy == 'x':
        axes = ax.xaxis,
    elif xy == 'y':
        axes = ax.yaxis,
    else:
        axes = ax.xaxis, ax.yaxis

    for axis in axes:
        axis.set_major_locator(
            ticker.MaxNLocator(nbins=nbins, steps=steps, prune=prune)
        )
        if minor:
            axis.set_minor_locator(ticker.AutoMinorLocator(minor))


def _observables_plots():
    Nch_ET = [
        ('dNch_deta', None, r'$N_\mathrm{ch}$', 'Greys'),
        ('dET_deta', None, r'$E_T$', 'PuRd'),
    ]

    def id_parts(obs):
        return [
            (obs, 'pion',   r'$\pi^\pm$', 'Blues'),
            (obs, 'kaon',   r'$K^\pm$', 'Greens'),
            (obs, 'proton', r'$p\bar p$', 'Reds'),
        ]

    flows = [
        ('vnk', (n, 2), '$v_{}$'.format(n), c)
        for n, c in enumerate(['GnBu', 'Purples', 'Oranges'], start=2)
    ]

    return [
        ('Yields',  r',\ '.join([
            r'$dN_\mathrm{ch}/d\eta',
            r'dN/dy',
            r'dE_T/d\eta\ [\mathrm{GeV}]$',
        ]), (1., 1e5), Nch_ET + id_parts('dN_dy')),
        ('Mean $p_T$', r'$p_T$ [GeV]', (0, 2.), id_parts('mean_pT')),
        ('Flow cumulants', r'$v_n\{2\}$', (0, 0.15), flows),
    ]


def _observables(posterior=False):
    """
    Model observables at all design points or drawn from the posterior with
    experimental data points.

    """
    plots = _observables_plots()

    fig, axes = plt.subplots(
        nrows=len(systems), ncols=len(plots),
        figsize=(fullwidth, .55*fullwidth)
    )

    if posterior:
        samples = mcmc.Chain().samples(100)

    for (system, (title, ylabel, ylim, subplots)), ax in zip(
            itertools.product(systems, plots), axes.flat
    ):
        for obs, subobs, label, cmap in subplots:
            factor = 5**dict(dNch_deta=1, dET_deta=2).get(obs, 0)
            color = getattr(plt.cm, cmap)(.6)

            x = model.data[system][obs][subobs]['x']
            Y = (
                samples[system][obs][subobs]
                if posterior else
                model.data[system][obs][subobs]['Y']
            )

            for y in Y * factor:
                ax.plot(x, y, color=color, alpha=.08, lw=.3)

            ax.text(
                x[-1] + 2.5,
                np.median(Y[:, -1]) * factor,
                label,
                color=darken(color), ha='left', va='center'
            )

            try:
                dset = expt.data[system][obs][subobs]
            except KeyError:
                continue

            x = dset['x']
            y = dset['y'] * factor
            yerr = np.sqrt(sum(
                e**2 for e in dset['yerr'].values()
            )) * factor

            ax.errorbar(
                x, y, yerr=yerr, fmt='o', ms=1.7,
                capsize=0, color='.25', zorder=1000
            )

        if title == 'Yields':
            ax.set_yscale('log')
            ax.minorticks_off()
        else:
            auto_ticks(ax, 'y', nbins=4, minor=2)

        if ax.is_first_row():
            ax.set_title(title)
        elif ax.is_last_row():
            ax.set_xlabel('Centrality %')

        if ax.is_last_col():
            proj, energy = parse_system(system)
            ax.text(
                1.07, .5, '{} {:.2f} TeV'.format('+'.join(proj), energy/1000),
                transform=ax.transAxes, ha='left', va='center',
                size=plt.rcParams['axes.titlesize'], rotation=-90
            )

        l = ax.set_ylabel(ylabel)
        if len(ylabel) > 30:
            l.set_fontsize(.75*plt.rcParams['axes.labelsize'])
        ax.set_ylim(ylim)

    set_tight(fig, w_pad=1, rect=[0, 0, .97, 1])


@plot
def observables_design():
    _observables(posterior=False)


@plot
def observables_posterior():
    _observables(posterior=True)


@plot
def observables_map():
    """
    Model observables and ratio to experiment at the maximum a posteriori
    (MAP) estimate.

    """
    plots = _observables_plots()

    fig = plt.figure(figsize=(fullwidth, .85*fullheight))
    gs = plt.GridSpec(3*len(systems), len(plots))

    for (nsys, system), (nplot, (title, ylabel, ylim, subplots)) in \
            itertools.product(enumerate(systems), enumerate(plots)):
        nrow = 3*nsys
        ax = fig.add_subplot(gs[nrow:nrow+2, nplot])
        ratio_ax = fig.add_subplot(gs[nrow+2, nplot])

        for obs, subobs, label, cmap in subplots:
            factor = 5 if obs == 'dNch_deta' else 1
            color = getattr(plt.cm, cmap)(.6)

            x = model.map_data[system][obs][subobs]['x']
            y = model.map_data[system][obs][subobs]['Y'] * factor

            ax.plot(x, y, color=color, lw=.5)

            ax.text(
                x[-1] + 2.5,
                model.map_data[system][obs][subobs]['Y'][-1] * factor,
                label,
                color=darken(color), ha='left', va='center'
            )

            try:
                dset = expt.data[system][obs][subobs]
            except KeyError:
                continue

            x = dset['x']
            yexp = dset['y'] * factor
            yerr = dset['yerr']

            ax.errorbar(
                x, yexp, yerr=yerr.get('stat'), fmt='o', ms=1.7,
                capsize=0, color='.25', zorder=1000
            )

            yerrsys = yerr.get('sys', yerr.get('sum'))
            ax.fill_between(
                x, yexp - yerrsys, yexp + yerrsys,
                color='.9', zorder=-10
            )

            ratio_ax.plot(x, y/yexp, color=color)

        if title == 'Yields':
            ax.set_yscale('log')
            ax.minorticks_off()
        else:
            auto_ticks(ax, 'y', nbins=4, minor=2)

        if ax.is_first_row():
            ax.set_title(title)
        elif ratio_ax.is_last_row():
            ratio_ax.set_xlabel('Centrality %')

        if ax.is_last_col():
            proj, energy = parse_system(system)
            ax.text(
                1.07, 0, '{} {:.2f} TeV'.format('+'.join(proj), energy/1000),
                transform=ax.transAxes, ha='left', va='bottom',
                size=plt.rcParams['axes.titlesize'], rotation=-90
            )

        ax.set_ylabel(ylabel)
        ax.set_ylim({'mean_pT': (0, 1.75), 'vnk': (0, .12)}.get(obs, ylim))

        ratio_ax.axhline(1, lw=.5, color='0.5', zorder=-100)
        ratio_ax.axhspan(0.9, 1.1, color='0.95', zorder=-200)
        ratio_ax.text(
            ratio_ax.get_xlim()[1], .9, '±10%',
            color='.6', zorder=-50,
            ha='right', va='bottom',
            size=plt.rcParams['xtick.labelsize']
        )

        ratio_ax.set_ylim(0.8, 1.2)
        ratio_ax.set_yticks(np.arange(80, 121, 20)/100)
        ratio_ax.set_ylabel('Ratio')

    set_tight(fig, w_pad=1, rect=[0, 0, .97, 1])


def format_ci(samples, ci=.9):
    """
    Compute the median and a credible interval for an array of samples and
    return a TeX-formatted string.

    """
    cil, cih = mcmc.credible_interval(samples, ci=ci)
    median = np.median(samples)
    ul = median - cil
    uh = cih - median

    # decide precision for formatting numbers
    # this is NOT general but it works for the present data
    if abs(median) < .2 and ul < .02:
        precision = 3
    elif abs(median) < 1:
        precision = 2
    else:
        precision = 1

    fmt = str(precision).join(['{:#.', 'f}'])

    return ''.join([
        '$', fmt.format(median),
        '_{-', fmt.format(ul), '}',
        '^{+', fmt.format(uh), '}$'
    ])


def _posterior(
        params=None, ignore=None,
        scale=1, padr=.99, padt=.98,
        cmap=None
):
    """
    Triangle plot of posterior marginal and joint distributions.

    """
    chain = mcmc.Chain()

    if params is None and ignore is None:
        params = set(chain.keys)
    elif params is not None:
        params = set(params)
    elif ignore is not None:
        params = set(chain.keys) - set(ignore)

    keys, labels, ranges = map(list, zip(*(
        i for i in zip(chain.keys, chain.labels, chain.range)
        if i[0] in params
    )))
    ndim = len(params)

    data = chain.load(*keys).T

    cmap = plt.get_cmap(cmap)
    cmap.set_bad('white')

    line_color = cmap(.8)
    fill_color = cmap(.5, alpha=.1)

    fig, axes = plt.subplots(
        nrows=ndim, ncols=ndim,
        sharex='col', sharey='row',
        figsize=2*(scale*fullheight,)
    )

    for ax, d, lim in zip(axes.diagonal(), data, ranges):
        counts, edges = np.histogram(d, bins=50, range=lim)
        x = (edges[1:] + edges[:-1]) / 2
        y = .85 * (lim[1] - lim[0]) * counts / counts.max() + lim[0]
        # smooth histogram with monotonic cubic interpolation
        interp = PchipInterpolator(x, y)
        x = np.linspace(x[0], x[-1], 10*x.size)
        y = interp(x)
        ax.plot(x, y, lw=.5, color=line_color)
        ax.fill_between(x, lim[0], y, color=fill_color, zorder=-10)

        ax.set_xlim(lim)
        ax.set_ylim(lim)

        ticks = [lim[0], (lim[0] + lim[1])/2, lim[1]]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        ax.annotate(
            format_ci(d), (.62, .92), xycoords='axes fraction',
            ha='center', va='bottom', fontsize=4.5
        )

    for ny, nx in zip(*np.tril_indices_from(axes, k=-1)):
        axes[ny][nx].hist2d(
            data[nx], data[ny], bins=100,
            range=(ranges[nx], ranges[ny]),
            cmap=cmap, cmin=1
        )
        axes[nx][ny].set_axis_off()

    for n, label in enumerate(labels):
        for ax, xy in [(axes[-1, n], 'x'), (axes[n, 0], 'y')]:
            getattr(ax, 'set_{}label'.format(xy))(
                label.replace(r'\ [', '$\n$['), fontdict=dict(size=4)
            )
            ticklabels = getattr(ax, 'get_{}ticklabels'.format(xy))()
            for t in ticklabels:
                t.set_fontsize(3)
                if (
                        scale / ndim < .13 and
                        xy == 'x' and
                        len(str(sum(ranges[n])/2)) > 4
                ):
                    t.set_rotation(30)
            if xy == 'x':
                ticklabels[0].set_horizontalalignment('left')
                ticklabels[-1].set_horizontalalignment('right')
            else:
                ticklabels[0].set_verticalalignment('bottom')
                ticklabels[-1].set_verticalalignment('top')

    set_tight(fig, pad=.05, h_pad=.3, w_pad=.3, rect=[0., 0., padr, padt])


@plot
def posterior():
    _posterior(
        ignore={'norm {}'.format(s) for s in systems} | {'dmin3', 'etas_hrg'}
    )


@plot
def posterior_withnorm():
    _posterior(scale=1.2, ignore={'dmin3', 'etas_hrg'})


@plot
def posterior_shear():
    _posterior(
        scale=.35, padt=.96, padr=1.,
        params={'etas_min', 'etas_slope', 'etas_curv'}
    )


@plot
def posterior_bulk():
    _posterior(
        scale=.3, padt=.96, padr=1.,
        params={'zetas_max', 'zetas_width'}
    )


@plot
def posterior_p():
    """
    Distribution of trento p parameter with annotations for other models.

    """
    plt.figure(figsize=(.65*textwidth, .25*textwidth))
    ax = plt.axes()

    data = mcmc.Chain().load('trento_p').ravel()

    counts, edges = np.histogram(data, bins=50)
    x = (edges[1:] + edges[:-1]) / 2
    y = counts / counts.max()
    interp = PchipInterpolator(x, y)
    x = np.linspace(x[0], x[-1], 10*x.size)
    y = interp(x)
    ax.plot(x, y, color=plt.cm.Blues(0.8))
    ax.fill_between(x, y, color=plt.cm.Blues(0.15), zorder=-10)

    ax.set_xlabel('$p$')

    for spine in ax.spines.values():
        spine.set_visible(False)

    for label, x, err in [
            ('KLN', -.67, .01),
            ('EKRT /\nIP-Glasma', 0, .1),
            ('Wounded\nnucleon', 1, None),
    ]:
        args = ([x], [0], 'o') if err is None else ([x - err, x + err], [0, 0])
        ax.plot(*args, lw=4, ms=4, color=offblack, alpha=.58, clip_on=False)

        if label.startswith('EKRT'):
            x -= .275

        ax.text(x, .05, label, va='bottom', ha='center')

    ax.text(.1, .8, format_ci(data))
    ax.set_xticks(np.arange(-10, 11, 5)/10)
    ax.set_xticks(np.arange(-75, 76, 50)/100, minor=True)

    for t in ax.get_xticklabels():
        t.set_y(-.03)

    xm = 1.2
    ax.set_xlim(-xm, xm)
    ax.add_artist(
        patches.FancyArrowPatch(
            (-xm, 0), (xm, 0),
            linewidth=.6,
            arrowstyle=patches.ArrowStyle.CurveFilledAB(
                head_length=3, head_width=1.5
            ),
            facecolor=offblack, edgecolor=offblack,
            clip_on=False, zorder=100
        )
    )

    ax.set_yticks([])
    ax.set_ylim(0, 1.01*y.max())

    set_tight(pad=0)


region_style = dict(color='.93', zorder=-100)
Tc = .154


def _region_shear(mode='full', scale=.6):
    """
    Estimate of the temperature dependence of shear viscosity eta/s.

    """
    plt.figure(figsize=(scale*textwidth, scale*aspect*textwidth))
    ax = plt.axes()

    def etas(T, m=0, s=0, c=0):
        return m + s*(T - Tc)*(T/Tc)**c

    chain = mcmc.Chain()

    rangedict = dict(zip(chain.keys, chain.range))
    ekeys = ['etas_' + k for k in ['min', 'slope', 'curv']]

    T = np.linspace(Tc, .3, 100)

    prior = ax.fill_between(
        T, etas(T, *(rangedict[k][1] for k in ekeys)),
        **region_style
    )

    ax.set_xlim(xmin=.15)
    ax.set_ylim(0, .6)
    ax.set_xticks(np.arange(150, 301, 50)/1000)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    auto_ticks(ax, 'y', minor=2)

    ax.set_xlabel('Temperature [GeV]')
    ax.set_ylabel(r'$\eta/s$')

    if mode == 'empty':
        return

    if mode == 'examples':
        for args in [
                (.05, 1.0, -1),
                (.10, 1.7, 0),
                (.15, 2.0, 1),
        ]:
            ax.plot(T, etas(T, *args), color=plt.cm.Blues(.7))
        return

    eparams = chain.load(*ekeys).T
    intervals = np.array([
        mcmc.credible_interval(etas(t, *eparams))
        for t in T
    ]).T

    band = ax.fill_between(T, *intervals, color=plt.cm.Blues(.32))

    ax.plot(T, np.full_like(T, 1/(4*np.pi)), color='.6')
    ax.text(.299, .07, r'KSS bound $1/4\pi$', va='top', ha='right', color='.4')

    median, = ax.plot(
        T, etas(T, *map(np.median, eparams)),
        color=plt.cm.Blues(.77)
    )

    ax.legend(*zip(*[
        (prior, 'Prior range'),
        (median, 'Posterior median'),
        (band, '90% credible region'),
    ]), loc='upper left', bbox_to_anchor=(0, 1.03))


@plot
def region_shear():
    _region_shear()


@plot
def region_shear_empty():
    _region_shear('empty')


@plot
def region_shear_examples():
    _region_shear('examples', scale=.5)


def _region_bulk(mode='full', scale=.6):
    """
    Estimate of the temperature dependence of bulk viscosity zeta/s.

    """
    plt.figure(figsize=(scale*textwidth, scale*aspect*textwidth))
    ax = plt.axes()

    def zetas(T, zetas_max=0, zetas_width=1):
        return zetas_max / (1 + ((T - Tc)/zetas_width)**2)

    chain = mcmc.Chain()

    keys, ranges = map(list, zip(*(
        i for i in zip(chain.keys, chain.range)
        if i[0].startswith('zetas')
    )))

    T = Tc*np.linspace(.5, 1.5, 1000)

    maxdict = {k: r[1] for k, r in zip(keys, ranges)}
    ax.fill_between(
        T, zetas(T, **maxdict),
        label='Prior range',
        **region_style
    )

    ax.set_xlim(T[0], T[-1])
    ax.set_ylim(0, 1.05*maxdict['zetas_max'])
    auto_ticks(ax, minor=2)

    ax.set_xlabel('Temperature [GeV]')
    ax.set_ylabel(r'$\zeta/s$')

    if mode == 'empty':
        return

    if mode == 'examples':
        for args in [
                (.025, .01),
                (.050, .03),
                (.075, .05),
        ]:
            ax.plot(T, zetas(T, *args), color=plt.cm.Blues(.7))
        return

    # use a Gaussian mixture model to classify zeta/s parameters
    samples = chain.load(*keys, thin=10)
    gmm = GaussianMixture(n_components=3, covariance_type='full').fit(samples)
    labels = gmm.predict(samples)

    for n in range(gmm.n_components):
        params = dict(zip(
            keys,
            (mcmc.credible_interval(s)[1] for s in samples[labels == n].T)
        ))

        if params['zetas_max'] > .05:
            cmap = 'Blues'
        elif params['zetas_width'] > .03:
            cmap = 'Greens'
        else:
            cmap = 'Oranges'

        curve = zetas(T, **params)
        color = getattr(plt.cm, cmap)(.65)

        ax.plot(T, curve, color=color, zorder=-10)
        ax.fill_between(T, curve, color=color, alpha=.1, zorder=-20)

    ax.legend(loc='upper left')


@plot
def region_bulk():
    _region_bulk()


@plot
def region_bulk_empty():
    _region_bulk('empty')


@plot
def region_bulk_examples():
    _region_bulk('examples', scale=.5)


@plot
def flow_corr():
    """
    Symmetric cumulants SC(m, n) at the MAP point compared to experiment.

    """
    plots, width_ratios = zip(*[
        (('sc_central', 1e-7), 2),
        (('sc', 2.9e-6), 3),
    ])

    def label(*mn):
        return r'$\mathrm{{SC}}({}, {})$'.format(*mn)

    fig, axes = plt.subplots(
        figsize=(textwidth, .42*textwidth),
        ncols=len(plots), gridspec_kw=dict(width_ratios=width_ratios)
    )

    cmapx_normal = .7
    cmapx_pred = .5
    dashes_pred = [3, 2]

    for (obs, ylim), ax in zip(plots, axes):
        for (mn, cmap), sys in itertools.product(
                [
                    ((4, 2), 'Blues'),
                    ((3, 2), 'Oranges'),
                ],
                systems
        ):
            x = model.map_data[sys][obs][mn]['x']
            y = model.map_data[sys][obs][mn]['Y']

            pred = obs not in expt.data[sys]
            cmapx = cmapx_pred if pred else cmapx_normal

            kwargs = {}

            if pred:
                kwargs.update(dashes=dashes_pred)

            if ax.is_last_col():
                if not pred:
                    kwargs.update(label=label(*mn))
            else:
                fmt = '{:.2f} TeV'
                if pred:
                    fmt += ' (prediction)'
                lbl = fmt.format(parse_system(sys)[1]/1000)
                if not any(l.get_label() == lbl for l in ax.get_lines()):
                    ax.add_line(lines.Line2D(
                        [], [], color=plt.cm.Greys(cmapx),
                        label=lbl, **kwargs
                    ))

            ax.plot(
                x, y, lw=.75,
                color=getattr(plt.cm, cmap)(cmapx),
                **kwargs
            )

            if pred:
                continue

            x = expt.data[sys][obs][mn]['x']
            y = expt.data[sys][obs][mn]['y']
            yerr = expt.data[sys][obs][mn]['yerr']

            ax.errorbar(
                x, y, yerr=yerr['stat'],
                fmt='o', ms=2, capsize=0, color='.25', zorder=100
            )

            ax.fill_between(
                x, y - yerr['sys'], y + yerr['sys'],
                color='.9', zorder=-10
            )

        ax.axhline(
            0, color='.75', lw=plt.rcParams['xtick.major.width'],
            zorder=-100
        )

        ax.set_xlabel('Centrality %')
        ax.set_ylim(-ylim, ylim)

        auto_ticks(ax, 'y', nbins=6, minor=2)

        if ax.is_first_col():
            ax.set_ylabel(label('m', 'n'))

        ax.legend(loc='upper left')

        ax.set_title(dict(
            sc_central='Most central collisions',
            sc='Minimum bias'
        )[obs])


@plot
def flow_extra():
    """
    vn{2} in central bins and v2{4}.

    """
    plots, width_ratios = zip(*[
        (('vnk_central', 'Central two-particle cumulants', r'$v_n\{2\}$'), 2),
        (('vnk', 'Four-particle cumulants', r'$v_2\{4\}$'), 3),
    ])

    fig, axes = plt.subplots(
        figsize=(textwidth, .42*textwidth),
        ncols=len(plots), gridspec_kw=dict(width_ratios=width_ratios)
    )

    cmaps = {2: plt.cm.GnBu, 3: plt.cm.Purples}

    for (obs, title, ylabel), ax in zip(plots, axes):
        for sys, (cmapx, dashes, fmt) in zip(
                systems, [
                    (.7, (None, None), 'o'),
                    (.6, (3, 2), 's'),
                ]
        ):
            syslabel = '{:.2f} TeV'.format(parse_system(sys)[1]/1000)
            for subobs, dset in model.map_data[sys][obs].items():
                x = dset['x']
                y = dset['Y']

                ax.plot(
                    x, y,
                    color=cmaps[subobs](cmapx), dashes=dashes,
                    label='Model ' + syslabel
                )

                try:
                    dset = expt.data[sys][obs][subobs]
                except KeyError:
                    continue

                x = dset['x']
                y = dset['y']
                yerr = dset['yerr']

                ax.errorbar(
                    x, y, yerr=yerr['stat'],
                    fmt=fmt, ms=2.2, capsize=0, color='.25', zorder=100,
                    label='ALICE ' + syslabel
                )

                ax.fill_between(
                    x, y - yerr['sys'], y + yerr['sys'],
                    color='.9', zorder=-10
                )

                if obs == 'vnk_central':
                    ax.text(
                        x[-1] + .15, y[-1], '$v_{}$'.format(subobs),
                        color=cmaps[subobs](.99), ha='left', va='center'
                    )

        auto_ticks(ax, 'y', minor=2)
        ax.set_xlim(0, dset['cent'][-1][1])

        ax.set_xlabel('Centrality %')
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    ax.legend(loc='lower right')


@plot
def design():
    """
    Projection of a LH design into two dimensions.

    """
    fig = plt.figure(figsize=(.5*textwidth, .5*textwidth))
    ratio = 5
    gs = plt.GridSpec(ratio + 1, ratio + 1)

    ax_j = fig.add_subplot(gs[1:, :-1])
    ax_x = fig.add_subplot(gs[0, :-1], sharex=ax_j)
    ax_y = fig.add_subplot(gs[1:, -1], sharey=ax_j)

    d = Design(systems[0])

    keys = ('etas_min', 'etas_slope')
    indices = tuple(d.keys.index(k) for k in keys)

    x, y = (d.array[:, i] for i in indices)
    ax_j.plot(x, y, 'o', color=plt.cm.Blues(0.75), mec='white', mew=.3)

    hist_kw = dict(bins=30, color=plt.cm.Blues(0.4), edgecolor='white', lw=.5)
    ax_x.hist(x, **hist_kw)
    ax_y.hist(y, orientation='horizontal', **hist_kw)

    for ax in fig.axes:
        ax.tick_params(top='off', right='off')
        spines = ['top', 'right']
        if ax is ax_x:
            spines += ['left']
        elif ax is ax_y:
            spines += ['bottom']
        for spine in spines:
            ax.spines[spine].set_visible(False)
        for ax_name in 'xaxis', 'yaxis':
            getattr(ax, ax_name).set_ticks_position('none')

    auto_ticks(ax_j)

    for ax in ax_x, ax_y:
        ax.tick_params(labelbottom='off', labelleft='off')

    for i, xy in zip(indices, 'xy'):
        for f, l in [('lim', d.range), ('label', d.labels)]:
            getattr(ax_j, 'set_{}{}'.format(xy, f))(l[i])


@plot
def gp():
    """
    Conditioning a Gaussian process.

    """
    fig, axes = plt.subplots(
        figsize=(.45*textwidth, .85*textheight),
        nrows=2, sharex='col'
    )

    def dummy_optimizer(obj_func, initial_theta, bounds):
        return initial_theta, 0.

    gp = GPR(1.*kernels.RBF(.8), optimizer=dummy_optimizer)

    def sample_y(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return gp.sample_y(*args, **kwargs)

    x = np.linspace(0, 5, 1000)
    X = x[:, np.newaxis]

    x_train = np.linspace(.5, 4.5, 4)
    X_train = x_train[:, np.newaxis]

    for title, ax in zip(['Random functions', 'Conditioned on data'], axes):
        if title.startswith('Conditioned'):
            y = sample_y(X_train, random_state=23158).squeeze()
            y -= .5*(y.max() + y.min())
            gp.fit(X_train, y)
            training_data, = plt.plot(x_train, y, 'o', color='.3', zorder=50)

        for s, c in zip(
                sample_y(X, n_samples=4, random_state=34576).T,
                ['Blues', 'Greens', 'Oranges', 'Purples']
        ):
            ax.plot(x, s, color=getattr(plt.cm, c)(.6))

        mean, std = gp.predict(X, return_std=True)
        std = ax.fill_between(x, mean - std, mean + std, color='.92')
        mean, = ax.plot(x, mean, color='.42', dashes=(3.5, 1.5))

        ax.set_ylim(-2, 2)
        ax.set_ylabel('Output')
        auto_ticks(ax)

        ax.set_title(title, y=.9)

    ax.set_xlabel('Input')
    ax.legend(*zip(*[
        (mean, 'Mean prediction'),
        (std, 'Uncertainty'),
        (training_data, 'Training data'),
    ]), loc='lower left')

    set_tight(fig, h_pad=1)


@plot
def pca():
    fig = plt.figure(figsize=(.45*textwidth, .45*textwidth))
    ratio = 5
    gs = plt.GridSpec(ratio + 1, ratio + 1)

    ax_j = fig.add_subplot(gs[1:, :-1])
    ax_x = fig.add_subplot(gs[0, :-1], sharex=ax_j)
    ax_y = fig.add_subplot(gs[1:, -1], sharey=ax_j)

    x, y = (
        model.data['PbPb2760'][obs][subobs]['Y'][:, 3]
        for obs, subobs in [('dN_dy', 'pion'), ('vnk', (2, 2))]
    )
    xlabel = r'$dN_{\pi^\pm}/dy$'
    ylabel = r'$v_2\{2\}$'
    xlim = 0, 1500
    ylim = 0, 0.15

    cmap = plt.cm.Blues

    ax_j.plot(x, y, 'o', color=cmap(.75), mec='white', mew=.25, zorder=10)

    for d, ax, orientation in [(x, ax_x, 'vertical'), (y, ax_y, 'horizontal')]:
        ax.hist(
            d, bins=20,
            orientation=orientation, color=cmap(.4), edgecolor='white'
        )

    xy = np.column_stack([x, y])
    xymean = xy.mean(axis=0)
    xystd = xy.std(axis=0)
    xy -= xymean
    xy /= xystd
    pca = PCA().fit(xy)
    pc = (
        7 * xystd *
        pca.explained_variance_ratio_[:, np.newaxis] *
        pca.components_
    )

    for w, p in zip(pca.explained_variance_ratio_, pc):
        if np.all(p < 0):
            p *= -1
        ax_j.annotate(
            '', xymean + p, xymean, zorder=20,
            arrowprops=dict(
                arrowstyle='->', shrinkA=0, shrinkB=0,
                color=offblack, lw=.7
            )
        )
        ax_j.text(
            *(xymean + p + (.8, .002)*np.sign(p)), s='{:.0f}%'.format(100*w),
            color=offblack, ha='center', va='top' if p[1] < 0 else 'bottom',
            zorder=20
        )

    for ax in fig.axes:
        ax.tick_params(top='off', right='off')
        spines = ['top', 'right']
        if ax is ax_x:
            spines += ['left']
        elif ax is ax_y:
            spines += ['bottom']
        for spine in spines:
            ax.spines[spine].set_visible(False)
        for ax_name in 'xaxis', 'yaxis':
            getattr(ax, ax_name).set_ticks_position('none')

    for ax in ax_x, ax_y:
        ax.tick_params(labelbottom='off', labelleft='off')

    auto_ticks(ax_j)

    ax_j.set_xlim(xlim)
    ax_j.set_ylim(ylim)

    ax_j.set_xlabel(xlabel)
    ax_j.set_ylabel(ylabel)

    set_tight(pad=.1, h_pad=.3, w_pad=.3)


@plot
def trento_events():
    """
    Random trento events.

    """
    fig, axes = plt.subplots(
        nrows=3, sharex='col',
        figsize=(.28*textwidth, .85*textheight)
    )

    xymax = 8.
    xyr = [-xymax, xymax]

    with tempfile.NamedTemporaryFile(suffix='.hdf') as t:
        subprocess.run((
            'trento Pb Pb {} --quiet --b-max 12 '
            '--grid-max {} --grid-step .1 '
            '--random-seed 6347321 --output {}'
        ).format(axes.size, xymax, t.name).split())

        with h5py.File(t.name, 'r') as f:
            for dset, ax in zip(f.values(), axes):
                ax.pcolorfast(xyr, xyr, np.array(dset), cmap=plt.cm.Blues)
                ax.set_aspect('equal')
                for xy in ['x', 'y']:
                    getattr(ax, 'set_{}ticks'.format(xy))([-5, 0, 5])

    axes[-1].set_xlabel('$x$ [fm]')
    axes[1].set_ylabel('$y$ [fm]')

    set_tight(fig, h_pad=.5)


default_system = 'PbPb2760'


@plot
def diag_pca(system=default_system):
    """
    Diagnostic: histograms of principal components and scatterplots of pairs.

    """
    Y = [g.y_train_ for g in emulators[system].gps]
    n = len(Y)
    ymax = np.ceil(max(np.fabs(y).max() for y in Y))
    lim = (-ymax, ymax)

    fig, axes = plt.subplots(nrows=n, ncols=n, figsize=2*(n,))

    for y, ax in zip(Y, axes.diagonal()):
        ax.hist(y, bins=30)
        ax.set_xlim(lim)

    for ny, nx in zip(*np.tril_indices_from(axes, k=-1)):
        ax = axes[ny][nx]
        ax.scatter(Y[nx], Y[ny])
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        axes[nx][ny].set_axis_off()

    for i in range(n):
        label = 'PC {}'.format(i)
        axes[-1][i].set_xlabel(label)
        axes[i][0].set_ylabel(label)


@plot
def diag_emu(system=default_system):
    """
    Diagnostic: plots of each principal component vs each input parameter,
    overlaid by emulator predictions at several points in design space.

    """
    gps = emulators[system].gps
    nrows = len(gps)
    ncols = gps[0].X_train_.shape[1]

    w = 1.8
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(ncols*w, .8*nrows*w)
    )

    ymax = np.ceil(max(np.fabs(g.y_train_).max() for g in gps))
    ylim = (-ymax, ymax)

    design = Design(system)

    for ny, (gp, row) in enumerate(zip(gps, axes)):
        y = gp.y_train_

        for nx, (x, label, xlim, ax) in enumerate(zip(
                gp.X_train_.T, design.labels, design.range, row
        )):
            ax.plot(x, y, 'o', ms=.8, color='.75', zorder=10)

            x = np.linspace(xlim[0], xlim[1], 100)
            X = np.empty((x.size, ncols))

            for k, r in enumerate([.2, .5, .8]):
                X[:] = r*design.min + (1 - r)*design.max
                X[:, nx] = x
                mean, std = gp.predict(X, return_std=True)

                color = plt.cm.tab10(k)
                ax.plot(x, mean, lw=.2, color=color, zorder=30)
                ax.fill_between(
                    x, mean - std, mean + std,
                    lw=0, color=color, alpha=.3, zorder=20
                )

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            ax.set_xlabel(label)
            ax.set_ylabel('PC {}'.format(ny))


if __name__ == '__main__':
    import argparse

    choices = list(plot_functions)

    def arg_to_plot(arg):
        arg = Path(arg).stem
        if arg not in choices:
            raise argparse.ArgumentTypeError(arg)
        return arg

    parser = argparse.ArgumentParser(description='generate plots')
    parser.add_argument(
        'plots', nargs='*', type=arg_to_plot, metavar='PLOT',
        help='{} (default: all)'.format(', '.join(choices).join('{}'))
    )
    args = parser.parse_args()

    if args.plots:
        for p in args.plots:
            plot_functions[p]()
    else:
        for f in plot_functions.values():
            f()
