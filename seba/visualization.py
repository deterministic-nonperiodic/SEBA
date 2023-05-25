import warnings
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import SymLogNorm
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import ScalarFormatter
from scipy.ndimage import gaussian_filter

from spectral_analysis import kappa_from_deg, kappa_from_lambda, lambda_from_deg
from tools import find_intersections

warnings.filterwarnings('ignore')
plt.style.use('default')

params = {'xtick.labelsize': 'medium',
          'ytick.labelsize': 'medium', 'text.usetex': True,
          'font.size': 14, 'legend.title_fontsize': 15,
          'font.family': 'serif', 'font.weight': 'normal'}

plt.rcParams.update(params)
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab10.colors)
plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

# load colormaps
with open('../data/cet_d13.cm', 'r') as cfile:
    cet_d13 = cfile.read().split('\n')

cet_bwg = LinearSegmentedColormap.from_list('BWG', cet_d13)

VARIABLE_KEYMAP = {
    'hke': r'$E_K$',
    'rke': r'$E_R$',
    'dke': r'$E_D$',
    'vke': r'$E_w$',
    'ape': r'$E_A$',
    'pi_hke': r'$\Pi_K$',
    'pi_dke': r'$\Pi_D$',
    'pi_rke': r'$\Pi_R$',
    'pi_ape': r'$\Pi_A$',
    'cdr': r'$\mathcal{C}_{D \rightarrow R}$',
    'cdr_w': r'$\mathcal{C}_{D \rightarrow R}^{\omega}$',
    'cdr_v': r'$\mathcal{C}_{D \rightarrow R}^{\zeta}$',
    'cdr_c': r'$\mathcal{C}_{D \rightarrow R}^{f}$',
    'cad': r'$\mathcal{C}_{A \rightarrow D}$',
    'vf_dke': r'$\mathcal{F}_{D\uparrow}$',
    'vfd_dke': r'$\partial_{p}\mathcal{F}_{D\uparrow}$',
    'vfd_tot': r'$\Delta\mathcal{F}_{\uparrow} |^{p_b}_{p_t}$',
    # r'$\mathcal{F}_{\uparrow}(p_b) - \mathcal{F}_{\uparrow}(p_t)$',
    'vf_ape': r'$\mathcal{F}_{A\uparrow}$',
    'vfd_ape': r'$\partial_{p}\mathcal{F}_{A\uparrow}$',
    'mf_uw': r'$\rho\overline{u^{\prime}w^{\prime}}$',
    'mf_vw': r'$\rho\overline{v^{\prime}w^{\prime}}$',
    'mf_gw': r'$\overline{u^{\prime}w^{\prime}}$ + $\overline{v^{\prime}w^{\prime}}$',
    'dvf_dl_dke': r'$\partial_{\kappa}(\partial_{p}\mathcal{F}_{D\uparrow})$',
    'dcdr_dl': r'$\partial_{\kappa} \mathcal{C}_{D \rightarrow R}$',
    'dis_rke': r'$\mathcal{D}_R$',
    'dis_dke': r'$\mathcal{D}_D$',
    'dis_hke': r'$\mathcal{D}_K$'
}

LINES_KEYMAP = {
    'hke': ('black', 'solid', 1.5),
    'rke': ('red', 'dashed', 1.5),
    'dke': ('green', 'dashed', 1.5),
    'vke': ('black', '-.', 1.5),
    'ape': ('navy', 'solid', 1.5),
    'pi_hke': ('red', 'solid', 2.0),
    'pi_dke': ('green', 'solid', 1.6),
    'pi_rke': ('red', 'solid', 1.6),
    'pi_ape': ('navy', 'solid', 2.0),
    'cdr': ('blue', 'dashed', 2.0),
    'cdr_w': ('black', '-.', 1.6),
    'cdr_v': ('red', '-.', 1.6),
    'cdr_c': ('green', '-.', 1.6),
    'cad': ('green', 'dashed', 2.0),
    'vfd_dke': ('magenta', '-.', 1.6),
    'vfd_tot': ('magenta', '-.', 1.6),
    'vfd_ape': ('blue', '-.', 1.6),
    'dis_rke': ('blue', '-.', 1.6),
    'dis_dke': ('blue', '-.', 1.6),
    'dis_hke': ('cyan', '-.', 1.6),
}


def minmax_scaler(x, feature_range=None, axis=0):
    x = np.moveaxis(x, axis, 0)

    rs_msg = "Unknown type for 'feature_range'. "
    rs_msg += "Expecting a 2-tuple or list containing the min/max of the resulting scaled data."

    if feature_range is None:
        feature_range = (0, 1)
    elif isinstance(feature_range, (list, tuple)):
        assert len(feature_range) == 2, ValueError(rs_msg)
        feature_range = sorted(feature_range)
    else:
        raise ValueError(rs_msg)

    scale = (feature_range[1] - feature_range[0]) / (np.nanmax(x, axis=0) - np.nanmin(x, axis=0))

    return np.moveaxis(feature_range[0] + scale * (x - np.nanmin(x, axis=0)), 0, axis)


def find_symlog_params(data):
    """
    Determines appropriate values of linscale and linthresh for use with a SymLogNorm color map.

    Parameters:
        data (ndarray): The data to be plotted.

    Returns:
        tuple: A tuple of two floats representing the values of linscale and linthresh.
    """

    # Compute the range of the data
    data_range = np.ptp(data)

    # Scale the data to the range [0, 1] using MinMaxScaler
    scaled_data = minmax_scaler(data.ravel(), feature_range=(0, 1))

    # Find the value of linthresh that separates the linear and logarithmic parts
    # We want the linear part to be as large as possible while still excluding the
    # top and bottom 5% of the data, which could skew the normalization
    linthresh = np.nanpercentile(scaled_data, [5, 95])
    linthresh = np.interp(0.5, [0, 1], linthresh)  # use the median value

    # Compute the scale factor for the logarithmic part of the normalization
    # We want the logarithmic part to span a decade or so
    if 0.0 <= data_range <= 1.0:
        linscale = 0.9 * data_range
    else:
        # Find the value of linscale that stretches the linear part to cover most of the range
        # We want the linear part to cover about half of the range
        qr_5 = np.nanpercentile(data, [5, 95]).ptp()  # / 2.0
        linscale = qr_5 / (linthresh * np.log10(data_range))

    abs_max = 0.65 * np.nanmax(abs(data))

    v_min = 0.0 if np.nanmin(data) > 0 else -abs_max
    v_max = 0.0 if np.nanmax(data) < 0 else abs_max

    return dict(linthresh=linthresh, linscale=linscale, vmin=v_min, vmax=v_max)


def mean_confidence_interval(data, confidence=0.95, axis=0):
    a = np.asanyarray(data)
    n = a.shape[axis]

    m, se = np.nanmean(a, axis=axis), stats.sem(a, axis=axis)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def transform_spectra(s):
    return mean_confidence_interval(np.reshape(s, (-1, s.shape[-1])), confidence=0.95, axis=0)


def spectra_base_figure(n_rows=1, n_cols=1, x_limits=None, y_limits=None, y_label=None,
                        lambda_lines=None, y_scale='log', base=10, ax_titles=None, aligned=True,
                        frame=True, truncation=None, figure_size=None, shared_ticks=True,
                        **figure_kwargs):
    """Creates a figure template for spectral line plots

    :param n_rows: number of rows in figure
    :param n_cols: number of columns in figure
    :param x_limits: limits of the horizontal axis
    :param y_limits: limits of the vertical axis
    :param y_label: label of the vertical axis
    :param y_scale: 'linear', 'log', 'symlog'
    :param base: base of the logarithmic axis
    :param frame: draw frame on ax titles
    :param lambda_lines: positions to draw vertical lines
    :param ax_titles: title to display on the left corner of each axis
    :param aligned:
    :param truncation: spectral truncation
    :param figure_size:
    :param shared_ticks: always share y/x ticks for multiple subplots.
    :param figure_kwargs: additional keyword arguments to pass to plt.figure()
    :return: fig, axes
    """
    if y_limits is None:
        y_limits = [1e-10, 1e2]

    if truncation is None:
        truncation = 1000
    elif isinstance(truncation, str):
        # truncation is given as grid resolution string, compute nlat
        truncation = 2 * int(truncation.split('n')[-1])
    else:
        truncation = int(truncation)

    if x_limits is None:
        x_limits = kappa_from_lambda([40e3, 1e-3 * lambda_from_deg(truncation)])

    if truncation > 1000:
        x_ticks = np.array([2, 20, 200, 2000])
    else:
        x_ticks = np.array([1, 10, 100, 1000])

    show_limit = True
    if lambda_lines is None:
        lambda_lines = [18., ]
        show_limit = False

    kappa_lines = kappa_from_lambda(np.asarray(lambda_lines))

    if ax_titles is None:
        pass
    elif isinstance(ax_titles, str):
        ax_titles = n_rows * n_cols * [ax_titles, ]
    else:
        ax_titles = list(ax_titles)

    if figure_size is None:
        figure_size = (5.5, 5.5)
    elif isinstance(figure_size, (tuple, list)):
        pass
    else:
        reduced_size = 0.92 * float(figure_size)
        aspect = reduced_size if n_cols > 2 else figure_size
        figure_size = (aspect * n_cols, figure_size * n_rows)

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
                             figsize=figure_size, gridspec_kw={'wspace': 0.01},
                             constrained_layout=True, **figure_kwargs)

    if n_rows * n_cols == 1:
        axes = np.array([axes, ])

    for m, ax in enumerate(axes.flatten()):

        # axes frame limits
        ax.set_xscale('log', base=base)
        ax.set_yscale(y_scale)

        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)

        # axes title as annotation
        if ax_titles is not None:
            at = AnchoredText(ax_titles[m], prop=dict(size=15), frameon=frame, loc='upper right')
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax.add_artist(at)

        # draw vertical reference lines
        if show_limit:

            for lambda_line in lambda_lines:
                kappa_line = kappa_from_lambda(lambda_line)
                ax.axvline(x=kappa_line, ymin=0, ymax=1,
                           color='gray', linewidth=0.8,
                           alpha=0.6, linestyle='solid')

                ax.annotate('{:d}km'.format(int(lambda_line)),
                            xy=(0.90, 0.98), xycoords='axes fraction',
                            color='black', fontsize=10, horizontalalignment='left',
                            verticalalignment='top')

            ax.axvspan(np.max(kappa_lines), x_limits[1], alpha=0.15, color='gray')

        # Align left ytick labels:
        if aligned:
            for label in ax.yaxis.get_ticklabels():
                label.set_horizontalalignment('left')
            ax.yaxis.set_tick_params(pad=30)

        # set y label only for left most panels
        if not (m % n_cols):
            ax.set_ylabel(y_label, fontsize=16)
        else:
            if shared_ticks:
                ax.axes.get_yaxis().set_visible(False)

        if m >= n_cols * (n_rows - 1):  # lower boundary only
            ax.set_xlabel('wavenumber', fontsize=14, labelpad=4)
        else:
            if shared_ticks:
                ax.axes.get_xaxis().set_visible(False)

        # set lower x ticks
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xticks(1e3 * kappa_from_deg(x_ticks))
        ax.set_xticklabels(x_ticks)

        if m < n_cols:  # upper boundary only
            secax = ax.secondary_xaxis('top', functions=(kappa_from_lambda, kappa_from_lambda))
            secax.xaxis.set_major_formatter(ScalarFormatter())

            secax.set_xlabel(r'wavelength / $km$', fontsize=14, labelpad=6)

    return fig, axes


def reference_slopes(ax, k_scales, magnitude, slopes, name='horizontal'):
    # Function for adding reference slopes to spectral plot

    if name == 'horizontal':
        s_scales = [r"$l^{{{0}}}$".format(s) for s in slopes]
    else:
        s_scales = [r"$m^{{{0}}}$".format(s) for s in slopes]

    n_slopes = [float(reduce(lambda x, y: float(x) / float(y), s.split('/')))
                for s in slopes]
    # Plot reference slopes
    for k, m, sn, ss in zip(k_scales, magnitude, n_slopes, s_scales):
        y_scale = m * k ** sn
        ax.plot(k, y_scale, lw=1.2, ls='dashed', color='gray')

        scale_pos = np.argmax(y_scale)

        x_scale_pos = k[scale_pos]
        y_scale_pos = y_scale[scale_pos]
        x_text_pos = -10. if scale_pos == y_scale.size else -4

        ax.annotate(ss, xy=(x_scale_pos, y_scale_pos), xycoords='data',
                    xytext=(x_text_pos, 16.), textcoords='offset points',
                    color='k', horizontalalignment='left',
                    verticalalignment='top', fontsize=13)


def _parse_variable(varname):
    # parser variable names and return the corresponding line properties

    if varname in VARIABLE_KEYMAP:
        color, style, width = LINES_KEYMAP[varname]
        label = VARIABLE_KEYMAP[varname]
    elif "+" in varname:
        color, style, width = 'black', 'solid', 2
        label = VARIABLE_KEYMAP[varname.split('+')[0]].split('_')[0] + r'=$ '
        label += r' $+$ '.join([VARIABLE_KEYMAP[name] for name in varname.split('+')])
    else:
        raise ValueError(f'Unknown variable name: {varname}.')

    return dict(label=label, lw=width, color=color, linestyle=style)


def fluxes_slices_by_models(dataset, model=None, variables=None,
                            x_limits=None, y_limits=None,
                            cmap=None, fig_name=None):
    if variables is None:
        variables = list(dataset.data_vars)

    ax_titles = [VARIABLE_KEYMAP[name] for name in variables]

    if y_limits is None:
        y_limits = [1000., 100.]

    if cmap is None:
        cmap = cet_bwg

    # get coordinates
    level = 1e-2 * dataset['level'].values
    kappa = 1e3 * dataset['kappa'].values

    if 'time' in dataset.dims:
        dataset = dataset.mean(dim='time')

    # -----------------------------------------------------------------------------------
    # Visualization of Kinetic energy and Available potential energy
    # -----------------------------------------------------------------------------------
    n = len(variables)
    cols = 2 if not n % 2 else n
    rows = max(1, n // cols)

    fig, axes = spectra_base_figure(n_rows=rows, n_cols=cols, x_limits=x_limits,
                                    y_limits=y_limits, figure_size=4.6,
                                    y_label=r'Pressure / hPa',
                                    y_scale='linear', ax_titles=ax_titles,
                                    frame=True, truncation=kappa.size)
    axes = axes.ravel()

    cs_levels = 100

    for m, (ax, varname) in enumerate(zip(axes, variables)):
        spectra = 1e3 * dataset[varname].values

        smoothed_data = gaussian_filter(spectra, 0.25)

        # Create plots:
        cs = ax.contourf(kappa, level, smoothed_data,
                         cmap=cmap, levels=cs_levels,
                         norm=SymLogNorm(**find_symlog_params(spectra)))

        ax.contour(kappa, level, smoothed_data,
                   color='black', linewidths=0.6, linestyles='solid',
                   levels=[0, ], alpha=0.8)

        ax.set_ylim(y_limits)

        # add a colorbar to all axes
        cb = plt.colorbar(cs, ax=ax, orientation='vertical', pad=0.001, format="%.2f")
        cb.ax.set_title(r"($\times 10^{3}~W/m^{2}$)", fontsize=11, loc='center', pad=10)
        cb.ax.tick_params(labelsize=12)

    if model is not None:
        at = AnchoredText(model.upper(), prop=dict(size=15), frameon=True, loc='upper left')
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        axes[0].add_artist(at)

    plt.show()
    if fig_name is not None:
        fig.savefig(fig_name, dpi=300)

    plt.close(fig)


def energy_spectra_by_levels(dataset, model=None, variables=None, layers=None,
                             x_limits=None, y_limits=None, fig_name=None):
    if model is None:
        model = ''

    if variables is None:
        variables = ['hke', 'rke', 'dke', 'vke', 'ape']

    if layers is None:
        layers = {'': [10e2, 1000e2]}

    if y_limits is None:
        y_limits = {name: [7e-5, 3e7] for name in layers.keys()}

    # get coordinates
    kappa = 1e3 * dataset['kappa'].values
    level = 1e-2 * dataset['level'].values
    level_range = np.int32([level.max(), level.min()])

    degree_max = 2 * kappa.size - 1
    degree_eff = int(degree_max / 3)
    # mesoscale reference slope goes from 100 to the effective wavenumber
    mesoscale = 1e3 * kappa_from_deg([100, degree_eff])

    x_scales = [kappa_from_lambda(np.linspace(3500, 650, 2)), mesoscale]
    scale_st = ['-3', '-5/3']
    scale_mg = [2.5e-4, 0.1]

    # -----------------------------------------------------------------------------------
    # Visualization of Kinetic energy and Available potential energy
    # -----------------------------------------------------------------------------------
    n = len(layers)
    cols = 2 if not n % 2 else n
    rows = max(1, n // cols)

    legend_cols = 1 + int(len(variables) > 3)

    fig, axes = spectra_base_figure(n_rows=rows, n_cols=cols, shared_ticks=False,
                                    figure_size=(cols * 6.2, rows * 5.8),
                                    x_limits=x_limits, y_limits=None, aligned=True,
                                    y_label=r'Energy density / $J ~ m^{-2}$',
                                    y_scale='log', ax_titles=None,
                                    frame=False, truncation=degree_max)

    axes = axes.ravel()
    indicator = 'abcdefghijklmn'

    for m, (layer, prange) in enumerate(layers.items()):

        data = dataset.integrate_range(coord_range=prange)

        for varname in variables:
            # compute mean and standard deviation
            spectra, spectra_sel, spectra_seu = mean_confidence_interval(data[varname].values,
                                                                         confidence=0.95, axis=0)

            # plot confidence interval
            axes[m].fill_between(kappa, spectra_seu, spectra_sel,
                                 color='gray', interpolate=False, alpha=0.1)

            axes[m].plot(kappa, spectra, **_parse_variable(varname))

        # crossing scales: intersections between rotational and divergent kinetic energies
        kappa_c, spectra_c = find_intersections(kappa,
                                                data['rke'].mean('time').values,
                                                data['dke'].mean('time').values,
                                                direction='decreasing')

        # take median of multiple crossings
        kappa_c = np.median(kappa_c)
        spectra_c = np.median(spectra_c)

        # vertical lines denoting crossing scales
        if not np.isnan(kappa_c).all():
            axes[m].vlines(x=kappa_c, ymin=0., ymax=spectra_c,
                           color='black', linewidth=0.8,
                           linestyle='dashed', alpha=0.6)

            kappa_c_pos = [2, -60][kappa_c > kappa_from_lambda(40)]

            # Scale is defined as half-wavelength
            axes[m].annotate(r'$L_{c}\sim$' + '{:d} km'.format(int(kappa_from_lambda(kappa_c))),
                             xy=(kappa_c, y_limits[layer][0]), xycoords='data',
                             xytext=(kappa_c_pos, 20.), textcoords='offset points',
                             color='black', fontsize=9, horizontalalignment='left',
                             verticalalignment='top')

        # plot reference slopes
        reference_slopes(axes[m], x_scales, scale_mg, scale_st)

        axes[m].set_ylim(*y_limits[layer])

        layer_lim = level_range if prange is None else (1e-2 * np.sort(prange)[::-1]).astype(int)
        axes[m].legend(title=r"{} ({:3d} - {:3d} hPa)".format(layer, *layer_lim),
                       loc='upper right', fontsize=14, labelspacing=0.4, borderaxespad=0.4,
                       ncol=legend_cols, frameon=False, columnspacing=2.5)

        art = AnchoredText(f"({indicator[m]})",
                           loc='lower left', prop=dict(size=20), frameon=False,
                           bbox_to_anchor=(-0.14, 0.98), bbox_transform=axes[m].transAxes)
        art.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        axes[m].add_artist(art)

    art = AnchoredText(model.upper(), prop=dict(size=20), frameon=False, loc='lower left',
                       bbox_to_anchor=(-0.02, 0.86), bbox_transform=axes[0].transAxes)
    art.patch.set_boxstyle("round,pad=-0.3, rounding_size=0.2")
    axes[0].add_artist(art)

    plt.show()

    if fig_name is not None:
        fig.savefig(fig_name, dpi=300)

    plt.close(fig)


def fluxes_spectra_by_levels(dataset, model=None, variables=None, layers=None,
                             show_injection=False, x_limits=None, y_limits=None,
                             fig_name=None):
    if model is None:
        model = ''

    if variables is None:
        variables = list(dataset.data_vars)

    if layers is None:
        layers = {'': [20e2, 950e2]}

    if y_limits is None:
        y_limits = {name: [-1.5, 1.5] for name in layers.keys()}

    # get coordinates
    kappa = 1e3 * dataset['kappa'].values
    level = 1e-2 * dataset['level'].values
    level_range = np.int32([level.max(), level.min()])

    degree_max = 2 * kappa.size - 1
    # -----------------------------------------------------------------------------------
    # Visualization of Kinetic energy and Available potential energy
    # -----------------------------------------------------------------------------------
    n = len(layers)
    cols = 2 if not n % 2 else n
    rows = max(1, n // cols)

    legend_cols = 1 + int(len(variables) >= 6)

    fig, axes = spectra_base_figure(n_rows=rows, n_cols=cols,
                                    figure_size=(cols * 6.4, rows * 5.8),
                                    x_limits=x_limits, y_limits=None, aligned=False,
                                    y_label=r'Cumulative energy flux / $W ~ m^{-2}$',
                                    y_scale='linear', ax_titles=None, frame=False,
                                    truncation=degree_max, shared_ticks=False)
    axes = axes.ravel()
    indicator = 'abcdefghijklmn'

    for m, (layer, prange) in enumerate(layers.items()):

        data = dataset.integrate_range(coord_range=prange).mean(dim='time')

        for varname in variables:
            spectra = np.sum([data[name] for name in varname.split('+')], axis=0)
            if 'pi_ape' in varname:
                #  correction for ape spectral flux due to small deviations
                #  from zero due to numerical precision.
                spectra[:2] = 0.0

            axes[m].plot(kappa, spectra, **_parse_variable(varname))

        axes[m].axhline(y=0.0, xmin=0, xmax=1, color='gray',
                        linewidth=1.2, linestyle='dashed', alpha=0.5)

        if show_injection:
            # compute energy injection scale (PI_HKE crosses zero with positive slope)
            kappa_in, _ = find_intersections(kappa, data['pi_hke'].values, 0.0,
                                             direction='increasing')

            # select the closest point to the largest wavenumber
            if not np.isscalar(kappa_in):
                kappa_in = kappa_in[kappa_in < 0.9 * kappa.max()]
                if len(kappa_in) > 0:
                    kappa_in = kappa_in[-1]

            # vertical lines denoting crossing scales
            if not np.isnan(kappa_in).all():
                axes[m].vlines(x=kappa_in, ymin=y_limits[layer][0], ymax=0.0,
                               color='black', linewidth=0.8,
                               linestyle='dashed', alpha=0.6)

                kappa_in_pos = [2, -60][kappa_in > kappa_from_lambda(40)]

                # Scale is defined as half-wavelength
                axes[m].annotate(r'$L_{in}\sim$' + '{:d} km'.format(
                    int(kappa_from_lambda(kappa_in))),
                                 xy=(kappa_in, y_limits[layer][0]), xycoords='data',
                                 xytext=(kappa_in_pos, 20.), textcoords='offset points',
                                 color='black', fontsize=9, horizontalalignment='left',
                                 verticalalignment='top')

        axes[m].set_ylim(*y_limits[layer])

        layer_lim = level_range if prange is None else (1e-2 * np.sort(prange)[::-1]).astype(int)
        legend = axes[m].legend(title="{} ({:3d} - {:3d} hPa)".format(layer, *layer_lim),
                                loc='upper right', fontsize=14, labelspacing=0.5, borderaxespad=0.4,
                                ncol=legend_cols, frameon=False, columnspacing=1.4)
        legend.set_in_layout(False)

        art = AnchoredText(f"({indicator[m]})",
                           loc='lower left', prop=dict(size=20), frameon=False,
                           bbox_to_anchor=(-0.1, 1.0), bbox_transform=axes[m].transAxes)
        art.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        axes[m].add_artist(art)

    art = AnchoredText(model.upper(), prop=dict(size=20), frameon=False, loc='lower left',
                       bbox_to_anchor=(-0.02, 0.86), bbox_transform=axes[0].transAxes)
    art.patch.set_boxstyle("round,pad=-0.3, rounding_size=0.2")
    axes[0].add_artist(art)

    plt.show()

    if fig_name is not None:
        fig.savefig(fig_name, dpi=300)

    plt.close(fig)
