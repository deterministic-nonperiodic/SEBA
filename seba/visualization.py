import warnings
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import ScalarFormatter, NullFormatter, MultipleLocator

import constants as cn
from spectral_analysis import kappa_from_deg, kappa_from_lambda, lambda_from_deg, deg_from_lambda
from tools import find_intersections

warnings.filterwarnings('ignore')
plt.style.use('default')

params = {'xtick.labelsize': 'medium', 'ytick.labelsize': 'medium', 'font.size': 16,
          'legend.title_fontsize': 15, 'legend.fontsize': 15,
          'font.family': 'serif', 'text.usetex': True}

plt.rcParams.update(params)
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab10.colors)
plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

# load colormaps
with open('../data/cet_d13.cm', 'r') as cfile:
    cet_d13 = cfile.read().split('\n')

cet_bwg = LinearSegmentedColormap.from_list('cet_bwg', cet_d13)

ind_offset = (-0.046, 0.976)
# (-0.085, 0.965)

VARIABLE_KEYMAP = {
    'hke': r'$E_K$',
    'rke': r'$E_R$',
    'dke': r'$E_D$',
    'vke': r'$E_w$',
    'hke_IG': r'$E_{IG}$',
    'hke_RO': r'$E_{RO}$',
    'vke_IG': r'$E_{IG_w}$',
    'ape': r'$E_A$',
    'ape_IG': r'$E^{IG}_{A}$',
    'ape_RO': r'$E^{RO}_{A}$',
    'pi_hke': r'$\Pi_K$',
    'pi_nke': r'$\Pi_N$',
    'pi_dke': r'$\Pi_D$',
    'pi_rke': r'$\Pi_R$',
    'pi_hke_RO': r'$\Pi_{RO}$',
    'pi_hke_IG': r'$\Pi_{IG}$',
    'pi_hke_CF': r'$\Pi_{CF}$',
    'pi_dke_RO': r'$\Pi^{RO}_D$',
    'pi_dke_IG': r'$\Pi^{IG}_D$',
    'pi_dke_CF': r'$\Pi^{CF}_D$',
    'pi_rke_RO': r'$\Pi^{RO}_R$',
    'pi_rke_IG': r'$\Pi^{IG}_R$',
    'pi_rke_CF': r'$\Pi^{CF}_R$',
    'pi_ape': r'$\Pi_A$',
    'pi_ape_IG': r'$\Pi_A$',
    'cdr': r'$\mathcal{C}_{D \rightarrow R}$',
    'cdr_IG': r'$\mathcal{C}^{IG}_{D \rightarrow R}$',
    'cdr_RO': r'$\mathcal{C}^{RO}_{D \rightarrow R}$',
    'cdr_CF': r'$\mathcal{C}_{IG \rightarrow RO}$',
    'cdr_w': r'$\mathcal{C}_{D \rightarrow R}^{\omega}$',
    'cdr_v': r'$\mathcal{C}_{D \rightarrow R}^{\zeta}$',
    'cdr_c': r'$\mathcal{C}_{D \rightarrow R}^{f}$',
    'cad': r'$\mathcal{C}_{A \rightarrow D}$',
    'cad_IG': r'$\mathcal{C}_{A \rightarrow IG}$',
    'vfd': r'$\Delta^{p_b}_{p_t}\mathcal{F}_{\uparrow}$',
    'vf': r'$\Delta^{p_b}_{p_t}\mathcal{F}_{\uparrow}$',
    'vfb': r'$\mathcal{F}_{\uparrow}(p_b)$',
    'vft': r'$\mathcal{F}_{\uparrow}(p_t)$',
    'vf_dke': r'$\mathcal{F}_{D\uparrow}$',
    'vfd_dke': r'$\Delta^{p_b}_{p_t}\mathcal{F}_{D\uparrow}$',
    'vfd_dke_IG': r'$\Delta^{p_b}_{p_t}\mathcal{F}_{IG\uparrow}$',
    'vf_ape': r'$\mathcal{F}_{A\uparrow}$',
    'vfd_ape': r'$\Delta^{p_b}_{p_t}\mathcal{F}_{A\uparrow}$',
    'mf_uw': r'$\rho\overline{u^{\prime}w^{\prime}}$',
    'mf_vw': r'$\rho\overline{v^{\prime}w^{\prime}}$',
    'mf_gw': r'$\overline{u^{\prime}w^{\prime}}$ + $\overline{v^{\prime}w^{\prime}}$',
    'dvf_dl_dke': r'$\partial_{\kappa}(\partial_{p}\mathcal{F}_{D\uparrow})$',
    'dcdr_dl': r'$\partial_{\kappa} \mathcal{C}_{D \rightarrow R}$',
    'dis_rke': r'$\mathcal{D}_R$',
    'dis_dke': r'$\mathcal{D}_D$',
    'dis_hke': r'$\mathcal{D}_K$',
    'lc': r'$\Pi_L$',
    'pi_lke': r'$\Pi_L$',
}

m_lw = 2.
s_lw = 0.75 * m_lw

LINES_KEYMAP = {
    'hke': ('black', 'solid', m_lw),
    'rke': ('red', 'solid', s_lw),
    'dke': ('green', 'solid', s_lw),
    'vke': ('black', '-.', m_lw),
    'hke_RO': ('red', '-.', m_lw),
    'hke_IG': ('teal', '-.', m_lw),
    'vke_IG': ('black', '-.', s_lw),
    'ape': ('darkblue', 'solid', m_lw),
    'ape_IG': ('darkblue', 'dashed', m_lw),
    'ape_RO': ('darkblue', '-.', m_lw),
    'pi_hke': ('red', 'solid', m_lw),
    'pi_nke': ('red', 'solid', m_lw),
    'pi_dke': ('teal', 'solid', s_lw),
    'pi_rke': ('red', 'solid', s_lw),
    'pi_ape': ('mediumblue', 'solid', m_lw),
    'pi_ape_IG': ('mediumblue', 'solid', s_lw),
    'pi_hke_RO': ('red', '-.', s_lw),
    'pi_hke_IG': ('teal', '-.', s_lw),
    'pi_hke_CF': ('black', '-.', s_lw),
    'pi_dke_RO': ('red', '-.', s_lw),
    'pi_dke_IG': ('teal', '-.', s_lw),
    'pi_dke_CF': ('black', '-.', s_lw),
    'pi_rke_RO': ('red', '-.', s_lw),
    'pi_rke_IG': ('teal', '-.', s_lw),
    'pi_rke_CF': ('black', '-.', s_lw),
    'cdr': ('rebeccapurple', '-.', m_lw),
    'cdr_IG': ('teal', '-.', s_lw),
    'cdr_RO': ('red', '-.', s_lw),
    'cdr_CF': ('darkviolet', '-.', m_lw),
    'cdr_w': ('black', '-.', s_lw),
    'cdr_v': ('red', '-.', s_lw),
    'cdr_c': ('blue', '-.', s_lw),
    'cad': ('green', 'dashed', m_lw),
    'cad_IG': ('green', '-.', s_lw),
    'vfd_dke': ('magenta', 'solid', s_lw),
    'vf_dke': ('magenta', 'dashed', s_lw),
    'vfb': ('mediumorchid', '--', m_lw),
    'vft': ('mediumorchid', ':', m_lw),
    'vfd_dke_IG': ('magenta', 'solid', s_lw),
    'vfd': ('magenta', 'solid', m_lw),
    'vfd_ape': ('mediumblue', '-.', s_lw),
    'vf_ape': ('mediumblue', '-.', s_lw),
    'dis_rke': ('red', '-.', s_lw),
    'dis_dke': ('green', '-.', s_lw),
    'dis_hke': ('cyan', '-.', s_lw),
    'lc': ('darkcyan', 'solid', m_lw),
    'pi_lke': ('darkcyan', 'solid', m_lw),
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
    linthresh = np.nanpercentile(scaled_data, [10, 90])
    linthresh = np.interp(0.5, [0, 1], linthresh)  # use the median value

    # Compute the scale factor for the logarithmic part of the normalization
    # We want the logarithmic part to span a decade or so
    if 0.0 <= data_range <= 1.0:
        linscale = 0.9 * data_range
    else:
        # Find the value of linscale that stretches the linear part to cover most of the range
        # We want the linear part to cover about half of the range
        qr_5 = np.nanpercentile(data, [10, 90]).ptp()  # / 2.0
        linscale = qr_5 / (linthresh * np.log10(data_range))

    abs_max = min(0.65 * np.nanmax(abs(data)), 0.5)

    v_min = 0.0 if np.nanmin(data) > 0 else -abs_max
    v_max = 0.0 if np.nanmax(data) < 0 else abs_max

    return dict(linthresh=linthresh, linscale=linscale, vmin=v_min, vmax=v_max)


def mean_confidence_interval(data, confidence=0.95, axis=0):
    a = np.asanyarray(data)
    n = a.shape[axis]

    m, se = np.nanmean(a, axis=axis), stats.sem(a, axis=axis)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, h


def spectra_base_figure(n_rows=1, n_cols=1, x_limits=None, y_limits=None, y_label=None,
                        lambda_lines=None, y_scale='log', base=10, ax_titles=None, aligned=True,
                        frame=True, truncation=None, figure_size=None, shared_ticks=None,
                        n_ticks=None, **figure_kwargs):
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
    :param n_ticks:
    :return: fig, axes
    """
    if shared_ticks is None:
        shared_xticks = shared_yticks = True
    elif hasattr(shared_ticks, '__len__'):
        shared_xticks, shared_yticks = shared_ticks
    else:
        shared_xticks = shared_yticks = shared_ticks

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
        x_limits = 1e3 * kappa_from_lambda([lambda_from_deg(0.9),
                                            lambda_from_deg(truncation)])

    if truncation > 1024:
        x_ticks = np.array([2, 20, 200, 2000])
    else:
        x_ticks = np.array([1, 10, 100, 1000])

    label_fonts_size = 15

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

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figure_size,
                             constrained_layout=True, **figure_kwargs)

    if n_rows * n_cols == 1:
        axes = np.array([axes, ])

    for m, ax in enumerate(axes.flatten()):

        # axes frame limits
        ax.set_xscale('log', base=base)
        # ax.set(xscale='log')
        ax.set_yscale(y_scale)

        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)

        # reduce the number of ticks of y-axis
        if y_scale != 'log' and n_ticks is not None:
            ax.yaxis.set_major_locator(MultipleLocator(n_ticks))

        # axes title as annotation
        if ax_titles is not None:
            at = AnchoredText(ax_titles[m], prop=dict(size=label_fonts_size),
                              frameon=frame, loc='upper right')
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
                            xy=(0.92, 0.06), xycoords='axes fraction',
                            color='black', fontsize=12, horizontalalignment='left',
                            verticalalignment='top')

            ax.axvspan(np.max(kappa_lines), x_limits[1], alpha=0.15, color='gray')

        # Align left ytick labels:
        if aligned:
            for label in ax.yaxis.get_ticklabels():
                label.set_horizontalalignment('left')
            ax.yaxis.set_tick_params(pad=30)

        # set y label only for left most panels
        if not (m % n_cols):
            ax.set_ylabel(y_label, fontsize=label_fonts_size)
        else:
            if shared_yticks:
                ax.axes.get_yaxis().set_visible(False)

        if m >= n_cols * (n_rows - 1):  # lower boundary only
            ax.set_xlabel('wavenumber', fontsize=label_fonts_size, labelpad=3)
        else:
            if shared_xticks:
                ax.axes.get_xaxis().set_visible(False)

        # set lower x ticks
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xticks(1e3 * kappa_from_deg(x_ticks))
        ax.set_xticklabels(x_ticks)

        if m < n_cols:  # upper boundary only
            secax = ax.secondary_xaxis('top', functions=(kappa_from_lambda, kappa_from_lambda))
            secax.xaxis.set_major_formatter(ScalarFormatter())
            secax.set_xlabel(r'wavelength / km', fontsize=label_fonts_size, labelpad=6)

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
        x_text_pos = -10. if scale_pos == y_scale.size else -1

        ax.annotate(ss, xy=(x_scale_pos, y_scale_pos), xycoords='data',
                    xytext=(x_text_pos, 16.), textcoords='offset points',
                    color='k', horizontalalignment='left',
                    verticalalignment='top', fontsize=14)


def _parse_variable(varname):
    # parser variable names and return the corresponding line properties

    if varname in VARIABLE_KEYMAP:
        color, style, width = LINES_KEYMAP[varname]
        label = VARIABLE_KEYMAP[varname]
    elif "+" in varname:
        color, style, width = 'black', 'solid', 2
        suffix = r'_{K} = $ ' if 'dke' in varname and 'rke' in varname else r'=$ '
        label = VARIABLE_KEYMAP[varname.split('+')[0]].split('_')[0] + suffix
        label += r' $+$ '.join([VARIABLE_KEYMAP[name] for name in varname.split('+')])
    else:
        raise ValueError(f'Unknown variable name: {varname}.')

    return dict(label=label, lw=width, color=color, linestyle=style)


def visualize_sections(dataset, model=None, variables=None, truncation=None,
                       x_limits=None, y_limits=None, share_cbar=False, cmap=None,
                       show_crossing=False, show_injection=False, start_index=''):
    if variables is None:
        variables = list(dataset.data_vars)

    ax_titles = [VARIABLE_KEYMAP[name] for name in variables]

    if 'vfd_dke' in variables:
        ax_titles[variables.index('vfd_dke')] = r'$\partial_p\mathcal{F}_{D\uparrow}$'

    if 'vfd_ape' in variables:
        ax_titles[variables.index('vfd_ape')] = r'$\partial_p\mathcal{F}_{A\uparrow}$'

    if 'vfd' in variables:
        ax_titles[variables.index('vfd')] = r'$\partial_p\mathcal{F}_{\uparrow}$'

    if y_limits is None:
        y_limits = [1000., 100.]

    if cmap is None:
        cmap = cet_bwg

    # get coordinates
    level = 1e-2 * dataset['level'].values
    kappa = 1e3 * dataset['kappa'].values

    if truncation is None:
        truncation = kappa.size

    if x_limits is None:
        x_limits = 1e3 * kappa_from_deg([1, truncation])

    if 'time' in dataset.dims:
        dataset = dataset.mean(dim='time')

    reference_ticks = [50, 100, 200, 400, 600, 800, 1000]
    last_tick = np.searchsorted(reference_ticks, min(y_limits), side='left')
    y_ticks = reference_ticks[last_tick:]

    degree_max = min(2 * kappa.size - 1, deg_from_lambda(9.9e3))
    degree_eff = max(512, int(degree_max / 4))

    kappa_c = 0
    if show_crossing:
        # crossing scales: intersections between rotational and divergent kinetic energies
        kappa_c = np.array([
            # np.median()
            find_intersections(kappa[1:-1], dke[1:-1], rke[1:-1], direction='increasing')[0][0]
            for rke, dke in zip(dataset['rke'].values, dataset['dke'].values)]
        )
        # vertical lines denoting crossing scales
        kappa_c[kappa_c == 0] = np.nan

    kappa_in = 0.0
    if show_injection:
        # compute energy injection scale (PI_HKE crosses zero with positive slope)
        kappa_in = np.array([
            find_intersections(kappa[1:-1], pi_hke[1:-1], 0.0, direction='increasing')[0][-1]
            for pi_hke in dataset['pi_hke'].values]
        )

        # select the closest point to the largest wavenumber
        kappa_se = np.logical_and(kappa_in < 1e3 * kappa_from_deg(degree_eff),
                                  kappa_in > 1e3 * kappa_from_deg(1))
        kappa_in[kappa_se] = np.nan

    layers = [450.0, 250.0, 50.]
    # -----------------------------------------------------------------------------------
    # Visualization of Kinetic energy and Available potential energy
    # -----------------------------------------------------------------------------------
    n = len(variables)
    cols = 2 if not n % 2 else n
    rows = max(1, n // cols)
    cols, rows = max(rows, cols), min(rows, cols)

    fig, axes = spectra_base_figure(n_rows=rows, n_cols=cols, x_limits=x_limits,
                                    y_limits=y_limits, figure_size=5.2,
                                    y_label='Pressure / hPa', aligned=False,
                                    y_scale='log', ax_titles=ax_titles,
                                    frame=True, truncation=truncation)
    axes = axes.ravel()
    indicator = 'abcdefghijklmn'
    indicator = indicator[indicator.find(start_index):]

    # create normalizers for shared colorbar
    dataset = 1e4 * dataset

    cs_levels = 50
    cs_levels = np.linspace(-2.0, 2.0, cs_levels)
    cb_ticks = [-2., -1.5, -1., -0.5, 0., 0.5, 1., 1.5, 2.]

    cs_reference = []
    for m, (ax, varname) in enumerate(zip(axes, variables)):
        spectra = dataset[varname].values

        cs = ax.contourf(kappa, level, spectra, extend='both',
                         cmap=cmap, levels=cs_levels)

        # This is a fix for the white lines between contour levels
        for c in cs.collections:
            c.set_edgecolor("face")

        cs_reference.append(cs)

        ax.contour(kappa, level, spectra, levels=[0, ], color='black', interpolate=True,
                   linewidths=1.0, linestyles='solid', alpha=0.8)

        # # plot flux vectors
        # if 'vfd' in varname:
        #     vf = dataset[varname.replace('d', '')].differentiate('kappa')
        #     ax.quiver(kappa, level, np.zeros_like(vf), -kappa * vf, pivot='middle')

        # add a colorbar to all axes
        if not share_cbar:
            cb = plt.colorbar(cs, ax=ax, orientation='vertical', pad=0.001,
                              format="%.2f", extend='both')
            cb.ax.set_title(r"($\times 10^{-4}W/m^{2}$)", fontsize=11, loc='center', pad=11)
            cb.ax.tick_params(labelsize=12)
            cb.ax.set_ticks(cb_ticks)
            cb.ax.set_ticklabels(cb_ticks)

        ax.hlines(layers, xmin=kappa.min(), xmax=kappa.max(),
                  color='gray', linewidth=1., linestyle='dashed', alpha=0.5)

        # add subplot indicator
        subplot_label = r'\textbf{{({})}}'.format(indicator[m])

        art = AnchoredText(subplot_label, loc='lower left',
                           prop=dict(size=18, weight="bold"),
                           frameon=False, bbox_to_anchor=(-0.085, 0.965),
                           bbox_transform=ax.transAxes)
        art.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(art)

        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks)
        ax.set_ylim(y_limits)

        if show_crossing:
            # crossing scales: intersections between rotational and divergent kinetic energies
            if not np.isnan(kappa_c).all():
                ax.plot(kappa_c[1:], level[1:], marker='.', color='black',
                        linestyle='--', linewidth=0.8, alpha=0.6)

        if show_injection:
            # crossing scales: intersections between rotational and divergent kinetic energies
            if not np.isnan(kappa_in).all():
                ax.plot(kappa_in, level, marker='.', color='red',
                        linestyle='--', linewidth=0.8, alpha=0.6)

    if model is not None:
        art = AnchoredText('-'.join(model.split('_')).upper(),
                           prop=dict(size=15), frameon=True, loc='lower left',
                           bbox_to_anchor=(0.0, 0.87), bbox_transform=axes[0].transAxes)

        art.patch.set_boxstyle("round,pad=0., rounding_size=0.2")
        axes[0].add_artist(art)

    if share_cbar:
        for r in range(rows):
            kwargs = dict(orientation='vertical', format="%.2f", pad=0.01,
                          extend='both', spacing='proportional')
            cb = fig.colorbar(cs_reference[r * cols], ax=axes[(r + 1) * cols - 1], **kwargs)
            cb.ax.set_title(r"[$ 10^{-4}~W/kg$]", fontsize=11, loc='center', pad=11)
            # cb.ax.set_ylabel(r"($\times 10^{-3}~W m^{-2}$)", fontsize=12, loc='center')
            cb.ax.tick_params(labelsize=13)
            cb.ax.set_yticks(cb_ticks)
            cb.ax.set_yticklabels(cb_ticks)

    return fig


def visualize_energy(dataset, model=None, variables=None, layers=None, x_limits=None):
    if variables is None:
        variables = ['hke', 'rke', 'dke', 'vke', 'ape']

    if layers is None:
        layers = {'': [50e2, 1000e2]}
    else:
        assert isinstance(layers, dict), ValueError('layers must be a dictionary')

    y_limits = {}
    for name, limits in layers.items():
        if np.size(limits) == 4:
            layers[name], y_limits[name] = limits
        elif np.size(limits) == 2:
            y_limits[name] = [7e-5, 4e7]
        else:
            raise ValueError('Wrong size for layers')

    # get coordinates
    kappa = 1e3 * dataset['kappa'].values
    level = 1e-2 * dataset['level'].values
    level_range = np.int32([level.max(), level.min()])

    degree_max = min(2 * kappa.size - 1, deg_from_lambda(9.9e3))
    degree_eff = max(512, int(degree_max / 4))
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

    if cols < 3:
        figure_size = (cols * 6., rows * 5.8)
    else:
        # square subplots for multiple columns
        figure_size = (cols * 5.8, rows * 5.8)

    # share y-ticks if the y limits are identical
    shared_yticks = len(y_limits) == 1 or np.allclose(*y_limits.values())

    fig, axes = spectra_base_figure(n_rows=rows, n_cols=cols, figure_size=figure_size,
                                    x_limits=x_limits, y_limits=None, aligned=True,
                                    y_label=r'Energy density / $J ~ m^{-2}$',
                                    y_scale='log', ax_titles=None, lambda_lines=[20, ],
                                    frame=False, truncation=degree_max,
                                    shared_ticks=(True, shared_yticks))

    axes = axes.ravel()
    indicator = 'abcdefghijklmn'

    for m, (layer, prange) in enumerate(layers.items()):

        data = dataset.integrate_range(coord_range=prange)

        for varname in variables:
            # compute mean and standard deviation
            spectra, ci = mean_confidence_interval(data[varname].values, confidence=0.95, axis=0)

            # plot confidence interval
            axes[m].fill_between(kappa, spectra + ci, spectra - ci,
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
        if not np.isnan(kappa_c).all() and np.all(kappa_c > 0):
            axes[m].vlines(x=kappa_c, ymin=0., ymax=spectra_c,
                           color='black', linewidth=0.8,
                           linestyle='dashed', alpha=0.6)

            kappa_c_pos = [2, -58][kappa_c > kappa_from_lambda(50)]

            # Scale is defined as half-wavelength
            axes[m].annotate(r'$L_{c}\sim$' + '{:d} km'.format(int(kappa_from_lambda(kappa_c))),
                             xy=(kappa_c, y_limits[layer][0]), xycoords='data',
                             xytext=(kappa_c_pos, 20.), textcoords='offset points',
                             color='black', fontsize=12, horizontalalignment='left',
                             verticalalignment='top')

        # plot reference slopes
        reference_slopes(axes[m], x_scales, scale_mg, scale_st)

        axes[m].set_ylim(*y_limits[layer])

        if model is not None:
            art = AnchoredText('-'.join(model.split('_')).upper(),
                               prop=dict(size=20), frameon=False, loc='lower left',
                               bbox_to_anchor=(-0.025, 0.87), bbox_transform=axes[m].transAxes)

            art.patch.set_boxstyle("round,pad=0., rounding_size=0.2")
            axes[m].add_artist(art)

        layer_lim = level_range if prange is None else (1e-2 * np.sort(prange)[::-1]).astype(int)
        layer_str = r"{} ($p_b\! =${:3d} hPa, $p_t\! =${:3d} hPa)"
        legend = axes[m].legend(title=layer_str.format(layer, *layer_lim),
                                loc='upper right', labelspacing=0.6, borderaxespad=0.4,
                                ncol=legend_cols, frameon=False, columnspacing=2.4)
        legend.set_in_layout(False)
        # noinspection PyProtectedMember
        legend._legend_box.align = "right"
        # legend._legend_box.sep = 12

        subplot_label = r'\textbf{{({})}}'.format(indicator[m])

        art = AnchoredText(subplot_label, loc='lower left',
                           prop=dict(size=20, fontweight='bold'), frameon=False,
                           bbox_to_anchor=ind_offset,
                           bbox_transform=axes[m].transAxes)
        art.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        axes[m].add_artist(art)

    return fig


def visualize_fluxes(dataset, model=None, variables=None, layers=None,
                     show_injection=False, x_limits=None):
    if variables is None:
        # exclude non-flux variables
        exclude_variables = ['hke', 'rke', 'dke', 'vke', 'ape', 'vf_ape', 'vf_dke']
        variables = [name for name in dataset.data_vars if name not in exclude_variables]

    if layers is None:
        layers = {'': [50e2, 1000e2]}
    else:
        assert isinstance(layers, dict), ValueError('layers must be a dictionary')

    y_limits = {}
    for name, limits in layers.items():
        if np.size(limits) == 4:
            layers[name], y_limits[name] = limits
        elif np.size(limits) == 2:
            y_limits[name] = [-1.5, 1.5]
        else:
            raise ValueError('Wrong size for layers')

    # get coordinates
    kappa = 1e3 * dataset['kappa'].values
    level = 1e-2 * dataset['level'].values
    level_range = np.int32([level.max(), level.min()])

    degree_max = min(2 * kappa.size - 1, deg_from_lambda(9.9e3))
    degree_eff = max(512, int(degree_max / 4))
    # -----------------------------------------------------------------------------------
    # Visualization of Kinetic energy and Available potential energy
    # -----------------------------------------------------------------------------------
    n = len(layers)
    cols = 2 if not n % 2 else n
    rows = max(1, n // cols)

    legend_cols = 1 + int(len(variables) >= 6)

    if cols < 3:
        figure_size = (cols * 6., rows * 5.8)
    else:
        # square subplots for multiple columns
        figure_size = (cols * 5.8, rows * 5.8)

    # share y-ticks if the y limits are identical
    shared_yticks = len(y_limits) == 1 or np.allclose(*y_limits.values())

    fig, axes = spectra_base_figure(n_rows=rows, n_cols=cols, figure_size=figure_size,
                                    x_limits=x_limits, y_limits=None, aligned=False,
                                    y_label=r'Cumulative energy flux / $W ~ m^{-2}$',
                                    y_scale='linear', ax_titles=None, frame=False,
                                    truncation=degree_max,
                                    shared_ticks=(True, shared_yticks))
    axes = axes.ravel()
    indicator = 'abcdefghijklmn'

    for m, (layer, prange) in enumerate(layers.items()):

        data = dataset.integrate_range(coord_range=prange)

        for varname in variables:
            # parse spectra from varname
            spectra = np.sum([data[name] for name in varname.split('+')], axis=0)

            spectra, ci = mean_confidence_interval(spectra, confidence=0.95, axis=0)

            # plot confidence interval
            axes[m].fill_between(kappa, spectra + ci, spectra - ci,
                                 color='gray', interpolate=False, alpha=0.1)

            # plot confidence interval
            kwargs = _parse_variable(varname)

            axes[m].plot(kappa, spectra, **kwargs)

        axes[m].axhline(y=0.0, xmin=0, xmax=1, color='gray', linewidth=1.,
                        linestyle='dashed', alpha=0.5)

        if show_injection:
            # compute energy injection scale (PI_HKE crosses zero with positive slope)
            kappa_in, _ = find_intersections(kappa, data['pi_nke'].mean(dim='time').values, 0.0,
                                             direction='increasing')

            # select the closest point to the largest wavenumber
            if not np.isscalar(kappa_in):
                kappa_se = np.logical_and(kappa_in < 1e3 * kappa_from_deg(degree_eff),
                                          kappa_in > 1e3 * kappa_from_deg(1))
                kappa_in = kappa_in[kappa_se]
                if len(kappa_in) > 0:
                    kappa_in = kappa_in[-1]

            # vertical lines denoting crossing scales
            if not np.isnan(kappa_in).all():
                axes[m].vlines(x=kappa_in, ymin=y_limits[layer][0], ymax=0.0,
                               color='black', linewidth=0.8,
                               linestyle='dashed', alpha=0.6)

                kappa_in_pos = [2, -60][kappa_in > kappa_from_lambda(60)]

                # Scale is defined as half-wavelength
                axes[m].annotate(r'$L_{in}\sim$' + '{:d} km'.format(
                    int(kappa_from_lambda(kappa_in))),
                                 xy=(kappa_in, y_limits[layer][0]), xycoords='data',
                                 xytext=(kappa_in_pos, 20.), textcoords='offset points',
                                 color='black', fontsize=12, horizontalalignment='left',
                                 verticalalignment='top')

        if model is not None:
            art = AnchoredText('-'.join(model.split('_')).upper(),
                               prop=dict(size=20), frameon=False, loc='lower left',
                               bbox_to_anchor=(-0.025, 0.87), bbox_transform=axes[m].transAxes)
            art.patch.set_boxstyle("round,pad=0., rounding_size=0.2")
            axes[m].add_artist(art)

        axes[m].set_ylim(*y_limits[layer])

        layer_lim = level_range if prange is None else (1e-2 * np.sort(prange)[::-1]).astype(int)
        layer_str = r"{} ($p_b\! =${:3d} hPa, $p_t\! =${:3d} hPa)"
        legend = axes[m].legend(title=layer_str.format(layer, *layer_lim),
                                loc='upper right', labelspacing=0.6, borderaxespad=0.4,
                                ncol=legend_cols, frameon=False, columnspacing=1.25)
        legend.set_in_layout(False)
        # noinspection PyProtectedMember
        legend._legend_box.align = "right"

        subplot_label = r'\textbf{{({})}}'.format(indicator[m])

        art = AnchoredText(subplot_label, loc='lower left',
                           prop=dict(size=18, fontweight='bold'),
                           frameon=False, bbox_to_anchor=ind_offset,
                           bbox_transform=axes[m].transAxes)
        art.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        axes[m].add_artist(art)

    return fig


def compare_model_fluxes(datasets, models=None, variables=None, layers=None, shared_ticks=True,
                         show_injection=False, x_limits=None, start_index='', y_label=None,
                         zoom=None, legend_cols=None, orientation='horizontal'):
    if models is None:
        models = ['', '']

    zoom = zoom or ''

    if y_label is None:
        y_label = r'Cumulative energy flux / $W ~ m^{-2}$'

    dataset = datasets[models[0]]

    if variables is None:
        # exclude non-flux variables
        exclude_variables = ['hke', 'rke', 'dke', 'vke', 'ape']
        variables = [name for name in dataset.data_vars if name not in exclude_variables]

    if layers is None:
        layer_limits = {'': [50e2, 1000e2]}
    else:
        assert isinstance(layers, dict), ValueError('layers must be a dictionary')
        layer_limits = layers.copy()

    y_limits = {}
    for name, limits in layer_limits.items():
        if np.size(limits) == 4:
            layer_limits[name], y_limits[name] = limits
        elif np.size(limits) == 2:
            y_limits[name] = [-1.5, 1.5]
        else:
            raise ValueError('Wrong size for layers')

    # get vertical coordinates
    level = 1e-2 * dataset['level'].values
    level_range = np.int32([level.max(), level.min()])

    degree_max = 2 * max([ds.kappa.size for ds in datasets.values()]) - 1
    degree_max = min(degree_max, deg_from_lambda(9.9e3))

    data_range = max([np.ptp(lim) for lim in y_limits.values()])

    n_ticks = [0.2, 0.4, 1, 1][np.searchsorted([1.2, 2., 4.], data_range)]

    # -----------------------------------------------------------------------------------
    # Visualization of Kinetic energy and Available potential energy
    # -----------------------------------------------------------------------------------
    nm = len(models)
    rows = max(1, len(layer_limits))
    cols = max(1, nm)

    legend_pos = 0  # cols - 1

    if orientation == 'vertical':
        rows, cols = cols, rows

    if legend_cols is None:
        legend_cols = 1 + int(len(variables) >= 6)

    if degree_max < 1024:
        legend_offset = (1, 1)
        lambda_lines = None
        inset_pos = 0.5
    else:
        legend_offset = (0.926, 1.0)
        lambda_lines = [20., ]
        inset_pos = 0.42

    if cols < 3:
        if rows > 1:
            figure_size = (cols * 6.8, rows * 5.8)
        else:
            figure_size = (cols * 6.2, rows * 5.8)
    else:
        # square subplots for multiple columns
        figure_size = (cols * 5.8, rows * 5.8)

    legend_fontsize = params['legend.fontsize']

    fig, axes = spectra_base_figure(n_rows=rows, n_cols=cols, figure_size=figure_size,
                                    lambda_lines=lambda_lines, x_limits=x_limits,
                                    y_limits=None, aligned=False, n_ticks=n_ticks, y_label=y_label,
                                    y_scale='linear', ax_titles=None, frame=False,
                                    truncation=degree_max, shared_ticks=(True, shared_ticks))

    axes = axes.ravel()

    indicator = 'abcdefghijklmn'
    indicator = indicator[indicator.find(start_index):]

    for l, (layer, prange) in enumerate(layer_limits.items()):
        for m, model in enumerate(models):

            ax = axes[nm * l + m]

            kappa = 1e3 * datasets[model]['kappa'].values
            degree_eff = int(kappa.size / 2 - 1)

            data = datasets[model].integrate_range(coord_range=prange)

            # create inset axis if needed
            if indicator[nm * l + m] in zoom:
                ax_inset = ax.inset_axes([inset_pos, 0.525, 0.465, 0.38])
                inset_limits = 1e3 * kappa_from_deg([10, 40])
                ax_inset.set_xlim(*inset_limits)
                ax_inset.set_ylim(-0.08, 0.08)

                ax_inset.xaxis.set_major_formatter(ScalarFormatter())
                ax_inset.set_xticks(1e3 * kappa_from_deg([10, 20, 30, 40]))
                ax_inset.set_xticklabels([10, 20, 30, 40])
                for tick in ax_inset.xaxis.get_major_ticks():
                    tick.label.set_fontsize(9)
                for tick in ax_inset.yaxis.get_major_ticks():
                    tick.label.set_fontsize(9)

                ax.indicate_inset_zoom(ax_inset, edgecolor="gray")
            else:
                ax_inset = None

            for varname in variables:
                # parse spectra from varname
                spectra = np.sum([data[name] for name in varname.split('+')], axis=0)

                spectra, ci = mean_confidence_interval(spectra, confidence=0.9, axis=0)

                # plot confidence interval
                kwargs = _parse_variable(varname)

                ax.fill_between(kappa, spectra + ci, spectra - ci,
                                color='gray', interpolate=True, alpha=0.125)

                ax.plot(kappa, spectra, **kwargs)

                if ax_inset is not None:
                    ax_inset.fill_between(kappa, spectra + ci, spectra - ci,
                                          color='gray', interpolate=True, alpha=0.1)
                    ax_inset.plot(kappa, spectra, **kwargs)

            ax.axhline(y=0.0, xmin=0, xmax=1, color='gray', linewidth=1.,
                       linestyle='dashed', alpha=0.5)
            if ax_inset is not None:
                ax_inset.axhline(y=0.0, xmin=0, xmax=1, color='gray', linewidth=1.,
                                 linestyle='dashed', alpha=0.5)

            if show_injection:
                # compute energy injection scale (PI_HKE crosses zero with positive slope)
                kappa_in, _ = find_intersections(kappa,
                                                 data['pi_nke'].mean(dim='time').values, 0.0,
                                                 direction='increasing')

                # select the closest point to the largest wavenumber
                if not np.isscalar(kappa_in):
                    kappa_se = np.logical_and(kappa_in < 1e3 * kappa_from_deg(degree_eff),
                                              kappa_in > 1e3 * kappa_from_deg(5))
                    kappa_in = kappa_in[kappa_se]
                    if len(kappa_in) > 0:
                        kappa_in = kappa_in[0]

                # vertical lines denoting crossing scales
                if not np.isnan(kappa_in).all():
                    ax.vlines(x=kappa_in, ymin=y_limits[layer][0], ymax=0.0,
                              color='black', linewidth=0.8,
                              linestyle='dashed', alpha=0.6)

                    kappa_in_pos = [2, -60][kappa_in > kappa_from_lambda(40)]

                    # Scale is defined as half-wavelength
                    ax.annotate(r'$L_{in}\sim$' + '{:d} km'.format(
                        int(kappa_from_lambda(kappa_in))),
                                xy=(kappa_in, y_limits[layer][0]), xycoords='data',
                                xytext=(kappa_in_pos, 20.), textcoords='offset points',
                                color='black', fontsize=12, horizontalalignment='left',
                                verticalalignment='top')

            if model is not None:
                art = AnchoredText('-'.join(model.split('_')).upper(),
                                   prop=dict(size=20), frameon=False, loc='lower left',
                                   bbox_to_anchor=(-0.025, 0.87), bbox_transform=ax.transAxes)
                art.patch.set_boxstyle("round,pad=0., rounding_size=0.2")

                if orientation == 'vertical':
                    if l == 0:
                        ax.add_artist(art)
                else:
                    ax.add_artist(art)

            ax.set_ylim(*y_limits[layer])

            # Set legend
            layer_lim = (1e-2 * np.sort(prange)[::-1]).astype(int)
            layer_lim = level_range if prange is None else layer_lim
            layer_str = r"{} ($p_b\! =${:3d} hPa, $p_t\! =${:3d} hPa)".format(layer, *layer_lim)
            if m == legend_pos and (indicator[nm * l + m] not in zoom):
                legend = ax.legend(title=layer_str, fontsize=0.95 * legend_fontsize,
                                   loc='upper right', labelspacing=0.6, borderaxespad=0.4,
                                   ncol=legend_cols, frameon=False, columnspacing=0.6,
                                   bbox_to_anchor=legend_offset)
                legend.set_in_layout(False)
                # noinspection PyProtectedMember
                legend._legend_box.align = "right"
            else:
                # Insert legend title as Anchored text if legend is hidden
                art = AnchoredText(layer_str,
                                   prop=dict(size=0.95 * legend_fontsize),
                                   frameon=False, loc='upper right',
                                   bbox_to_anchor=(legend_offset[0], 1.0065),
                                   bbox_transform=ax.transAxes)
                # art.patch.set_boxstyle("round,pad=0., rounding_size=0.2")
                ax.add_artist(art)

            subplot_label = r'\textbf{{({})}}'.format(indicator[nm * l + m])

            art = AnchoredText(subplot_label, loc='lower left',
                               prop=dict(size=18, fontweight='black'),
                               frameon=False, bbox_to_anchor=ind_offset,
                               bbox_transform=ax.transAxes)
            art.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax.add_artist(art)

    return fig


def compare_fluxes_by_model(datasets, models=None, variables=None, layers=None, lambda_lines=None,
                            show_injection=False, x_limits=None, start_index='', y_label=None,
                            zoom=None, legend_cols=None, interface_flux=False, shared_ticks=True):
    if models is None:
        models = ['', '']

    zoom = zoom or ''

    if y_label is None:
        y_label = r'Cumulative energy flux / $W ~ m^{-2}$'

    dataset = datasets[models[0]]

    if variables is None:
        # exclude non-flux variables
        exclude_variables = ['hke', 'rke', 'dke', 'vke', 'ape']
        variables = [name for name in dataset.data_vars if name not in exclude_variables]

    if layers is None:
        layer_limits = {'': [50e2, 1000e2]}
    else:
        assert isinstance(layers, dict), ValueError('layers must be a dictionary')
        layer_limits = layers.copy()

    y_limits = {}
    for name, limits in layer_limits.items():
        if np.size(limits) == 4:
            layer_limits[name], y_limits[name] = limits
        elif np.size(limits) == 2:
            y_limits[name] = [-1.5, 1.5]
        else:
            raise ValueError('Wrong size for layers')

    # get vertical coordinates
    level = 1e-2 * dataset['level'].values
    level_range = np.int32([level.max(), level.min()])

    degree_max = 2 * max([ds.kappa.size for ds in datasets.values()]) - 1
    degree_max = min(degree_max, deg_from_lambda(9.9e3))

    data_range = max([np.ptp(lim) for lim in y_limits.values()])

    n_ticks = [0.2, 0.4, 1, 1][np.searchsorted([1.2, 2., 4.], data_range)]

    # -----------------------------------------------------------------------------------
    # Visualization of Kinetic energy and Available potential energy
    # -----------------------------------------------------------------------------------
    cols = max(1, len(layer_limits))
    rows = max(1, len(models))

    if legend_cols is None:
        legend_cols = 1 + int(len(variables) >= 6)

        if interface_flux and any('vfd' in vn for vn in variables):
            legend_cols = max(2, legend_cols)

    if degree_max < 1024:
        legend_offset = (1, 1)
        inset_pos = 0.48
    else:
        legend_offset = (0.93, 1.0)
        if lambda_lines is None:
            lambda_lines = [20., ]
        inset_pos = 0.42

    if cols < 3:
        if rows > 1:
            figure_size = (cols * 6.8, rows * 5.8)
        else:
            figure_size = (cols * 6.2, rows * 5.8)
    else:
        # square subplots for multiple columns
        figure_size = (cols * 5.8, rows * 5.8)

    fig, axes = spectra_base_figure(n_rows=rows, n_cols=cols, figure_size=figure_size,
                                    lambda_lines=lambda_lines, x_limits=x_limits,
                                    y_limits=None, aligned=False, n_ticks=n_ticks, y_label=y_label,
                                    y_scale='linear', ax_titles=None, frame=False,
                                    truncation=min(degree_max, 4500),
                                    shared_ticks=(True, shared_ticks))

    axes = axes.ravel()

    indicator = 'abcdefghijklmn'
    indicator = indicator[indicator.find(start_index):]

    inset_limits = 1e3 * kappa_from_deg([[512, 2048], [10, 40]])
    inset_xticks = [[512, 1024, 2048], [10, 20, 30, 40]]
    for l, (layer, prange) in enumerate(layer_limits.items()):
        for m, model in enumerate(models):

            ax = axes[cols * m + l]

            kappa = 1e3 * datasets[model]['kappa'].values
            degree_eff = int(kappa.size / 2 - 1)

            data = datasets[model].integrate_range(coord_range=prange)

            index = indicator[cols * m + l]

            # create inset axis if needed
            if index in zoom:
                ax_inset = ax.inset_axes([inset_pos, 0.52, 0.465, 0.38])
                # inset_limits = 1e3 * kappa_from_deg([10, 40])
                ax_inset.set_xlim(*inset_limits[l])
                ax_inset.set_ylim(-0.08, 0.08)

                ax_inset.xaxis.set_major_formatter(ScalarFormatter())
                ax_inset.set_xticks(1e3 * kappa_from_deg(inset_xticks[l]))
                ax_inset.set_xticklabels(inset_xticks[l])
                for tick in ax_inset.xaxis.get_major_ticks():
                    tick.label.set_fontsize(9)
                for tick in ax_inset.yaxis.get_major_ticks():
                    tick.label.set_fontsize(9)

                ax.indicate_inset_zoom(ax_inset, edgecolor="gray")
            else:
                ax_inset = None

            spectra_layers = {}
            for varname in variables:
                # parse spectra from varname
                if 'vfd' not in varname:
                    spectra = np.sum([data[name] for name in varname.split('+')], axis=0)

                    spectra, ci = mean_confidence_interval(spectra, confidence=0.9, axis=0)

                    # plot confidence interval
                    kwargs = _parse_variable(varname)

                    ax.fill_between(kappa, spectra + ci, spectra - ci,
                                    color='gray', interpolate=True, alpha=0.125)

                    ax.plot(kappa, spectra, **kwargs)

                    if ax_inset is not None:
                        ax_inset.fill_between(kappa, spectra + ci, spectra - ci,
                                              color='gray', interpolate=True, alpha=0.1)
                        ax_inset.plot(kappa, spectra, **kwargs)
                else:
                    get_name = varname.replace('d_', '_') if '_' in varname else varname.replace(
                        'd', '')

                    for interface, level in zip(['vft', 'vfb'], sorted(prange)):
                        if get_name in datasets[model]:
                            spectra_in = datasets[model][get_name].sel(level=level,
                                                                       method='nearest')
                        else:
                            spectra_in = datasets[model][varname].sortby('level', ascending=False)
                            spectra_in = spectra_in.sel(level=slice(None, level)).integrate('level')
                            spectra_in -= datasets[model][varname].integrate('level')

                        spectra_layers[interface] = spectra_in / cn.g

                    # plot total vertical flux
                    spectra = spectra_layers['vfb'] - spectra_layers['vft']
                    spectra, ci = mean_confidence_interval(spectra, confidence=0.65, axis=0)

                    # plot confidence interval
                    kwargs = _parse_variable(varname)

                    ax.fill_between(kappa, spectra + ci, spectra - ci,
                                    color='gray', interpolate=True, alpha=0.125)

                    ax.plot(kappa, spectra, **kwargs)

                    if ax_inset is not None:
                        ax_inset.fill_between(kappa, spectra + ci, spectra - ci,
                                              color='gray', interpolate=True, alpha=0.1)
                        ax_inset.plot(kappa, spectra, **kwargs)

                    # show the bottom and top layer vertical fluxes
                    if interface_flux:
                        for interface, spectra_in in spectra_layers.items():
                            kwargs_in = _parse_variable(interface)

                            var_id = "_{"
                            if '_' in varname:
                                varname.split('_')[-1][0].upper()
                            kwargs_in.update(label=var_id.join(kwargs_in['label'].split('_{')))

                            ax.plot(kappa, spectra_in.mean('time'), **kwargs_in)

                            if ax_inset is not None:
                                ax_inset.plot(kappa, spectra_in.mean('time'), **kwargs_in)

            ax.axhline(y=0.0, xmin=0, xmax=1, color='gray', linewidth=1.,
                       linestyle='dashed', alpha=0.5)
            if ax_inset is not None:
                ax_inset.axhline(y=0.0, xmin=0, xmax=1, color='gray', linewidth=1.,
                                 linestyle='dashed', alpha=0.5)

            if show_injection:
                # compute energy injection scale (PI_HKE crosses zero with positive slope)
                kappa_in, _ = find_intersections(kappa,
                                                 data['pi_hke'].mean(dim='time').values, 0.0,
                                                 direction='increasing')

                # select the closest point to the largest wavenumber
                if not np.isscalar(kappa_in):
                    kappa_se = np.logical_and(kappa_in < 1e3 * kappa_from_deg(degree_eff),
                                              kappa_in > 1e3 * kappa_from_deg(5))
                    kappa_in = kappa_in[kappa_se]
                    if len(kappa_in) > 0:
                        kappa_in = kappa_in[0]

                # vertical lines denoting crossing scales
                if not np.isnan(kappa_in).all():
                    ax.vlines(x=kappa_in, ymin=y_limits[layer][0], ymax=0.0,
                              color='black', linewidth=0.8,
                              linestyle='dashed', alpha=0.6)

                    kappa_in_pos = [2, -60][kappa_in > kappa_from_lambda(40)]

                    # Scale is defined as half-wavelength
                    ax.annotate(r'$L_{in}\sim$' + '{:d} km'.format(
                        int(kappa_from_lambda(kappa_in))),
                                xy=(kappa_in, y_limits[layer][0]), xycoords='data',
                                xytext=(kappa_in_pos, 20.), textcoords='offset points',
                                color='black', fontsize=12, horizontalalignment='left',
                                verticalalignment='top')

            if model is not None and l == 0:
                art = AnchoredText('-'.join(model.split('_')).upper(),
                                   prop=dict(size=20), frameon=False, loc='lower left',
                                   bbox_to_anchor=(0.0, 0.865), bbox_transform=ax.transAxes)
                art.patch.set_boxstyle("round,pad=0., rounding_size=0.2")

                ax.add_artist(art)

            ax.set_ylim(*y_limits[layer])

            # Set legend
            layer_lim = (1e-2 * np.sort(prange)[::-1]).astype(int)
            layer_lim = level_range if prange is None else layer_lim
            layer_str = r"{} ($p_b\! =${:3d} hPa, $p_t\! =${:3d} hPa)".format(layer, *layer_lim)
            if m + l == 0 and (index not in zoom):
                legend = ax.legend(title=layer_str,
                                   loc='upper right', labelspacing=0.6, borderaxespad=0.36,
                                   ncol=legend_cols, frameon=False, columnspacing=0.6,
                                   bbox_to_anchor=legend_offset)
                legend.set_in_layout(False)
                # noinspection PyProtectedMember
                legend._legend_box.align = "right"
            else:
                # Insert legend title as Anchored text if legend is hidden
                art = AnchoredText(layer_str,
                                   prop=dict(size=params['legend.fontsize']),
                                   frameon=False, loc='upper right',
                                   bbox_to_anchor=(legend_offset[0], 1.008),
                                   bbox_transform=ax.transAxes)
                ax.add_artist(art)

            subplot_label = r'\textbf{{({})}}'.format(index)

            art = AnchoredText(subplot_label, loc='lower left',
                               prop=dict(size=18, fontweight='black'),
                               frameon=False, bbox_to_anchor=ind_offset,
                               bbox_transform=ax.transAxes)
            art.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax.add_artist(art)

    return fig


def compare_model_energy(datasets, models=None, variables=None, layers=None, zoom=None,
                         x_limits=None, start_index='', show_crossing=True, shared_ticks=True,
                         legend_cols=None, orientation='horizontal', compensated=False):
    if models is None:
        models = ['', '']

    zoom = zoom or ''

    dataset = datasets[models[0]]

    if variables is None:
        variables = ['hke', 'rke', 'dke', 'ape', 'vke']

    # if compensated and 'vke' in variables:
    #     variables.remove('vke')

    if layers is None:
        layer_limits = {'': [50e2, 1000e2]}
    else:
        assert isinstance(layers, dict), ValueError('layers must be a dictionary')
        layer_limits = layers.copy()

    y_limits = {}
    for name, limits in layer_limits.items():
        if np.size(limits) == 4:
            layer_limits[name], y_limits[name] = limits
        elif np.size(limits) == 2:
            y_limits[name] = [0.6e-2, 1.2e2] if compensated else [1e-6, 1e9]  # [7e-5, 4e7]
        else:
            raise ValueError('Wrong size for layers')

    # get coordinates
    level = 1e-2 * dataset['level'].values

    level_range = np.int32([level.max(), level.min()])

    degree_min = 2 * min([ds.kappa.size for ds in datasets.values()]) - 1
    degree_max = 2 * max([ds.kappa.size for ds in datasets.values()]) - 1

    degree_max = min(degree_max, deg_from_lambda(9.9e3))

    # mesoscale reference slope goes from 100 to the effective wavenumber
    large_scales = deg_from_lambda(1e3 * np.linspace(3200, 650, 2))

    if compensated:
        scale_st = [r'$l^{-3}$', r'$(\Delta p/g)^{\!1/3}(r\tilde{\Pi}_{K}\!)^{2/3}l^{-5/3}$']
        scale_sp = [-4 / 3, 0]
        y_label = r'Compensated energy $\tilde{E} /$ --'
        # r'Compensated energy'
    else:
        scale_st = [r'$l^{-3}$', r'$l^{-5/3}$']
        scale_sp = [-3, -5 / 3]
        y_label = r'Energy density $E(l)$ / $J ~ m^{-2}$'

    scale_mg = [2e8, 3e5]

    # -----------------------------------------------------------------------------------
    # Visualization of Kinetic energy and Available potential energy
    # -----------------------------------------------------------------------------------
    nm = len(models)
    rows = max(1, len(layer_limits))
    cols = max(1, nm)

    if orientation == 'vertical':
        rows, cols = cols, rows

    if legend_cols is None:
        legend_cols = 1 + int(len(variables) > 3)

    if degree_min < 1024 and degree_min == degree_max:
        legend_offset = (1, 1)
        lambda_lines = None
    else:
        legend_offset = (0.926, 1.0)
        lambda_lines = [20., ]

    if cols < 3:
        figure_size = (cols * 6.2, rows * 5.8)
    else:
        # square subplots for multiple columns
        figure_size = (cols * 5.8, rows * 5.8)

    fig, axes = spectra_base_figure(n_rows=rows, n_cols=cols, figure_size=figure_size,
                                    shared_ticks=(True, shared_ticks), lambda_lines=lambda_lines,
                                    x_limits=x_limits, y_limits=None, aligned=True,
                                    y_label=y_label, y_scale='log', ax_titles=None, frame=False,
                                    truncation=degree_max)

    axes = axes.ravel()
    indicator = 'abcdefghijklmn'
    indicator = indicator[indicator.find(start_index):]

    norm = 1.0
    pi_max = np.max([
        datasets[model].integrate_range().isel(kappa=slice(110, None)).pi_hke.values.max()
        for model in models
    ])

    for l, (layer, prange) in enumerate(layer_limits.items()):
        for m, model in enumerate(models):

            ax = axes[nm * l + m]

            data = datasets[model].integrate_range(coord_range=prange)

            kappa = 1e3 * datasets[model]['kappa'].values
            degree = np.arange(data.kappa.size).astype(float)

            kappa_ms = int(kappa.size / 3) + 1

            rke = data['rke'].mean('time').values
            dke = data['dke'].mean('time').values
            vke = data['vke'].mean('time').values

            # (cn.earth_radius / effective_height) ** 2
            scale_factor = (dke[kappa_ms] / vke[kappa_ms]) * kappa_ms ** 2

            print(layer, model, " effective height: {:.2f} m".format(
                cn.earth_radius / np.sqrt(scale_factor)))
            # mesoscale reference slope goes from 100 to the effective wavenumber
            if compensated:
                # compute normalization factor for compensated spectra
                # pi_max = datasets[model].integrate_range().isel(kappa=slice(110, None))
                # pi_max = pi_max.pi_hke.values.max()
                norm = 2.0 * (np.ptp(prange) / cn.g) ** (- 1 / 3)
                norm *= (cn.earth_radius * pi_max) ** (- 2 / 3)
                scale_mg = [8e2 / rows, 1.0 / rows]

                mesoscale = [120, 1820]
            else:
                mesoscale = [120, max(512, kappa_ms)]

            x_scales = np.array([large_scales, mesoscale])

            # crate zoomed axis
            index = indicator[nm * l + m]

            # create inset axis if needed
            if index in zoom:
                ax_inset = ax.inset_axes([0.3, 0.1, 0.34, 0.25])

                inset_ticks = [512, 1024, 2048]
                inset_limits = 1e3 * kappa_from_deg([min(inset_ticks), max(inset_ticks)])
                ax_inset.set_xlim(*inset_limits)
                ax_inset.set_ylim(0.06, 0.3)

                ax_inset.xaxis.set_major_formatter(ScalarFormatter())
                ax_inset.set_xticks(1e3 * kappa_from_deg(inset_ticks))
                ax_inset.set_xticklabels(inset_ticks)
                for tick in ax_inset.xaxis.get_major_ticks():
                    tick.label.set_fontsize(9)
                for tick in ax_inset.yaxis.get_major_ticks():
                    tick.label.set_fontsize(9)

                ax.indicate_inset_zoom(ax_inset, edgecolor="gray")
            else:
                ax_inset = None

            for varname in variables:

                spectra = data[varname].values

                # nondimensional spectra
                if compensated:
                    spectra *= norm * degree ** (5 / 3)

                    # compensated spectra
                    if varname == 'vke':
                        spectra *= scale_factor * degree ** (-2)

                # compute mean and standard deviation
                spectra, ci = mean_confidence_interval(spectra, confidence=0.95, axis=0)

                # plot confidence interval
                ax.fill_between(kappa, spectra + ci, spectra - ci,
                                color='gray', interpolate=False, alpha=0.125)

                kwargs = _parse_variable(varname)
                ax.plot(kappa, spectra, **kwargs)

                if ax_inset is not None:
                    ax_inset.fill_between(kappa, spectra + ci, spectra - ci,
                                          color='gray', interpolate=True, alpha=0.1)
                    ax_inset.plot(kappa, spectra, **kwargs)

            if show_crossing:
                # crossing scales: intersections between rotational and divergent kinetic energies

                kappa_c, spectra_c = find_intersections(kappa,
                                                        norm * rke * degree ** (5 / 3),
                                                        norm * dke * degree ** (5 / 3),
                                                        direction='decreasing')

                # take median of multiple crossings
                kappa_c = np.median(kappa_c)
                spectra_c = np.median(spectra_c)

                # vertical lines denoting crossing scales
                if not np.isnan(kappa_c).all() and np.all(kappa_c > 0):
                    ax.vlines(x=kappa_c, ymin=0., ymax=spectra_c,
                              color='black', linewidth=0.8,
                              linestyle='dashed', alpha=0.8)

                    kappa_c_pos = [2, -60][kappa_c > kappa_from_lambda(80)]

                    # Scale is defined as half-wavelength
                    ax.annotate(r'$L_{c}\sim$' + '{:d} km'.format(int(kappa_from_lambda(kappa_c))),
                                xy=(kappa_c, y_limits[layer][0]), xycoords='data',
                                xytext=(kappa_c_pos, 20.), textcoords='offset points',
                                color='black', fontsize=12, horizontalalignment='left',
                                verticalalignment='top')

            # plot reference slopes
            for degree, mg, sn, ss in zip(x_scales, scale_mg, scale_sp, scale_st):
                y_scale = mg * degree.astype(float) ** sn

                k = 1e3 * kappa_from_deg(degree)
                ax.plot(k, y_scale, lw=1.4, ls='dashed', color='grey')

                scale_pos = np.argmax(y_scale)

                x_scale_pos = k[scale_pos]
                y_scale_pos = y_scale[scale_pos]

                if compensated and min(degree) > 60:
                    x_text_pos, y_text_pos = 14, 20
                else:
                    x_text_pos = -10. if scale_pos == y_scale.size else -4
                    y_text_pos = 16.

                ax.annotate(ss, xy=(x_scale_pos, y_scale_pos), xycoords='data',
                            xytext=(x_text_pos, y_text_pos), textcoords='offset points',
                            color='k', horizontalalignment='left',
                            verticalalignment='top', fontsize=12)

            ax.set_ylim(*y_limits[layer])
            ax.yaxis.set_tick_params(pad=26.)

            if model is not None:
                art = AnchoredText('-'.join(model.split('_')).upper(),
                                   prop=dict(size=20), frameon=False, loc='lower left',
                                   bbox_to_anchor=(-0.025, 0.87),
                                   bbox_transform=ax.transAxes)

                art.patch.set_boxstyle("round,pad=0., rounding_size=0.2")
                ax.add_artist(art)

            layer_lim = level_range if prange is None else (1e-2 * np.sort(prange)[::-1])
            layer_str = r"{} ($p_b\! =${:3d} hPa, $p_t\! =${:3d} hPa)".format(layer,
                                                                              *layer_lim.astype(
                                                                                  int))
            if m == nm - 1:
                legend = ax.legend(title=layer_str,
                                   loc='upper right', labelspacing=0.6, borderaxespad=0.4,
                                   ncol=legend_cols, frameon=False, columnspacing=1.5,
                                   bbox_to_anchor=legend_offset)
                legend.set_in_layout(False)
                # noinspection PyProtectedMember
                legend._legend_box.align = "right"
            else:
                # Insert legend title as Anchored text if legend is hidden
                art = AnchoredText(layer_str,
                                   prop=dict(size=params['legend.fontsize']),
                                   frameon=False, loc='upper right',
                                   bbox_to_anchor=(legend_offset[0], 1.008),
                                   bbox_transform=ax.transAxes)
                ax.add_artist(art)

            subplot_label = r'\textbf{{({})}}'.format(index)

            art = AnchoredText(subplot_label, loc='lower left',
                               prop=dict(size=18, fontweight='bold'), frameon=False,
                               bbox_to_anchor=ind_offset, bbox_transform=ax.transAxes)
            art.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax.add_artist(art)

    return fig
