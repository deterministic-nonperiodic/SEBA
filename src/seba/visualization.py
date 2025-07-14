import warnings
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from xarray import Dataset
import re

import matplotlib.cm as cm
import matplotlib.colors as mcolors

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import ScalarFormatter, NullFormatter, MultipleLocator

from . import cm_data
from .constants import g, earth_radius
from .spectral_analysis import kappa_from_deg, kappa_from_lambda, deg_from_lambda
from .tools import find_intersections
from .tools import ewm_confidence_interval as mean_confidence_interval

warnings.filterwarnings('ignore')
plt.style.use('default')

params = {'xtick.labelsize': 'medium', 'ytick.labelsize': 'medium', 'font.size': 16,
          'legend.title_fontsize': 15, 'legend.fontsize': 15,
          'font.family': 'serif', 'text.usetex': True}

plt.rcParams.update(params)
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab10.colors)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

import importlib.resources as pkg_resources


# Create truncated colormap, removing darkest ~20%
def truncate_colormap(cmap, min_val=0.2, max_val=1.0, n=256):
    """Truncate a colormap between min_val and max_val."""
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{min_val:.2f},{max_val:.2f})',
        cmap(np.linspace(min_val, max_val, n))
    )
    return new_cmap


# load colormaps
# with open('cm_data/cet_d13.cm', 'r') as cfile:
#     cet_d13 = cfile.read().split('\n')
#
# with open('cm_data/bcw.cm', 'r') as cfile:
#     bcw = cfile.read().split('\n')
with pkg_resources.files(cm_data).joinpath('cet_d13.cm').open('r') as cfile:
    cet_d13 = cfile.read().splitlines()

with pkg_resources.files(cm_data).joinpath('bcw.cm').open('r') as cfile:
    bcw = cfile.read().splitlines()

colormaps = {
    'cet_bwg': LinearSegmentedColormap.from_list('cet_bwg', cet_d13),
    'bcw_bwr': LinearSegmentedColormap.from_list('bcw_bwr', bcw[:-1]),
    'mpl_ygb': truncate_colormap(cm.get_cmap('YlGnBu_r'), min_val=0.09, max_val=0.96)
}

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
    'pi_nke_IG': r'$\Pi_{IG}$',
    'pi_nke_CF': r'$\Pi_{CF}$',
    'pi_nke_adv': r'$\Pi^{adv}_{CF}$',
    'pi_nke_vtn': r'$\Pi^{vtp}_{CF}$',
    'pi_adv_IG': r'$\Pi^{int}_{IG}$',
    'pi_adv_RO': r'$\Pi^{int}_{RO}$',
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
    'cdr_w_CF': r'$\mathcal{C}_{IG \rightarrow RO}^{\omega}$',
    'cdr_v_CF': r'$\mathcal{C}_{IG \rightarrow RO}^{\zeta}$',
    'cdr_w': r'$\mathcal{C}_{D \rightarrow R}^{\omega}$',
    'cdr_v': r'$\mathcal{C}_{D \rightarrow R}^{\zeta}$',
    'cdr_c': r'$\mathcal{C}_{D \rightarrow R}^{f}$',
    'cad': r'$\mathcal{C}_{A \rightarrow D}$',
    'cad_IG': r'$\mathcal{C}_{A \rightarrow IG}$',
    'cad_RO': r'$\mathcal{C}_{A \rightarrow RO}$',
    'cad_CF': r'$\mathcal{C}_{A \rightarrow IG}$',
    'vfd': r'$\Delta^{p_b}_{p_t}\mathcal{F}_{\uparrow}$',
    'vf': r'$\mathcal{F}_{\uparrow}$',
    'vfb': r'$\mathcal{F}_{\uparrow}(p_b)$',
    'vft': r'$\mathcal{F}_{\uparrow}(p_t)$',
    'vf_dke': r'$\mathcal{F}_{D\uparrow}$',
    'vfd_dke': r'$\Delta^{p_b}_{p_t}\mathcal{F}_{D\uparrow}$',
    'vfd_dke_IG': r'$\Delta^{p_b}_{p_t}\mathcal{F}_{IG\uparrow}$',
    'vf_ape': r'$\mathcal{F}_{A\uparrow}$',
    'vfd_ape': r'$\Delta^{p_b}_{p_t}\mathcal{F}_{A\uparrow}$',
    'nlf_hke': r'$\mathcal{L}_{D}=\mathcal{C}_{A\rightarrow D}+\Delta\mathcal{F}_{D\uparrow}$',
    'dlf_hke': r'$\mathcal{L}_{D}=\mathcal{C}_{A\rightarrow D}+\partial_{p}\mathcal{F}_{D\uparrow}$',
    'mf_uw': r'$\rho\overline{u^{\prime}w^{\prime}}$',
    'mf_vw': r'$\rho\overline{v^{\prime}w^{\prime}}$',
    'mf_gw': r'$\overline{u^{\prime}w^{\prime}}$ + $\overline{v^{\prime}w^{\prime}}$',
    'dvf_dl_dke': r'$\partial_{\kappa}(\partial_{p}\mathcal{F}_{D\uparrow})$',
    'dcdr_dl': r'$\partial_{\kappa} \mathcal{C}_{D \rightarrow R}$',
    'dis_rke': r'$\mathcal{D}_R$',
    'dis_dke': r'$\mathcal{D}_D$',
    'dis_hke': r'$\mathcal{D}_K$',
    'dke_gwd': r'$\partial_t\mathcal{E}_D$ (GWD)',
    'dke_vd': r'$\partial_t\mathcal{E}_D$ (VD)',
    'dke_conv': r'$\partial_t\mathcal{E}_D$ (CONV)',
    'rke_gwd': r'$\partial_t\mathcal{E}_R$ (GWD)',
    'rke_vd': r'$\partial_t\mathcal{E}_R$ (VD)',
    'rke_conv': r'$\partial_t\mathcal{E}_R$ (CONV)',
    'hke_gwd': r'$\partial_t\mathcal{E}_K$ (GWD)',
    'hke_vd': r'$\partial_t\mathcal{E}_K$ (VD)',
    'hke_conv': r'$\partial_t\mathcal{E}_K$ (CONV)',
    'ape_gwd': r'$\partial_t\mathcal{E}_A$ (GWD)',
    'ape_vd': r'$\partial_t\mathcal{E}_A$ (VD)',
    'ape_conv': r'$\partial_t\mathcal{E}_A$ (CONV)',
    'lc': r'$\Pi_L$',
    'pi_lke': r'$\Pi_L$',
    'eg': r'$\mathcal{G}_A - \mathcal{D}$',
    'Fr_h': r"$Fr_h$",
    'Fr_v': r"$Fr_v$",
    'ST': r"$Fr^{2}_h/\alpha^{2}$",
    "Ro": r"$Ro$"
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
    'pi_hke': ('k', 'solid', m_lw),
    'pi_nke': ('k', 'solid', m_lw),
    'pi_dke': ('teal', 'solid', s_lw),
    'pi_rke': ('red', 'solid', s_lw),
    'pi_ape': ('mediumblue', 'solid', m_lw),
    'pi_ape_IG': ('mediumblue', 'solid', s_lw),
    'pi_hke_RO': ('red', '-', s_lw),
    'pi_hke_IG': ('teal', '-', s_lw),
    'pi_hke_CF': ('black', '-.', s_lw),
    'pi_nke_IG': ('teal', '-.', s_lw),
    'pi_nke_CF': ('black', '-.', s_lw),
    'pi_nke_adv': ('red', '--', s_lw),
    'pi_nke_vtn': ('blue', '--', s_lw),
    'pi_adv_RO': ('red', '--', s_lw),
    'pi_adv_IG': ('teal', '--', s_lw),
    'pi_dke_RO': ('red', '-.', s_lw),
    'pi_dke_IG': ('teal', '-.', s_lw),
    'pi_dke_CF': ('black', '-.', s_lw),
    'pi_rke_RO': ('red', '-.', s_lw),
    'pi_rke_IG': ('teal', '-.', s_lw),
    'pi_rke_CF': ('black', '-.', s_lw),
    'cdr': ('blue', '-.', m_lw),  # rebeccapurple
    'cdr_IG': ('teal', '-.', s_lw),
    'cdr_RO': ('red', '-.', s_lw),
    'cdr_CF': ('mediumblue', '-.', m_lw),
    'cdr_w_CF': ('teal', '-.', m_lw),
    'cdr_w_IG': ('teal', '-.', m_lw),
    'cdr_v_CF': ('red', '-.', m_lw),
    'cdr_w': ('black', '-.', s_lw),
    'cdr_v': ('red', '-.', s_lw),
    'cdr_c': ('blue', '-.', s_lw),
    'cad': ('green', 'solid', m_lw),
    'cad_IG': ('green', '-.', s_lw),
    'cad_RO': ('red', '-.', s_lw),
    'cad_CF': ('black', '-.', s_lw),
    'vfd_dke': ('magenta', 'solid', s_lw),
    'vf_dke': ('magenta', 'dashed', s_lw),
    'vf': ('magenta', '--', m_lw),
    'vfb': ('mediumorchid', '--', m_lw),
    'vft': ('mediumorchid', ':', m_lw),
    'vfd_dke_IG': ('magenta', 'solid', s_lw),
    'vfd': ('magenta', 'solid', m_lw),
    'vfd_ape': ('mediumblue', '-.', s_lw),
    'vf_ape': ('mediumblue', '-.', s_lw),
    'dis_rke': ('red', '-.', s_lw),
    'dis_dke': ('green', '-.', s_lw),
    'dis_hke': ('cyan', '-.', s_lw),
    'hke_gwd': ('blue', '-', s_lw),
    'hke_vd': ('magenta', '--', s_lw),
    'hke_conv': ('green', '-.', s_lw),
    'dke_gwd': ('darkorange', '-.', s_lw),
    'dke_vd': ('magenta', '--', s_lw),
    'dke_conv': ('green', '-.', s_lw),
    'rke_gwd': ('darkorange', '--', s_lw),
    'rke_vd': ('magenta', '--', s_lw),
    'rke_conv': ('green', '-.', s_lw),
    'ape_gwd': ('darkorange', '-', s_lw),
    'ape_vd': ('magenta', '--', s_lw),
    'ape_conv': ('green', '-.', s_lw),
    'lc': ('darkcyan', 'solid', m_lw),
    'pi_lke': ('darkcyan', 'solid', m_lw),
    'eg': ('black', 'dashed', s_lw),
    'nlf_hke': ('green', 'dashed', m_lw),
    'dlf_hke': ('green', 'dashed', m_lw)
}


def cf_to_latex(unit_string):
    # Replace '**' with '^' for LaTeX exponentiation
    unit_string = unit_string.replace('**', '^')

    # Use regex to find and replace occurrences of '-n' or 'n' with '^{-n}' or '^{n}'
    unit_string = re.sub(r'([a-zA-Z])(\^?)([\-]?\d+)', r'\1^{\3}', unit_string)

    # Enclose the entire string in LaTeX math mode $
    latex_string = f"${unit_string}$"

    return latex_string


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


def create_cb_ticks(data_min, data_max, num_ticks=10):
    # Ensure data_min is less than data_max
    if data_min > data_max:
        data_min, data_max = data_max, data_min

    if np.sign(data_min * data_max) > 0:
        # Generate equally spaced ticks
        return np.linspace(data_min, data_max, num_ticks)

    data_abs = max(abs(data_min), abs(data_max))
    cb_ticks = np.linspace(0.0, data_abs, num_ticks // 2)

    cb_ticks = np.append(cb_ticks, -cb_ticks, axis=0)
    cb_ticks = np.sort(np.unique(cb_ticks))

    # Ensure the ticks do not exceed the bounds
    cb_ticks = cb_ticks[cb_ticks <= data_max]
    cb_ticks = cb_ticks[cb_ticks >= data_min]

    return cb_ticks


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
        x_limits = 1e3 * kappa_from_deg([0.9, truncation])

    if truncation > 2024:
        x_ticks = np.array([2, 20, 200, 2000])
    else:
        x_ticks = np.array([1, 10, 100, 1000])

    start_degree = deg_from_lambda(1e3 * kappa_from_lambda(x_limits[0]))
    x_ticks = x_ticks[x_ticks.searchsorted(start_degree):]

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


def draw_contour_labels(ax, contour_set, y_target, n_labels=5, fmt="%.2f", **kwargs):
    """
    Place contour labels near a specific y coordinate.

    Parameters:
    - contour_set: the result of plt.contour(...)
    - y_target: y-coordinate near which to place labels
    - n_labels: number of labels to place per contour level
    - fmt: format string for label text
    - **kwargs: passed to plt.clabel()
    """
    label_locs = []

    for collection in contour_set.collections:
        level_paths = collection.get_paths()
        added = 0  # count of labels for this level

        for path in level_paths:
            if added >= n_labels:
                break

            verts = path.vertices
            x_vals, y_vals = verts[:, 0], verts[:, 1]

            # Find index of closest point to y_target
            idx = (np.abs(y_vals - y_target)).argmin()
            label_point = (x_vals[idx], y_vals[idx])

            # Avoid placing multiple labels at nearly the same x
            if all(abs(label_point[0] - x) > 1e-3 for x, _ in label_locs):
                label_locs.append(label_point)
                added += 1

    # Apply the labels
    ax.clabel(contour_set, fmt=fmt, manual=label_locs, **kwargs)


def visualize_sections(dataset, model=None, variables=None, cs_variable=None,
                       truncation=None, layers=None, x_limits=None, y_limits=None,
                       share_cbar=False, cmap=None, cs_labels=0., cb_range=2, show_names=True,
                       show_crossing=False, show_injection=False, start_index='', scale_factor=1e4):
    if variables is None:
        variables = list(dataset.data_vars)

    ax_titles = [VARIABLE_KEYMAP[name] for name in variables]

    if 'vfd_dke' in variables:
        ax_titles[variables.index('vfd_dke')] = r'$\partial_p\mathcal{F}_{D\uparrow}$'

    if 'vfd_ape' in variables:
        ax_titles[variables.index('vfd_ape')] = r'$\partial_p\mathcal{F}_{A\uparrow}$'

    if 'vfd' in variables:
        ax_titles[variables.index('vfd')] = r'$\partial_p\mathcal{F}_{\uparrow}$'

    if 'vfd_dke_IG' in variables:
        ax_titles[variables.index('vfd_dke_IG')] = r'$\partial_p\mathcal{F}_{IG\uparrow}$'

    if y_limits is None:
        y_limits = [1000., 100.]

    # Ensure cmap is a list
    if np.isscalar(cmap) or cmap is None:
        cmap = len(variables) * [cmap]
    else:
        assert len(cmap) == len(variables), "cmap must have same size as variables"

    # Map colormap names to actual colormaps
    cmaps = [
        colormaps.get(cm, colormaps['cet_bwg']) if cm is None or cm not in plt.colormaps else cm
        for cm in cmap]

    # get coordinates
    level = 1e-2 * dataset['level'].values
    kappa = 1e3 * dataset['kappa'].values

    if truncation is None:
        truncation = kappa.size

    if x_limits is None:
        x_limits = [1, truncation]

    # convert limits to wavenumber
    x_limits = 1e3 * kappa_from_deg(x_limits)

    if 'time' in dataset.dims:
        dataset = dataset.mean(dim='time')

    reference_ticks = [1, 10, 50, 100, 200, 400, 600, 800, 1000]
    last_tick = np.searchsorted(reference_ticks, min(y_limits), side='left')
    y_ticks = reference_ticks[last_tick:]

    degree_max = min(2 * kappa.size - 1, deg_from_lambda(9.9e3))
    degree_eff = max(512, int(degree_max / 3))

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
            find_intersections(kappa[1:degree_eff], pi_hke[1:degree_eff], 0.0,
                               direction='increasing')[0][-1]
            for pi_hke in dataset['pi_rke'].values]
        )

        mask = ~np.isnan(kappa_in)
        sorter = np.argsort(level[mask])
        kappa_in = np.interp(np.log(level), np.log(level[mask][sorter]), kappa_in[mask][sorter])

    if layers is None:
        layers = [450.0, 200.0, 50.]

    # -----------------------------------------------------------------------------------
    # Visualization of Kinetic energy and Available potential energy
    # -----------------------------------------------------------------------------------
    n = len(variables)
    cols = 3 if not n % 3 else n
    rows = max(1, n // cols)
    cols, rows = max(rows, cols), min(rows, cols)

    fig, axes = spectra_base_figure(n_rows=rows, n_cols=cols, x_limits=x_limits,
                                    y_limits=y_limits, figure_size=5.4,
                                    y_label='Pressure / hPa', aligned=False,
                                    y_scale='log', ax_titles=ax_titles if show_names else None,
                                    frame=True, truncation=truncation)
    axes = axes.ravel()
    indicator = 'abcdefghijklmn'
    indicator = indicator[indicator.find(start_index):]

    default_label = ''
    if np.isscalar(scale_factor):
        label = f"{scale_factor:.0E}".split('E+0')[-1]
        default_label = rf"[$\times 10^{{-{label}}} ~W / kg$]"
        scale_factor = len(variables) * [scale_factor, ]
    elif hasattr(scale_factor, '__len__'):
        assert len(scale_factor) >= len(variables), "scale_factor must have same size as variables"
    else:
        raise ValueError(f"Wrong type for scale_factor")

    if np.isscalar(cb_range):
        cb_min, cb_max = -cb_range, cb_range

    elif hasattr(cb_range, '__len__'):
        cb_range = np.array(cb_range)

        if cb_range.ndim == 1 and cb_range.size == 2:
            cb_min, cb_max = cb_range
        elif cb_range.ndim == 2 and cb_range.shape == (len(variables), 2):
            cb_min, cb_max = cb_range[:, 0], cb_range[:, 1]
        else:
            raise ValueError(
                "Invalid shape for 'cb_range'. Expected scalar, 2-element list, or (N,2) array.")
    else:
        raise ValueError("Unsupported type for 'cb_range'")

    # Broadcast scalars to lists if needed
    if np.size(cb_min) == 1:
        cb_min = [cb_min] * len(variables)
        cb_max = [cb_max] * len(variables)

    cb_label = {}
    cs_levels = {}
    cb_ticks = {}

    seen_variables = []
    for i, (varname, scale) in enumerate(zip(variables, scale_factor)):

        units = dataset[varname].attrs.get('units') or ' '
        units = cf_to_latex(units)

        label = f"{scale:.0E}".split('E+0')[-1]

        if int(label):
            cb_label[varname] = rf"[$\times 10^{{-{label}}}~${{{units}}}]"
        else:
            cb_label[varname] = ""

        # create normalizers for shared colorbar
        if varname not in seen_variables:
            dataset[varname] = scale * dataset[varname]

        seen_variables.append(varname)

        data_min = np.nanmax([dataset[varname].values.min(), cb_min[i]])
        data_max = np.nanmin([dataset[varname].values.max(), cb_max[i]])

        # Check for approximate symmetry
        if np.abs(data_min + data_max) / np.max([np.abs(data_min), np.abs(data_max)]) < 0.1:
            data_max = np.max([np.abs(data_min), np.abs(data_max)])
            data_min = -data_max

        cs_levels[varname] = np.linspace(data_min, data_max, 35)

        cb_ticks[varname] = create_cb_ticks(data_min, data_max, num_ticks=10)

    if cs_variable is not None and cs_variable not in variables:
        dataset[cs_variable] = 1e4 * dataset[cs_variable]

    cv_labels = {}
    max_decimals = 0
    for i, varname in enumerate(variables):

        default_ticks = create_cb_ticks(cb_min[i], cb_max[i], num_ticks=10)
        if cs_labels is None:
            cv_labels[varname] = default_ticks
        elif hasattr(cs_labels, '__len__'):
            cv_labels[varname] = cs_labels
        else:
            cv_labels[varname] = [cs_labels, ]

        max_decimals = max(
            [len(str(tick).split('.')[-1]) if '.' in str(tick) else 0 for tick in
             np.round(default_ticks, 2)]
        )

    # Build format string, e.g. "%.1f" or "%.2f"
    default_format_string = f"%.{max_decimals}f"

    cs_reference = []
    for m, (ax, varname) in enumerate(zip(axes, variables)):
        spectra = dataset[varname].values

        if np.isclose(abs(cb_max[m]), abs(cb_min[m]), atol=1e-2 / max(scale_factor)):
            extend = 'both'
        else:
            extend = 'max'

        cs = ax.contourf(kappa, level, spectra, extend=extend, alpha=1.,
                         vmin=cb_min[m], vmax=cb_max[m],
                         cmap=cmaps[m], levels=cs_levels[varname], zorder=0)

        # This is a fix for the white lines between contour levels
        for c in cs.collections:
            c.set_edgecolor("face")

        cs_reference.append(cs)

        # spectra = ndimage.gaussian_filter(spectra, 0.15, mode='nearest')
        csl1 = ax.contour(kappa, level, spectra, levels=cv_labels[varname],
                          colors='black', interpolate=True,
                          linewidths=1.0, linestyles='solid', alpha=0.85)

        if len(np.nonzero(csl1.levels)[0]) > 0:
            ax.clabel(csl1, csl1.levels, fmt=lambda x: f"{x:.2f}")

        if cs_variable in dataset.data_vars:

            if model == 'ICON':
                csr_levels = [-0.6, -0.4, -0.2, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
                ref_levels = [0.2, 0.4, 0.6, 0.8]
            elif model == 'IFS':
                csr_levels = [-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
                ref_levels = [0.1, 0.2, 0.3, 0.4, ]
            else:
                csr_levels = np.linspace(-0.5, 0.5, num=11)
                ref_levels = [0.1, 0.2, 0.4, 0.6, 0.8]

            csl = ax.contour(kappa, level, dataset[cs_variable].values, levels=csr_levels,
                             # cmap=colormaps['bcw_bwr'],
                             colors='black',
                             linewidths=1., alpha=.95, zorder=1)

            print("Available contour levels:", csl.levels)

            if len(csl.levels) > 0:
                ax.clabel(csl, fmt=lambda x: f"{x:.1f}", fontsize=13, colors='k')

            csl = ax.contour(kappa, level, 1e4 * dataset['vfd_dke'].values,
                             levels=ref_levels, colors='k',
                             linewidths=1.8, alpha=1., zorder=1)
            # #ff7f0e

            # Place labels at specific y value (e.g., y = 1000) and some chosen x values
            draw_contour_labels(ax, csl, y_target=90, n_labels=len(ref_levels),
                                fmt="%.1f", inline=True, fontsize=14, zorder=1)

        # add a colorbar to all axes
        if not share_cbar:
            max_decimals = max(
                [len(str(tick).split('.')[-1]) if '.' in str(tick) else 0 for tick in
                 cb_ticks[varname]]
            )

            # Build format string, e.g. "%.1f" or "%.2f"
            format_string = f"%.{max_decimals}f"

            cb = plt.colorbar(cs, ax=ax, orientation='vertical', pad=0.001,
                              format=format_string, extend='both', ticks=cb_ticks[varname])
            cb.ax.set_title(cb_label[varname], fontsize=11, loc='center', pad=11)
            cb.ax.tick_params(labelsize=12)

        ax.hlines(layers, xmin=kappa.min(), xmax=kappa.max(),
                  color='gray', linewidth=1.4, linestyle='dashed', alpha=0.6)

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
                        linestyle='--', linewidth=1., alpha=0.6)

        if show_injection:
            # crossing scales: intersections between rotational and divergent kinetic energies
            if not np.isnan(kappa_in).all():
                ax.plot(kappa_in, level, marker='.', color='red',
                        linestyle='', linewidth=1., alpha=0.8)

    if model is not None:
        art = AnchoredText('-'.join(model.split('_')),
                           prop=dict(size=15), frameon=True, loc='lower left',
                           bbox_to_anchor=(0.0, 0.87), bbox_transform=axes[0].transAxes)

        art.patch.set_boxstyle("round,pad=0., rounding_size=0.2")
        axes[0].add_artist(art)

    if share_cbar:
        for r in range(rows):
            kwargs = dict(orientation='vertical', format=default_format_string, pad=0.01,
                          extend='both', spacing='proportional', ticks=default_ticks)
            cb = fig.colorbar(cs_reference[r * cols], ax=axes[(r + 1) * cols - 1], **kwargs)
            cb.ax.set_title(cb_label[variables[0]], fontsize=11, loc='center', pad=11)
            cb.ax.tick_params(labelsize=13)

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
                     show_injection=False, x_limits=None, interface_flux=False):
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
            y_limits[name] = [-1.5, 3.0]
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

    if interface_flux and any('vfd' in vn for vn in variables):
        legend_cols = max(2, legend_cols)

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


def compare_model_fluxes(datasets, models=None, variables=None, layers=None,
                         shared_ticks=True, show_injection=False, x_limits=None,
                         start_index='', y_label=None, zoom=None, legend_cols=None,
                         legend_loc=None, interface_flux=False, orientation='horizontal',
                         experiments=None):
    # -- Normalize inputs --
    models = [models] if isinstance(models, str) else (models or [''])
    experiments = [experiments] if isinstance(experiments, str) else (experiments or [''])
    zoom = zoom or ''
    y_label = y_label or r'Cumulative energy flux / $W ~ m^{-2}$'

    dataset = datasets[models[0]]

    # -- Determine variables if not specified --
    if variables is None:
        exclude_vars = {'hke', 'rke', 'dke', 'vke', 'ape'}
        variables = [v for v in dataset.data_vars if v not in exclude_vars]

    # -- Parse layers and associated y-limits --
    if layers is None:
        layers = {'': ([50e2, 1000e2], [-1.5, 1.5])}

    if not isinstance(layers, dict):
        raise ValueError("'layers' must be a dictionary of {name: (z_range, y_range)}")

    layer_limits, y_limits = {}, {}
    for name, limits in layers.items():
        if isinstance(limits, (tuple, list)) and len(limits) == 2:
            z_range, y_range = limits
            if len(z_range) != 2 or len(y_range) != 2:
                raise ValueError(f"Invalid format in '{name}': must be two 2-element lists.")
            layer_limits[name] = z_range
            y_limits[name] = y_range
        elif isinstance(limits, (tuple, list)) and len(limits) == 1:
            z_range = limits[0]
            if len(z_range) != 2:
                raise ValueError(f"Invalid layer range for '{name}'")
            layer_limits[name] = z_range
            y_limits[name] = [-1.5, 1.5]
        else:
            raise ValueError(f"Invalid layer entry for '{name}'")

    # -- Vertical level bounds (for display only) --
    level = 1e-2 * dataset['level'].values
    level_range = np.int32([level.max(), level.min()])

    if x_limits is not None:
        # convert limits to wavenumber
        x_limits = 1e3 * kappa_from_deg(x_limits)

    # -- Spectral degree resolution --
    degree_max = 2 * max(ds.kappa.size for ds in datasets.values()) - 1
    degree_max = min(degree_max, deg_from_lambda(9.9e3))

    # -- Determine appropriate number of y-axis ticks based on data range --
    data_range = max(np.ptp(np.asarray(lim)) for lim in y_limits.values())
    thresholds = [1.0, 2.0, 4.0]
    tick_candidates = [0.1, 0.4, 1.0, 1.0]
    idx = np.searchsorted(thresholds, data_range, side='right')
    n_ticks = tick_candidates[min(idx, len(tick_candidates) - 1)]

    # -- Layout dimensions --
    nm = len(models)
    n_layers = len(layer_limits)
    rows, cols = (nm, n_layers) if orientation == 'vertical' else (n_layers, nm)

    # -- Legend options --
    legend_pos = legend_loc or 0
    legend_cols = legend_cols or (2 if interface_flux and any('vfd' in v for v in variables)
                                  else 1 + int(len(variables) >= 5))

    # -- Figure size heuristic --
    if cols < 3:
        figure_size = (cols * 6.2, rows * 5.8)
    else:
        figure_size = (cols * 5.8, rows * 5.8)

    # -- Layout & styles --
    lambda_lines = None if degree_max < 2000 else [20.0]
    inset_pos = 0.48 if degree_max < 2000 else 0.42
    legend_offset = (1, 1) if degree_max < 2000 else (0.926, 1.0)
    legend_fontsize = params['legend.fontsize']

    # -- Generate base figure and axes --
    fig, axes = spectra_base_figure(
        n_rows=rows, n_cols=cols, figure_size=figure_size, lambda_lines=lambda_lines,
        x_limits=x_limits, y_limits=None, aligned=False, n_ticks=n_ticks,
        y_label=y_label, y_scale='linear', ax_titles=None, frame=False,
        truncation=min(degree_max, 4500), shared_ticks=(True, shared_ticks)
    )
    axes = axes.ravel()

    # -- Panel labels --
    indicator = 'abcdefghijklmn'
    indicator = indicator[indicator.find(start_index):]

    if isinstance(zoom, dict):
        inset_xticks, inset_yticks = {}, {}
        for index, limits in zoom.items():
            inset_xticks.update({index: limits[0]})
            inset_yticks.update({index: limits[1]})
    else:
        inset_xticks = {str(zoom): [20, 80, 140, 200]}
        inset_yticks = {str(zoom): [-0.1, 0.1]}

    for l, (layer, prange) in enumerate(layer_limits.items()):
        for m, model in enumerate(models):

            ax = axes[nm * l + m]

            kappa = 1e3 * datasets[model]['kappa'].values
            degree_eff = int(kappa.size / 2 - 1)

            data = datasets[model].integrate_range(coord_range=prange)

            index = indicator[nm * l + m]

            # create inset axis if needed
            if index in zoom:
                ax_inset = ax.inset_axes([inset_pos, 0.516, 0.465, 0.38])  # 065

                inset_limits = 1e3 * kappa_from_deg([min(inset_xticks[index]),
                                                     max(inset_xticks[index])])

                ax_inset.set_xlim(*inset_limits)
                ax_inset.set_ylim(*inset_yticks[index])

                ax_inset.xaxis.set_major_formatter(ScalarFormatter())
                ax_inset.set_xticks(1e3 * kappa_from_deg(inset_xticks[index]))
                ax_inset.set_xticklabels(inset_xticks[index])
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
                    spectra = np.sum(
                        [data[name] for name in varname.split('+') if name in data], axis=0)

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

                        spectra_layers[interface] = spectra_in / g

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

            # ------------------------------------------------->
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

            print(f'{layer} - before setting y_limits', ax.get_ylim())

            ax.set_ylim(*y_limits[layer])
            ax.yaxis.set_tick_params(pad=2.)
            ax.yaxis.labelpad = 0

            # Set legend
            layer_lim = (1e-2 * np.sort(prange)[::-1]).astype(int)
            layer_lim = level_range if prange is None else layer_lim
            layer_str = r"{} ($p_b\! =${:3d} hPa, $p_t\! =${:3d} hPa)".format(layer, *layer_lim)
            if l == legend_pos and (indicator[nm * l + m] not in zoom):
                legend = ax.legend(title=layer_str, fontsize=legend_fontsize,
                                   loc='upper right', labelspacing=0.6, borderaxespad=0.34,
                                   ncol=legend_cols, frameon=False, columnspacing=0.4,
                                   bbox_to_anchor=legend_offset)
                legend.set_in_layout(False)
                # noinspection PyProtectedMember
                legend._legend_box.align = "right"
            else:
                # Insert legend title as Anchored text if legend is hidden
                art = AnchoredText(layer_str,
                                   prop=dict(size=legend_fontsize),
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

            print(f'{layer} - after setting y_limits', ax.get_ylim())

    return (fig, axes)


def compare_model_energy(datasets, models=None, variables=None, layers=None, zoom=None,
                         x_limits=None, start_index='', show_crossing=True, show_reference=True,
                         shared_ticks=True, legend_cols=None, orientation='horizontal',
                         compensated=False, y_scale='log', return_axes=False,
                         reference_loc='lower'):
    if models is None:
        models = ['', ]

    if np.isscalar(models):
        models = [models, ]

    if isinstance(datasets, Dataset):
        datasets = {models[0]: datasets}

    zoom = zoom or ''
    legend_pos = 0

    if variables is None:
        variables = ['hke', 'rke', 'dke', 'ape', 'vke']

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
            y_limits[name] = [0.6e-2, 1.2e2] if compensated else [1e-6, 1e9]
        else:
            raise ValueError('Wrong size for layers')

    # get coordinates
    level = 1e-2 * datasets[models[0]]['level'].values
    ps = max(datasets[models[0]]['level'].values)

    level_range = np.int32([level.max(), level.min()])

    coords = [ds.kappa.size for ds in datasets.values()]
    degree_min = 2 * min(coords) - 1
    degree_max = 2 * max(coords) - 1

    degree_max = min(degree_max, deg_from_lambda(9.9e3))

    kappa_ms = min(666, int(max(coords) / 3) + 1)

    # mesoscale reference slope goes from 100 to the effective wavenumber
    large_scales = deg_from_lambda(1e3 * np.linspace(3200, 800, 2))

    if compensated:
        scale_st = [r'$l^{-3}$',
                    r'$a^{-1}(\Delta p/g)^{\!1/3}(r\tilde{\Pi}_{K}\!)^{2/3}l^{-5/3}$',
                    r'$l^{-2}$'
                    ]
        scale_sp = [-4 / 3, 0, -1 / 3]
        y_label = r'Compensated energy $\tilde{E}(l)$'
        # r'Compensated energy'
        scale_mg = [1e3, 1.0, 50.]  # , 2.0
        mesoscale = [120, 1820]
    else:
        scale_st = [r'$l^{-3}$', r'$l^{-5/3}$']
        scale_sp = [-3, -5 / 3]
        y_label = r'Energy density $E(l)$ / $J ~ m^{-2}$'
        scale_mg = [2.2e8, 3e5]
        mesoscale = [120, max(512, kappa_ms)]

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

    if degree_min < 1600 and degree_min == degree_max:
        legend_offset = (0.9, 1)
        lambda_lines = [50., ]
    else:
        legend_offset = (0.926, 1.0)
        lambda_lines = [20., ]

    if cols < 3:
        if rows > 1:
            figure_size = (cols * 6.2, rows * 5.8)
        else:
            figure_size = (cols * 6.2, rows * 5.8)
    else:
        # square subplots for multiple columns
        figure_size = (cols * 5.8, rows * 5.8)

    fig, axes = spectra_base_figure(n_rows=rows, n_cols=cols, figure_size=figure_size,
                                    shared_ticks=(True, shared_ticks), lambda_lines=lambda_lines,
                                    x_limits=x_limits, y_limits=None, aligned=True,
                                    y_label=y_label, y_scale=y_scale, ax_titles=None, frame=False,
                                    truncation=min(degree_max, 4500))

    axes = axes.ravel()
    indicator = 'abcdefghijklmn'
    indicator = indicator[indicator.find(start_index):]

    # define normalization factor
    # pi_max = 0.3
    pi_max = np.max([
        datasets[model].integrate_range(coord_range=prange).isel(
            kappa=slice(110, None)).pi_nke.values.max()
        for _, prange in layer_limits.items() for model in models
    ])
    pi_max = max(0.0001, pi_max)

    print("Pi_max: ", pi_max)

    for l, (layer, prange) in enumerate(layer_limits.items()):
        for m, model in enumerate(models):

            ax = axes[nm * l + m]

            data = datasets[model].integrate_range(coord_range=prange)

            kappa = 1e3 * datasets[model]['kappa'].values
            degree = np.arange(data.kappa.size).astype(float)

            rke = data['rke'].mean('time').values
            dke = data['dke'].mean('time').values
            vke = data['vke'].mean('time').values

            # (cn.earth_radius / effective_height) ** 2
            scale_factor = (dke[kappa_ms] / vke[kappa_ms]) * kappa_ms ** 2

            print(layer, model, " effective height: {:.2f} m".format(
                earth_radius / np.sqrt(scale_factor)))
            # mesoscale reference slope goes from 100 to the effective wavenumber
            if compensated:
                # compute normalization factor for compensated spectra
                if "tropo" in layer.lower() or "downscale" in layer.lower():
                    # Strongly Stratified Turbulence (Lindborg, 2006)
                    cnst = 3.0
                    scale_st[1] = r'$C^{-1}(\Delta p/g)^{\!1/3}(r\tilde{\Pi}_{K}\!)^{2/3}l^{-5/3}$'

                elif "upscale" in layer.lower() or "stratos" in layer.lower():
                    # Stratified (quasi 2D) Turbulence (Lilly, 1982)
                    cnst = 3.0 / 9.8
                elif "mesos" in layer.lower():
                    cnst = 1.0
                else:
                    cnst = 2.0
                # cnst = 0.5 if np.max(prange) < ps else 1.0

                # scaling from mass per unit area
                norm = cnst * (np.max(prange) / g) ** (- 1 / 3)
                norm *= (earth_radius * pi_max) ** (- 2 / 3)

            x_scales = np.array([large_scales, mesoscale, mesoscale])

            # crate zoomed axis
            index = indicator[nm * l + m]

            # create inset axis if needed
            if index in zoom:
                ax_inset = ax.inset_axes([0.08, 0.08, 0.4, 0.32])

                inset_ticks = [60, 140, 200]
                inset_limits = 1e3 * kappa_from_deg([min(inset_ticks), max(inset_ticks)])
                ax_inset.set_xlim(*inset_limits)
                ax_inset.set_ylim(1, 5)

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
            slopes = scale_st if model == 'IFS' and "tropo" not in layer.lower() else scale_st[:-1]

            if show_reference:
                for degree, mg, sn, ss in zip(x_scales, scale_mg, scale_sp, slopes):
                    y_scale = mg * degree.astype(float) ** sn

                    k = 1e3 * kappa_from_deg(degree)
                    ax.plot(k, y_scale, lw=1.4, ls='dashed', color='grey')

                    scale_pos = np.argmax(y_scale)

                    x_scale_pos = k[scale_pos]
                    y_scale_pos = y_scale[scale_pos]

                    if compensated and min(degree) > 60:
                        x_text_pos, y_text_pos = 15.25, 22
                    else:
                        x_text_pos = -10. if scale_pos == y_scale.size else -4
                        y_text_pos = 16.

                    if reference_loc == 'lower':
                        x_text_pos -= 10
                        y_text_pos -= 30

                    else:
                        x_text_pos += 15

                    ax.annotate(ss, xy=(x_scale_pos, y_scale_pos), xycoords='data',
                                xytext=(x_text_pos, y_text_pos), textcoords='offset points',
                                color='k', horizontalalignment='left',
                                verticalalignment='top', fontsize=12)

            ax.set_ylim(*y_limits[layer])
            ax.yaxis.set_tick_params(pad=30.)

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
            # if m == nm - 1:
            if m == legend_pos and (indicator[nm * l + m] not in zoom):
                legend = ax.legend(title=layer_str, fontsize=0.95 * params['legend.fontsize'],
                                   loc='upper right', labelspacing=0.6, borderaxespad=0.4,
                                   ncol=legend_cols, frameon=False, columnspacing=1.5,
                                   bbox_to_anchor=legend_offset)

                legend.set_in_layout(False)
                # noinspection PyProtectedMember
                legend._legend_box.align = "right"
            else:
                # Insert legend title as Anchored text if legend is hidden
                art = AnchoredText(layer_str,
                                   prop=dict(size=0.95 * params['legend.fontsize']),
                                   frameon=False, loc='upper right',
                                   bbox_to_anchor=(legend_offset[0], 1.0065),
                                   bbox_transform=ax.transAxes)
                ax.add_artist(art)

            subplot_label = r'\textbf{{({})}}'.format(index)

            art = AnchoredText(subplot_label, loc='lower left',
                               prop=dict(size=18, fontweight='bold'), frameon=False,
                               bbox_to_anchor=ind_offset, bbox_transform=ax.transAxes)
            art.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax.add_artist(art)

    if return_axes:
        return fig, axes
    else:
        return fig
