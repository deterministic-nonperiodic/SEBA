import os
import warnings
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import ScalarFormatter
# from scipy.signal import find_peaks

from spectral_analysis import kappa_from_deg, kappa_from_lambda
# from tools import _select_by_distance
from tools import transform_spectra, intersections

warnings.filterwarnings('ignore')
plt.style.use('default')

params = {'xtick.labelsize': 'medium',
          'ytick.labelsize': 'medium',
          'text.usetex': True, 'font.size': 12,
          'legend.title_fontsize': 10,
          'font.family': 'serif', 'font.weight': 'normal'}
plt.rcParams.update(params)
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab10.colors)

# global variables
color_sequences = {
    "models": {
        4: ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', ],
        8: ['#377eb8', '#e41a1c', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf'],
    },
    "mint": {
        2: ["#e4f1e1", "#0d585f"],
        3: ["#e4f1e1", "#63a6a0", "#0d585f"],
        4: ["#e4f1e1", "#89c0b6", "#448c8a", "#0d585f"],
        5: ["#E4F1E1", "#9CCDC1", "#63A6A0", "#337F7F", "#0D585F"],
        6: ["#E4F1E1", "#abd4c7", "#7ab5ad", "#509693", "#2c7778", "#0d585f"],
        7: ["#e4f1e1", "#b4d9cc", "#89c0b6", "#63a6a0", "#448c8a", "#287274", "#0d585f"]
    },
    "DarkMint": {
        2: ("#d2fbd4", "#123f5a"),
        3: ("#d2fbd4", "#559c9e", "#123f5a"),
        4: ("#d2fbd4", "#7bbcb0", "#3a7c89", "#123f5a"),
        5: ("#d2fbd4", "#8eccb9", "#559c9e", "#2b6c7f", "#123f5a"),
        6: ("#d2fbd4", "#9cd5be", "#6cafa9", "#458892", "#266377", "#123f5a"),
        7: ("#d2fbd4", "#a5dbc2", "#7bbcb0", "#559c9e", "#3a7c89", "#235d72", "#123f5a")
    },
}

cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905],
           [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143],
           [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952,
                                                  0.779247619],
           [0.1252714286, 0.3242428571, 0.8302714286],
           [0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238,
                                                        0.8819571429],
           [0.0059571429, 0.4086142857, 0.8828428571],
           [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571,
                                                  0.8719571429],
           [0.0498142857, 0.4585714286, 0.8640571429],
           [0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667,
                                                        0.8467],
           [0.0779428571, 0.5039857143, 0.8383714286],
           [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571,
                                                       0.8262714286],
           [0.0640571429, 0.5569857143, 0.8239571429],
           [0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524,
                                                        0.819852381], [0.0265, 0.6137, 0.8135],
           [0.0238904762, 0.6286619048,
            0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667],
           [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381,
                                                        0.7607190476],
           [0.0383714286, 0.6742714286, 0.743552381],
           [0.0589714286, 0.6837571429, 0.7253857143],
           [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429],
           [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429,
                                                        0.6424333333],
           [0.2178285714, 0.7250428571, 0.6192619048],
           [0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619,
                                                        0.5711857143],
           [0.3481666667, 0.7424333333, 0.5472666667],
           [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524,
                                                  0.5033142857],
           [0.4871238095, 0.7490619048, 0.4839761905],
           [0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476,
                                                        0.4493904762],
           [0.609852381, 0.7473142857, 0.4336857143],
           [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333],
           [0.7184095238, 0.7411333333, 0.3904761905],
           [0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667,
                                                  0.3632714286],
           [0.8185047619, 0.7327333333, 0.3497904762],
           [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217],
           [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857,
                                                        0.2886428571],
           [0.9738952381, 0.7313952381, 0.266647619],
           [0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857,
                                                       0.2164142857],
           [0.9955333333, 0.7860571429, 0.196652381],
           [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857],
           [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309],
           [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333,
                                                  0.0948380952],
           [0.9661, 0.9514428571, 0.0755333333],
           [0.9763, 0.9831, 0.0538]]

parula = LinearSegmentedColormap.from_list('parula', cm_data)


def string_fmt(x):
    # $\lambda_z\sim$
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    return rf" {s}~km" if plt.rcParams["text.usetex"] else rf" {s} km"


def spectra_base_figure(n_rows=1, n_cols=1, x_limits=None, y_limits=None, y_label=None,
                        lambda_lines=None, y_scale='log', base=10, ax_titles=None,
                        frame=True, truncation='n1024', **figure_kwargs):
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
    :param truncation: spectral truncation
    :param figure_kwargs: additional keyword arguments to pass to plt.figure()
    :return: fig, axes
    """
    if y_limits is None:
        y_limits = [1e-10, 1e2]

    if 'm' in truncation:
        prefix = 'Vertical'
        if x_limits is None:
            x_limits = kappa_from_lambda(np.array([40, 1.5]))
        xticks = np.array([1, 10, 100, 1000])

        scale_str = '{:.1f}km'
    else:
        scale_str = '{:d}km'
        if truncation == 'n1024':
            prefix = 'Spherical'
            if x_limits is None:
                x_limits = kappa_from_lambda(np.array([40e3, 20]))
            xticks = np.array([1, 10, 100, 1000])
        else:
            prefix = 'Spherical'
            if x_limits is None:
                x_limits = kappa_from_lambda(np.array([40e3, 15]))
            xticks = np.array([2, 20, 200, 2000])

    if lambda_lines is None:
        lambda_lines = [36., ]

    kappa_lines = kappa_from_lambda(np.asarray(lambda_lines))

    if ax_titles is None:
        pass
    elif isinstance(ax_titles, str):
        ax_titles = n_rows * n_cols * [ax_titles, ]
    else:
        ax_titles = list(ax_titles)

    figures_size = 5.5
    reduced_size = 0.92 * figures_size
    aspect = reduced_size if n_cols > 2 else figures_size

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
                             figsize=(aspect * n_cols, figures_size * n_rows),
                             constrained_layout=True, **figure_kwargs)

    if n_rows * n_cols == 1:
        axes = np.array([axes, ])

    for m, ax in enumerate(axes.flatten()):

        # axes frame limits
        ax.set_xscale('log', base=base)
        ax.set_yscale(y_scale, base=base)

        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)

        # axes title as annotation
        if ax_titles is not None:
            at = AnchoredText(ax_titles[m], prop=dict(size=15), frameon=frame, loc='upper left', )
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax.add_artist(at)

        # draw vertical reference lines
        for lambda_line in lambda_lines:
            kappa_line = kappa_from_lambda(lambda_line)
            ax.axvline(x=kappa_line, ymin=0, ymax=1,
                       color='gray', linewidth=0.8,
                       alpha=0.6, linestyle='solid')

            ax.annotate(scale_str.format(int(lambda_line / 2)),
                        xy=(0.90, 0.98), xycoords='axes fraction',
                        color='black', fontsize=10, horizontalalignment='left',
                        verticalalignment='top')

        ax.axvspan(np.max(kappa_lines), x_limits[1], alpha=0.15, color='gray')

        # Align left ytick labels:
        for label in ax.yaxis.get_ticklabels():
            label.set_horizontalalignment('left')

        ax.yaxis.set_tick_params(pad=33)

        # set y label only for left most panels
        if not (m % n_cols):
            ax.set_ylabel(y_label, fontsize=15)
        else:
            ax.axes.get_yaxis().set_visible(False)

        if m >= n_cols * (n_rows - 1):  # lower boundary only

            if 'm' not in truncation:
                ax.xaxis.set_major_formatter(ScalarFormatter())
                ax.set_xticks(1e3 * kappa_from_deg(xticks))
                ax.set_xticklabels(xticks)

            ax.set_xlabel('{} wavenumber'.format(prefix), fontsize=14, labelpad=4)
        else:
            ax.axes.get_xaxis().set_visible(False)

        if m < n_cols:  # upper boundary only
            secax = ax.secondary_xaxis('top', functions=(kappa_from_lambda, kappa_from_lambda))

            secax.set_xlabel('{} wavelength '.format(prefix) + r'$[km]$', fontsize=14, labelpad=5)

    return fig, axes


def reference_slopes(ax, k_scales, magnitude, slopes, name='horizontal'):
    # Function for adding reference slopes to spectral plot

    if name == 'horizontal':
        s_scales = [r"$\kappa^{{{0}}}$".format(s) for s in slopes]
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


def spectral_fluxes_by_models(dataset, models=None, varname='Eh', compensate=False,
                              leg_loc='best', leg_box=None, resolution='n1024',
                              y_limits=None, fig_name='test.png'):
    if models is None:
        models = dataset.keys()

    ax_titles = ['{}'.format(model.upper()) for model in models]

    colors = color_sequences['DarkMint'][7][1:]

    if y_limits is None:
        y_limits = [1e-10, 1e2]

    if compensate:
        y_label = r'Compensated energy ($\times\kappa^{5/3}$)'
    else:
        y_label = r'Kinetic energy $[m^2/s^2]$'

    if varname == 'Ew':

        x_scales = [kappa_from_lambda(np.linspace(2800, 600, 2)),
                    kappa_from_lambda(np.linspace(300, 60, 2))]
        scale_st = ['-1', '1/3']
        scale_mg = [8.0e-7, 2.5e-4]

    elif varname == 'Eh':

        x_scales = [kappa_from_lambda(np.linspace(3000, 600, 2)),
                    kappa_from_lambda(np.linspace(300, 50, 2))]
        scale_st = ['-3', '-5/3']
        scale_mg = [2.5e-6, 0.1e-2]

    else:
        raise ValueError('Wrong variable name')

    # get coordinates
    height = dataset['height']
    kappa = dataset['kappa']

    # -----------------------------------------------------------------------------------
    # Visualization of Kinetic energy and Available potential energy
    # -----------------------------------------------------------------------------------
    n = len(models)
    cols = 2 if not n % 2 else n
    rows = max(1, n // cols)

    fig, axes = spectra_base_figure(n_rows=rows, n_cols=cols, y_limits=y_limits,
                                    y_label=y_label, y_scale='log', ax_titles=ax_titles,
                                    frame=False, truncation=resolution)
    axes = axes.ravel()

    for m, (ax, model) in enumerate(zip(axes, models)):
        for i, level in enumerate(height):
            spectra, e_sel, e_seu = transform_spectra(dataset[model][varname][:, i])

            # Create plots:
            label = "{:>2d}".format(np.round(level).astype(int))

            ax.fill_between(kappa, e_seu, e_sel, color='gray', interpolate=False, alpha=0.1)

            ax.plot(kappa, spectra, lw=1.5, color=colors[i],
                    label=label, linestyle='solid', alpha=1.0)

        # plot reference slopes
        reference_slopes(ax, x_scales, scale_mg, scale_st)

    axes[0].legend(title='Altitude [km]', loc=leg_loc, frameon=False,
                   bbox_to_anchor=leg_box, fontsize=14, ncol=2)

    plt.show()

    fig.savefig(os.path.join('figures', fig_name), dpi=300)
    plt.close(fig)


def spectra_models_by_levels(dataset, models=None, varname='Eh',
                             leg_loc='best', leg_box=None, resolution='n1024',
                             y_limits=None, fig_name='test.png'):
    if models is None:
        models = dataset.keys()

    colors = color_sequences['models'][4]

    if y_limits is None:
        y_limits = [1e-10, 1e2]

    if varname == 'Ew':

        x_scales = [kappa_from_lambda(np.linspace(2800, 600, 2)),
                    kappa_from_lambda(np.linspace(300, 60, 2))]
        scale_st = [['-1', '1/3'], ['2/3', '1/3']]
        scale_mg = [[6.5e-7, 3.e-4], [1.6e-3, 0.36e-3]]

    elif varname == 'Eh':

        x_scales = [kappa_from_lambda(np.linspace(3000, 600, 2)),
                    kappa_from_lambda(np.linspace(300, 50, 2))]
        scale_st = ['-3', '-5/3']
        scale_mg = [2.5e-6, 0.1e-2]

    else:
        raise ValueError('Wrong variable name')

    # get coordinates
    height = dataset['height']
    kappa = dataset['kappa']

    y_label = r'Kinetic energy $[m^2/s^2]$'
    layers = ['Troposphere', 'Stratosphere']
    ax_titles = [layers[np.round(lv).astype(int) > 15.0] + ' (z ~{:>2d} km)'.format(int(lv))
                 for lv in height]

    # -----------------------------------------------------------------------------------
    # Visualization of Kinetic energy and Available potential energy
    # -----------------------------------------------------------------------------------
    cols = max(1, len(height))

    fig, axes = spectra_base_figure(n_rows=1, n_cols=cols, y_limits=y_limits,
                                    y_label=y_label, y_scale='log', ax_titles=ax_titles,
                                    frame=True, truncation=resolution)

    for i, (ax, level) in enumerate(zip(axes, height)):
        for m, model in enumerate(models):
            model_data = dataset[model][varname]
            # compute mean and standard deviation
            spectra, e_sel, e_seu = transform_spectra(model_data[:, i])

            # Create plots:
            label = model.upper()

            ax.fill_between(kappa, e_seu, e_sel, color='gray', interpolate=False, alpha=0.1)

            ax.plot(kappa, spectra, lw=1.25, color=colors[m],
                    label=label, linestyle='solid', alpha=1.0)

        # plot reference slopes
        if varname == 'Ew':
            level_id = level > 15.0
        else:
            level_id = slice(None)

        reference_slopes(ax, x_scales, scale_mg[level_id], scale_st[level_id])

    axes[0].legend(loc=leg_loc, bbox_to_anchor=leg_box,
                   frameon=False, fontsize=14, ncol=1)

    plt.show()

    fig.savefig(os.path.join('figures', fig_name), dpi=300)
    plt.close(fig)


def kinetic_energy_components(dataset, dataset_igws=None, dataset_rows=None, models=None,
                              show_helmholtz=True, leg_loc='best', leg_box=None,
                              resolution='n1024', y_limits=None, fig_name='test.png'):
    if models is None:
        models = dataset.keys()

    colors = color_sequences['models'][4]

    if y_limits is None:
        y_limits = [1e-10, 1e2]

    x_scales = [kappa_from_lambda(np.linspace(3500, 800, 2)),
                kappa_from_lambda(np.linspace(300, 50, 2))]
    scale_st = ['-3', '-5/3']
    scale_mg = [2.5e-6, 0.1e-2]

    # get coordinates
    height = dataset['height']
    kappa = dataset['kappa']

    show_igws = dataset_igws is not None
    show_rows = dataset_rows is not None

    if show_igws:
        kappa_ig = dataset_igws['kappa']
    else:
        kappa_ig = kappa

    y_label = r'Kinetic energy $[m^2/s^2]$'

    ax_titles = []

    for level in height:
        for model in models:
            ax_title = '{} ~ {:>2d} km'.format(model.upper(), np.round(level).astype(int))
            ax_titles.append(ax_title)

    # -----------------------------------------------------------------------------------
    # Visualization of Kinetic energy and Helmholtz decomposition
    # -----------------------------------------------------------------------------------
    rows = max(1, len(height))
    cols = max(1, len(models))

    fig, axes = spectra_base_figure(n_rows=rows, n_cols=cols, y_limits=y_limits,
                                    y_label=y_label, y_scale='log', ax_titles=ax_titles,
                                    truncation=resolution)

    for i, level in enumerate(height):
        for m, model in enumerate(models):

            ax = axes[i, m]
            # compute mean and standard deviation
            eh_spectra, eh_sel, eh_seu = transform_spectra(dataset[model]['Eh'][:, i])

            # Create plots:
            ax.fill_between(kappa, eh_seu, eh_sel, color='gray',
                            interpolate=False, alpha=0.1)

            ax.plot(kappa, eh_spectra, lw=1.4, color='k',
                    label=r'$E_h$', linestyle='solid', alpha=1.0)

            if 'Ew' in dataset[model].keys():
                ew_spectra, ew_sel, ew_seu = transform_spectra(dataset[model]['Ew'][:, i])

                ax.fill_between(kappa, ew_seu, ew_sel, color='gray',
                                interpolate=False, alpha=0.1)

                ax.plot(kappa, ew_spectra, lw=1.4, color='k',
                        label=r'$E_w$', linestyle='dashed', alpha=1.0)

            tr_spectra, tr_sel, tr_seu = transform_spectra(dataset[model]['Er'][:, i])
            td_spectra, td_sel, td_seu = transform_spectra(dataset[model]['Ed'][:, i])

            ax.fill_between(kappa, tr_seu, tr_sel, color='gray',
                            interpolate=False, alpha=0.1)
            ax.plot(kappa, tr_spectra, lw=1., color=colors[0],
                    label=r'$E_{r}$', linestyle='solid', alpha=1.0)

            ax.fill_between(kappa, td_seu, td_sel, color='gray',
                            interpolate=False, alpha=0.1)
            ax.plot(kappa, td_spectra, lw=1., color=colors[2],
                    label=r'$E_{d}$', linestyle='solid', alpha=1.0)

            # crossing scales: intersections between rotational and divergent kinetic energies
            kappa_cross, spectra_cross = intersections(kappa, td_spectra, tr_spectra,
                                                       direction='increasing', return_y=True)
            # plot IGW components
            ig_spectra = None
            ro_spectra = None
            if show_igws:
                ig_spectra, ig_sel, ig_seu = transform_spectra(dataset_igws[model]['Eh'][:, i])

                ax.fill_between(kappa_ig, ig_seu, ig_sel,
                                color='gray', interpolate=False, alpha=0.1)
                ax.plot(kappa_ig, ig_spectra, lw=1.4, color='k',
                        label=r'$E_{IG_{h}}$', linestyle='dashed', alpha=1.0)

                if show_helmholtz:
                    mr_spectra, mr_sel, mr_seu = transform_spectra(dataset_igws[model]['Er'][:, i])
                    md_spectra, md_sel, md_seu = transform_spectra(dataset_igws[model]['Ed'][:, i])

                    ax.fill_between(kappa_ig, mr_seu, mr_sel,
                                    color='gray', interpolate=False, alpha=0.1)
                    ax.plot(kappa_ig, mr_spectra, lw=1.2, color=colors[0],
                            label=r'$E_{IG_{r}}$', linestyle='dashed', alpha=1.0)

                    ax.fill_between(kappa_ig, md_seu, md_sel,
                                    color='gray', interpolate=False, alpha=0.1)
                    ax.plot(kappa_ig, md_spectra, lw=1.2, color=colors[2],
                            label=r'$E_{IG_{d}}$', linestyle='dashed', alpha=1.0)

            if show_rows:
                ro_spectra, ro_sel, ro_seu = transform_spectra(dataset_rows[model]['Eh'][:, i])

                ax.fill_between(kappa_ig, ro_seu, ro_sel,
                                color='gray', interpolate=False, alpha=0.1)
                ax.plot(kappa_ig, ro_spectra, lw=1.4, color='red',
                        label=r'$E_{RO_{h}}$', linestyle='dashed', alpha=1.0)

                if show_helmholtz:
                    mr_spectra, mr_sel, mr_seu = transform_spectra(dataset_rows[model]['Er'][:, i])
                    md_spectra, md_sel, md_seu = transform_spectra(dataset_rows[model]['Ed'][:, i])

                    ax.fill_between(kappa_ig, mr_seu, mr_sel,
                                    color='gray', interpolate=False, alpha=0.1)
                    ax.plot(kappa_ig, mr_spectra, lw=1.2, color=colors[0],
                            label=r'$E_{RO_{r}}$', linestyle='dashed', alpha=1.0)

                    ax.fill_between(kappa_ig, md_seu, md_sel,
                                    color='gray', interpolate=False, alpha=0.1)
                    ax.plot(kappa_ig, md_spectra, lw=1.2, color=colors[2],
                            label=r'$E_{RO_{d}}$', linestyle='dashed', alpha=1.0)

            if kappa_cross is None:
                # crossing scales: intersections between rotational and divergent kinetic energies
                kappa_cross, spectra_cross = intersections(kappa_ig, ig_spectra, ro_spectra,
                                                           direction='increasing', return_y=True)
            # take first of multiple crossings
            if not np.isscalar(kappa_cross):
                kappa_cross = kappa_cross[0]
                spectra_cross = spectra_cross[0]

            # ignore crossing scales below effective resolution
            if kappa_cross > kappa_from_lambda(40.):
                kappa_cross_pos = -60
            else:
                kappa_cross_pos = 4

            # vertical lines denoting crossing scales
            if not np.isnan(kappa_cross):
                ax.vlines(x=kappa_cross, ymin=0., ymax=spectra_cross,
                          color='black', linewidth=0.8, linestyle='dashed', alpha=0.6)

                # Scale is defined as half-wavelength
                lc = int(kappa_from_lambda(kappa_cross) / 2.0)
                ax.annotate(r'$L_{c}$' + '~ {:d} km'.format(lc),
                            xy=(kappa_cross, y_limits[0]), xycoords='data',
                            xytext=(kappa_cross_pos, 20.), textcoords='offset points',
                            color='black', fontsize=9, horizontalalignment='left',
                            verticalalignment='top')

            # plot reference slopes
            reference_slopes(ax, x_scales, scale_mg, scale_st)

    axes[0, 0].legend(loc=leg_loc, bbox_to_anchor=leg_box,
                      ncol=2, frameon=False, fontsize=14)

    plt.show()

    fig.savefig(os.path.join('figures', fig_name), dpi=300)
    plt.close(fig)
