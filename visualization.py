import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import xarray as xr
from lmfit import Model
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import ScalarFormatter

from regression_models import largescaleIG_model
from regression_models import masscont_model
from regression_models import mesoscaleIG_model
from tools import intersections
from tools import kappa_from_deg, kappa_from_lambda, lambda_from_deg

warnings.filterwarnings('ignore')
plt.style.use('default')

params = {'xtick.labelsize': 'medium',
          'ytick.labelsize': 'medium',
          'text.usetex': False, 'font.size': 12,
          'font.family': 'serif', 'font.weight': 'normal'}
plt.rcParams.update(params)
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab10.colors)
plt.rcParams['legend.title_fontsize'] = 10

# global variables
color_sequence = {
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

eq_model = {'icon': 'nwp2.5', 'nicam': 'nicam3',
            'ifs': 'ifs4', 'geos': 'geos3'}

name_models = {
    'nwp2.5': 'icon',
    'ifs4': 'ifs',
    'geos3': 'geos',
    'nicam3': 'nicam',
}

# global parameters for figure axes:
truncation = 'n1024'

level_id = np.array([1, 6, 10, 17, 25, 36, 41, 46, 53])

data_path = '/media/yanm/Resources/DYAMOND/spectra/'


def mean_confidence_interval(data, confidence=0.95, axis=0):
    a = np.asanyarray(data)
    n = a.shape[axis]

    m, se = np.nanmean(a, axis=axis), stats.sem(a, axis=axis)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def transform_spectra(s):
    st = np.reshape(s, (-1, s.shape[-1]))

    return mean_confidence_interval(st, confidence=0.95, axis=0)


def plot_spectral_models(
        models=None, v_levels=24.0, varname='w_psd', rotdiv=True, large_scale=True,
        mesoscale=False, resolution='n1024', y_limits=None, y_scale='log', show_fig=True,
        legend_loc='lower left', legend_cols=1):
    if models is None:
        models = ['icon', 'nicam', 'ifs', 'geos']

    if np.isscalar(v_levels):
        v_levels = [v_levels, ]

    colors = color_sequence['models'][4]

    rows = len(v_levels)
    cols = len(models)

    all_scales = large_scale & mesoscale

    w_models = [largescaleIG_model, mesoscaleIG_model]

    kappa_igs = [kappa_from_deg(10.), kappa_from_deg(40.)]
    kappa_ige = [kappa_from_deg(40.), kappa_from_deg(200.)]

    # model labels
    labels = [r'$E_{w_{LS}}$', r'$E_{w_{MS}}$'] if all_scales else [r'$E_{w_{IG}}$', r'$E_{w_{IG}}$']

    scale_colors = ['red', 'blue']

    true_spectra = 'dashed' if all_scales else 'solid'

    # Create figure
    fig, axes = plt.subplots(
        nrows=rows, ncols=cols,
        figsize=(4.2 * cols, 4.6 * rows), constrained_layout=True)

    axes = np.reshape(axes.T, (axes.shape[-1], -1))

    y_label = r'Kinetic energy [$m^2/s^2$]'

    if varname == 'e_psd':

        y_limits = [8e-6, 8e4]
        s_scales = (r'$\kappa^{-3}$', r'$\kappa^{-5/3}$')
        s_sscale = r'$\kappa^{-5/3}$'

        x_scales = (
            kappa_from_lambda(np.linspace(2000, 500., 2)),
            kappa_from_lambda(np.linspace(300, 30., 2))
        )

        scale_magnde = [1.0e-7, 1.0e-4]
        scale_slopes = [-3., -5. / 3.]

    elif varname == 'w_psd':

        if y_limits is None:
            y_limits = [1e-6, 1e-3]

        x_scales = (kappa_from_deg(np.linspace(8, 60, 2)), kappa_from_deg(np.linspace(60, 400, 2)))

        scale_magnde = [2.6e-4, 1.25e-3]
        scale_slopes = [1. / 3., 2. / 3.]

        s_scales = r'$\kappa^{-1}$'
        s_sscale = [r'$\kappa^{1/3}$', r'$\kappa^{2/3}$']

    else:
        raise ValueError('wrong variable name passed')

    if resolution == 'n1024':
        xlimits = kappa_from_deg(np.array([0, 1400]))
        xticks = np.array([1, 10, 100, 1000])
        kappa_vlines = [kappa_from_lambda(50.), ]
    else:
        xlimits = kappa_from_deg(np.array([0, 2800]))
        xticks = np.array([2, 20, 200, 2000])
        kappa_vlines = [kappa_from_lambda(20.), ]

    kdims = slice(1, -48)

    for i, model in enumerate(models):

        filename = os.path.join(
            data_path, '{}_energy_dynamcomp_TT_{}.nc'.format(model, resolution))
        dataset = xr.open_dataset(filename)

        filename = os.path.join(
            data_path, '{}_energy_helmholtz_TT_{}.nc'.format(model, 'n1024'))
        dataset_rd = xr.open_dataset(filename)

        filename = os.path.join(
            data_path, '{}_energy_dynamcomp_IG_n256.nc'.format(model))
        dataset_igw = xr.open_dataset(filename)

        filename = os.path.join(
            data_path, '{}_energy_helmholtz_IG_n256.nc'.format(model))

        dataset_gwrd = xr.open_dataset(filename)

        zlevels = 1e-3 * dataset.height.values

        er_spectra = dataset_rd['ur_psd'].values + dataset_rd['vr_psd'].values
        ed_spectra = dataset_rd['ud_psd'].values + dataset_rd['vd_psd'].values

        # wavenumbers
        kappa = 2.0 * dataset.kappa.values[kdims]
        kappa_rd = 2.0 * dataset_rd.kappa.values[kdims]
        kappa_ig = dataset_igw.kappa.values[kdims]

        for j, vlevel in enumerate(v_levels):

            ax = axes[i, j]

            ax.set_xscale('log')
            ax.set_yscale(y_scale)

            level = np.argmin(abs(zlevels - vlevel))

            ndims = (slice(None), slice(level, level + 1), kdims)

            print(vlevel, dataset_igw.height.values[level_id][ndims[1]])

            # energy spectra
            if varname == 'e_psd':
                spectra = dataset['u_psd'].values + dataset['v_psd'].values
                spectra, e_sel, e_seu = transform_spectra(spectra[ndims] / 2.0)

            else:

                spectra, e_sel, e_seu = transform_spectra(
                    dataset['w_psd'].values[ndims] / 2.0)

            # ratio divergent to rotational spectra:
            tr_spectra, _, _ = transform_spectra(er_spectra[ndims] / 2.0)
            td_spectra, _, _ = transform_spectra(ed_spectra[ndims] / 2.0)

            # GW rotational and divergent components
            mr_spectra = dataset_gwrd['ur_psd'].values[:, level_id][
                             ndims] + dataset_gwrd['vr_psd'].values[:, level_id][ndims]

            md_spectra = dataset_gwrd['ud_psd'].values[:, level_id][
                             ndims] + dataset_gwrd['vd_psd'].values[:, level_id][ndims]

            mr_spectra, mr_sel, mr_seu = transform_spectra(mr_spectra / 2.0)
            md_spectra, md_sel, md_seu = transform_spectra(md_spectra / 2.0)

            # crossing scales: intersections between rotational and divergent kinetic energies
            kappa_cross = intersections(
                kappa_rd, td_spectra, tr_spectra, direction='increasing')

            if not np.isscalar(kappa_cross):
                kappa_cross = kappa_cross[0]

            kappa_igcross = intersections(
                kappa_ig, md_spectra, mr_spectra, direction='increasing')

            if not np.isscalar(kappa_igcross):
                kappa_igcross = kappa_igcross[0]

            height = zlevels[level]

            lev_id = height > 16.0

            # Create plots:
            if varname == 'e_psd':

                ig_spectra = dataset_igw['u_psd'].values + dataset_igw['v_psd'].values

                ig_spectra, ig_sel, ig_seu = transform_spectra(ig_spectra[:, level_id][ndims] / 2.0)

                ax.plot(
                    kappa, spectra, lw=1.4, color='k',
                    label=r'$E_{h}$', linestyle='solid', alpha=1.0)

                ax.fill_between(kappa, e_seu, e_sel, color='gray',
                                interpolate=False, alpha=0.1)

                ax.plot(
                    kappa_rd, tr_spectra, lw=1., color=colors[0],
                    label=r'$E_{r}$', linestyle='solid', alpha=1.0)

                ax.plot(
                    kappa_rd, td_spectra, lw=1., color=colors[2],
                    label=r'$E_{d}$', linestyle='solid', alpha=1.0)

                ax.loglog(
                    kappa_ig, ig_spectra, lw=1.4, color='k',
                    label=r'$E_{IG_{h}}$', linestyle='dashed', alpha=1.0)
                ax.fill_between(kappa_ig, ig_seu, ig_sel,
                                color='gray', interpolate=False, alpha=0.1)

                if rotdiv:
                    ax.loglog(
                        kappa_ig, mr_spectra, lw=1.2, color=colors[0],
                        label=r'$E_{IG_{r}}$', linestyle='dashed', alpha=1.0)
                    ax.fill_between(kappa_ig, mr_seu, mr_sel,
                                    color='gray', interpolate=False, alpha=0.1)

                    ax.loglog(
                        kappa_ig, md_spectra, lw=1.2, color=colors[2],
                        label=r'$E_{IG_{d}}$', linestyle='dashed', alpha=1.0)
                    ax.fill_between(kappa_ig, md_seu, md_sel,
                                    color='gray', interpolate=False, alpha=0.1)

            else:
                # Create plots:
                ax.plot(
                    kappa, spectra, lw=1.5, color='black',
                    label=r"$E_w$", linestyle=true_spectra, alpha=1.0
                )

                ax.fill_between(kappa, e_seu, e_sel, color='gray',
                                interpolate=False, alpha=0.1)

                # --------------------------------------------------------------
                # Model regression for large-scale w spectra
                # --------------------------------------------------------------

                for cond, w_model, kappa_s, kappa_e, scale_color, m_label in zip([large_scale, mesoscale], w_models,
                                                                                 kappa_igs, kappa_ige, scale_colors,
                                                                                 labels):

                    if not cond:
                        continue

                    mask_w = (kappa_ig >= kappa_s) & (kappa_ig < kappa_e)

                    kappa_wa = kappa_ig[mask_w]

                    mask = (kappa <= kappa_wa[-1]) & (kappa >= kappa_wa[0])

                    regression_model = Model(w_model, independent_vars=[
                        'k', 'divergent_spectra', 'rotational_spectra', 'height', 'lat', 'mid_freq'],
                                             nan_policy='omit')

                    regression_model.set_param_hint('ganma', value=0.5, min=0.0, max=4.0)

                    result = regression_model.fit(
                        spectra[mask], k=kappa_wa,
                        divergent_spectra=md_spectra[
                            mask_w], mid_freq=large_scale,
                        rotational_spectra=mr_spectra[mask_w], height=height, lat=40.)

                    with open('regression_results.dat', 'a') as logfile:
                        header = '---------------- {} MODEL IG RESULTS AT {} ----------------'
                        print(header.format(model.upper(), int(height)), file=logfile)
                        print(result.fit_report(), file=logfile)
                        print('--------------------------------------------------------------', file=logfile)

                    ws21_spectra = result.eval(
                        k=kappa_ig, divergent_spectra=md_spectra, rotational_spectra=mr_spectra, height=height)

                    ws_sel = ws21_spectra[mask_w] - result.eval_uncertainty(sigma=3.8)
                    ws_seu = ws21_spectra[mask_w] + result.eval_uncertainty(sigma=3.8)

                    ax.plot(kappa_ig[4:], ws21_spectra[4:], lw=1., color=scale_color,
                            label=m_label, linestyle='solid', alpha=1.)

                    ax.fill_between(
                        kappa_wa, ws_seu, ws_sel, color=scale_color, interpolate=False, alpha=0.1)
                # --------------------------------------------------------------
                # Model regression for mesoscale  w spectra
                # --------------------------------------------------------------
                if mesoscale:
                    mask_m = kappa_rd >= kappa_from_deg(20.)

                    mask = (kappa <= kappa_rd[-1]) & (kappa >= kappa_rd[0])

                    target_spectra = spectra[mask]

                    regression_model = Model(masscont_model, independent_vars=[
                        'k', 'divergent_spectra', 'height'])

                    result = regression_model.fit(
                        target_spectra[mask_m], k=kappa_rd[mask_m],
                        divergent_spectra=td_spectra[mask_m], alpha=0.5, beta=0.11, height=height)

                    with open('regression_results.dat', 'a') as logfile:
                        header = '---------------- {} MODEL MC RESULTS AT {} ----------------'
                        print(header.format(model.upper(), int(height)), file=logfile)
                        print(result.fit_report(), file=logfile)
                        print('--------------------------------------------------------------', file=logfile)

                    ws19_spectra = result.best_fit
                    ws_sel = ws19_spectra - result.eval_uncertainty(sigma=3.8)
                    ws_seu = ws19_spectra + result.eval_uncertainty(sigma=3.8)

                    ax.plot(
                        kappa_rd[mask_m], ws19_spectra, lw=1., color='green',
                        label=r'$E_{w_{MC}}$', linestyle='solid', alpha=1.)

                    ax.fill_between(
                        kappa_rd[mask_m], ws_seu, ws_sel, color='green', interpolate=False, alpha=0.1)

                # --------------------------------------------------------------
                # Reference slopes at small scales:
                # --------------------------------------------------------------
                # if mesoscale:
                y_sscale = scale_magnde[lev_id] * x_scales[1] ** scale_slopes[lev_id]

                ax.plot(x_scales[1], y_sscale, lw=1.2, ls='dashed', color='gray')

                ax.annotate(s_sscale[lev_id],
                            xy=(0.8 * x_scales[1].max(), 1.6 * y_sscale.max()), xycoords='data', color='gray',
                            horizontalalignment='left', verticalalignment='top',
                            fontsize=12)

            # Annotations
            ax_title = '{} ~ {:>2d} km'.format(
                name_models[model].upper(), int(vlevel))

            at = AnchoredText(ax_title, prop=dict(size=12),
                              frameon=True, loc='upper left', )
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax.add_artist(at)

            for kappa_vline in kappa_vlines:
                ax.axvline(
                    x=kappa_vline,
                    ymin=0, ymax=1, color='gray', linewidth=0.8,
                    linestyle='solid')

                ax.annotate('{:d}km'.format(int(kappa_from_lambda(kappa_vline))),
                            xy=(1.05 * kappa_vline, 0.65 * y_limits[1]), xycoords='data', color='gray',
                            horizontalalignment='left', verticalalignment='top',
                            fontsize=10)

            ax.axvspan(np.min(kappa_vlines), xlimits[
                1], alpha=0.16, color='gray')

            # reference slopes
            if varname == 'e_psd':

                for scale in range(len(scale_magnde)):

                    y_scale = scale_magnde[scale] * x_scales[scale] ** scale_slopes[scale]

                    ax.plot(x_scales[scale], y_scale, lw=1.2, ls='dashed', color='gray')

                    mid = int(np.argmax(y_scale))
                    ax.annotate(s_scales[scale],
                                xy=(x_scales[mid], 3.2 * y_scale.max()), xycoords='data', color='k',
                                horizontalalignment='left', verticalalignment='top', fontsize=12)

                if not np.isnan(kappa_cross):
                    ax.axvline(
                        x=kappa_cross,
                        ymin=0., ymax=0.56, color='black', linewidth=0.8,
                        linestyle='dashed', alpha=0.6)

                    ax.annotate(r'$L_{c}$' + '~ {:d} km'.format(int(kappa_from_lambda(kappa_cross))),
                                xy=(1.05 * kappa_cross, 3.5 * y_limits[0]), xycoords='data', color='black',
                                horizontalalignment='left', verticalalignment='top',
                                fontsize=9)

            else:
                if not np.isnan(kappa_igcross):
                    ax.axvline(
                        x=kappa_igcross,
                        ymin=0., ymax=1., color='black', linewidth=0.8,
                        linestyle='dashed', alpha=0.6)

                    ax.annotate(r'$L_{IGc}$' + '~ {:d} km'.format(int(kappa_from_lambda(kappa_igcross))),
                                xy=(1.05 * kappa_igcross, 2.0 * y_limits[0]), xycoords='data', color='black',
                                horizontalalignment='left', verticalalignment='top',
                                fontsize=9)

            secax = ax.secondary_xaxis('top', functions=(
                kappa_from_lambda, kappa_from_lambda))

            ax.xaxis.set_major_formatter(ScalarFormatter())

            ax.set_xticks(1.0 / lambda_from_deg(xticks))
            ax.set_xticklabels(xticks)

            # Align left ytick labels:
            for label in ax.yaxis.get_ticklabels():
                label.set_horizontalalignment('left')

            ax.yaxis.set_tick_params(pad=33)

            ax.set_xlim(*xlimits)
            ax.set_ylim(*y_limits)

            if not (i % 3):
                ax.set_ylabel(y_label)
            else:
                ax.axes.get_yaxis().set_visible(False)

            if j == rows - 1:
                ax.set_xlabel(r'Spherical wavenumber', fontsize=12)
            else:
                ax.axes.get_xaxis().set_visible(False)

            if j == 0:
                secax.set_xlabel(r'Spherical wavelength $[km]$', fontsize=12)

    axes[0, 0].legend(loc=legend_loc[1], bbox_to_anchor=legend_loc[2],
                      ncol=legend_cols, frameon=False, fontsize=12)

    if show_fig:
        plt.show()

    return fig


def plot_models_height_spectra(dataset, varname='w_psd', v_levels=None, y_limits=None,
                               y_scale='log', leg_loc='best', resolution=None, show_fig=True):

    colors = color_sequence['DarkMint'][8][1:]

    models = [model_name for model_name in dataset.keys()]

    if resolution is None:
        resolution = 'n2048'

    rows = len(models) // 2

    # Create figure
    fig, axes = plt.subplots(nrows=rows, ncols=2, figsize=(12, 5.8 * rows), constrained_layout=True)

    if y_limits is None:
        y_limits = [1e-10, 1e2]

    if resolution == 'n1024':
        xlimits = kappa_from_deg(np.array([0, 2000]))
        xticks = np.array([1, 10, 100, 1000])
    else:
        xlimits = kappa_from_deg(np.array([0, 2800]))
        xticks = np.array([2, 20, 200, 2000])

    kappa_vlines = [kappa_from_lambda(20.), ]

    if varname == 'w_psd':

        y_label = r'Kinetic energy $[m^2/s^2]$'

        x_lscale = kappa_from_lambda(np.linspace(2800, 600., 2))
        x_sscale = kappa_from_lambda(np.linspace(300, 30., 2))

        y_lscale = 8e-8 * x_lscale ** (-1.)

        y_sscale = 2e-4 * x_sscale ** (1. / 3.)

        x_lscale_pos = x_lscale.min()
        x_sscale_pos = 0.65 * x_sscale.max()

        y_lscale_pos = 1.5 * y_lscale.max()
        y_sscale_pos = 1.45 * y_sscale.max()

        s_lscale = r'$\kappa^{-1}$'
        s_sscale = r'$\kappa^{1/3}$'

    elif varname == 'e_psd':

        y_label = r'Kinetic energy $[m^2/s^2]$'

        x_lscale = kappa_from_lambda(np.linspace(3000, 600., 2))
        x_sscale = kappa_from_lambda(np.linspace(300, 30., 2))

        y_lscale = 4.5e-8 * x_lscale ** (-3.0)
        y_sscale = 2.2e-4 * x_sscale ** (-5.0 / 3.0)

        x_lscale_pos = x_lscale.min()
        x_sscale_pos = x_sscale.min()

        y_lscale_pos = 2.6 * y_lscale.max()
        y_sscale_pos = 2.6 * y_sscale.max()

        s_lscale = r'$\kappa^{-3}$'
        s_sscale = r'$\kappa^{-5/3}$'

    else:
        raise ValueError('Wrong variable name')

    for m, (ax, model) in enumerate(zip(axes.flatten(), models)):

        filename = os.path.join(
            data_path, '{}_energy_dynamcomp_TT_{}.nc'.format(model, resolution)
        )

        dataset = xr.open_dataset(filename)

        zlevels = 1e-3 * dataset['height']
        kappa = 2.0 * dataset['kappa']

        for i, vlevel in enumerate(v_levels):

            level = np.argmin(abs(zlevels - vlevel))

            ndims = (slice(None), slice(level, level + 1), slice(1, -24))

            kappa_m = kappa[ndims[-1]]

            # energy spectra
            if varname == 'e_psd':

                u_psd = dataset['u_psd'][ndims]
                v_psd = dataset['v_psd'][ndims]

                spectra, e_sel, e_seu = transform_spectra(0.5 * (u_psd + v_psd))

            else:
                spectra, e_sel, e_seu = transform_spectra(0.5 * dataset[varname][ndims])

            # Create plots:
            label = "{:>2d}".format(int(vlevel))

            ax.fill_between(kappa_m, e_seu, e_sel, color='gray', interpolate=False, alpha=0.1)

            ax.plot(
                kappa_m, spectra, lw=1.5, color=colors[i],
                label=label, linestyle='solid', alpha=1.0)

        # Annotations
        ax_title = '{}'.format(name_models[model].upper())

        ax.set_xscale('log')
        ax.set_yscale(y_scale)

        at = AnchoredText(ax_title, prop=dict(size=14), frameon=True, loc='upper left', )
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at)

        for kappa_vline in kappa_vlines:
            ax.axvline(
                x=kappa_vline,
                ymin=0, ymax=1, color='k', linewidth=0.8,
                linestyle='solid', alpha=0.5)

            ax.annotate('{:d}km'.format(int(kappa_from_lambda(kappa_vline))),
                        xy=(1.05 * kappa_vline, 0.65 * y_limits[1]), xycoords='data', color='k',
                        horizontalalignment='left', verticalalignment='top',
                        fontsize=12)

        ax.axvspan(np.min(kappa_vlines), xlimits[1], alpha=0.16, color='gray')

        # reference slopes
        ax.plot(
            x_sscale, y_sscale,
            lw=1.2, ls='dashed', color='gray')

        ax.plot(
            x_lscale, y_lscale,
            lw=1.2, ls='dashed', color='gray')

        ax.annotate(s_lscale,
                    xy=(x_lscale_pos, y_lscale_pos), xycoords='data', color='k',
                    horizontalalignment='left', verticalalignment='top',
                    fontsize=14)
        ax.annotate(s_sscale,
                    xy=(x_sscale_pos, y_sscale_pos), xycoords='data', color='k',
                    horizontalalignment='left', verticalalignment='top',
                    fontsize=14)

        secax = ax.secondary_xaxis('top', functions=(kappa_from_lambda, kappa_from_lambda))

        ax.xaxis.set_major_formatter(ScalarFormatter())

        ax.set_xticks(1.0 / lambda_from_deg(xticks))
        ax.set_xticklabels(xticks)

        # Align left ytick labels:
        for label in ax.yaxis.get_ticklabels():
            label.set_horizontalalignment('left')

        ax.yaxis.set_tick_params(pad=33)

        ax.set_xlim(*xlimits)
        ax.set_ylim(*y_limits)

        if not (m % 2):
            ax.set_ylabel(y_label, fontsize=15)
        else:
            ax.axes.get_yaxis().set_visible(False)

        ax.set_xlabel(r'Spherical wavenumber', fontsize=14, labelpad=4)

        if m > rows:
            ax.axes.get_xaxis().set_visible(False)

        if m < 2:
            secax.set_xlabel(r'Spherical wavelength $[km]$', fontsize=14, labelpad=5)

        if m == 0:
            ax.legend(title='Altitude [km]', loc=leg_loc, ncol=2)

    if show_fig:
        plt.show()

    return fig
