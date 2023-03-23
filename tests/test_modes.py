import warnings

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.ticker import ScalarFormatter

from seba import EnergyBudget
from spectral_analysis import kappa_from_deg, kappa_from_lambda
from tools import cumulative_flux, map_func
from visualization import AnchoredText
from visualization import fluxes_slices_by_models

params = {'xtick.labelsize': 'medium',
          'ytick.labelsize': 'medium',
          'text.usetex': True, 'font.size': 14,
          'font.family': 'serif', 'font.weight': 'normal'}
plt.rcParams.update(params)
plt.rcParams['legend.title_fontsize'] = 15

warnings.filterwarnings('ignore')

if __name__ == '__main__':

    model = 'ICON'
    resolution = 'n256'
    data_path = '/media/yanm/Data/DYAMOND/data/'

    date_time = '20200202'
    file_names = data_path + '{}_IG_inst_{}_{}.nc'

    dataset_dyn = xr.open_mfdataset(file_names.format(model, date_time, resolution))

    # load earth topography and surface pressure
    dset_sfc = xr.open_dataset(
        '/home/yanm/PycharmProjects/SEBA/data/DYAMOND2_topography_{}.nc'.format(resolution))

    sfc_hgt = dset_sfc.topography_c
    sfc_pres = dataset_dyn.ps

    p_levels = np.linspace(1000e2, 10e2, 41)

    # Create energy budget object
    budget = EnergyBudget(dataset_dyn, ghsl=sfc_hgt, ps=sfc_pres, p_levels=p_levels,
                          truncation=360, jobs=1)

    # Compute diagnostics
    Ek = budget.horizontal_kinetic_energy()
    Ea = budget.available_potential_energy()
    Ew = budget.vertical_kinetic_energy()

    prange_trp = [250e2, 500e2]
    prange_stp = [50e2, 250e2]

    # Kinetic energy in vector form accumulate and integrate vertically and average over samples:
    Ek_trp = map_func(budget.vertical_integration, Ek, pressure_range=prange_trp).mean(dim='time')
    Ew_trp = map_func(budget.vertical_integration, Ew, pressure_range=prange_trp).mean(dim='time')
    Ea_trp = map_func(budget.vertical_integration, Ea, pressure_range=prange_trp).mean(dim='time')

    Ek_stp = map_func(budget.vertical_integration, Ek, pressure_range=prange_stp).mean(dim='time')
    Ew_stp = map_func(budget.vertical_integration, Ew, pressure_range=prange_stp).mean(dim='time')
    Ea_stp = map_func(budget.vertical_integration, Ea, pressure_range=prange_stp).mean(dim='time')

    # ----------------------------------------------------------------------------------------------
    # Visualization of Kinetic energy and Available potential energy
    # ----------------------------------------------------------------------------------------------
    kappa = 1e3 * Ek_trp.kappa.values  # km^-1

    if kappa.size < 1000:
        x_limits = 1e3 * kappa_from_deg(np.array([0, 1000]))
        xticks = np.array([1, 10, 100, 1000])
    else:
        x_limits = 1e3 * kappa_from_deg(np.array([0, 2048]))
        xticks = np.array([2, 20, 200, 2000])

    y_limits = [1e-4, 5e7]

    x_lscale = kappa_from_lambda(np.linspace(3200, 650., 2))
    x_sscale = kappa_from_lambda(np.linspace(450, 60., 2))

    y_lscale = 5.0e-4 * x_lscale ** (-3.0)
    y_sscale = 0.20 * x_sscale ** (-5.0 / 3.0)

    x_lscale_pos = x_lscale.min()
    x_sscale_pos = x_sscale.min()

    y_lscale_pos = 2.6 * y_lscale.max()
    y_sscale_pos = 2.6 * y_sscale.max()

    s_lscale = r'$l^{-3}$'
    s_sscale = r'$l^{-5/3}$'

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7., 5.8), constrained_layout=True)

    ax.loglog(kappa, Ek_trp, label=r'$E_K$', linewidth=1.5, linestyle='-', color='red', alpha=0.85)
    ax.loglog(kappa, Ea_trp, label=r'$E_A$', linewidth=1.5, linestyle='-', color='navy')
    ax.loglog(kappa, Ew_trp, label=r'$E_w$', linewidth=1., linestyle='-', color='black')

    ax.loglog(kappa, Ek_stp, label=r'    ', linewidth=1.5, linestyle='--', color='red', alpha=0.85)
    ax.loglog(kappa, Ea_stp, label=r'    ', linewidth=1.5, linestyle='--', color='navy')
    ax.loglog(kappa, Ew_stp, label=r'    ', linewidth=1., linestyle='--', color='black')

    # Plot reference slopes
    ax.loglog(x_sscale, y_sscale, lw=1.2, ls='dashed', color='gray')
    ax.loglog(x_lscale, y_lscale, lw=1.2, ls='dashed', color='gray')

    ax.annotate(s_lscale,
                xy=(x_lscale_pos, y_lscale_pos), xycoords='data', color='gray',
                horizontalalignment='left', verticalalignment='top', fontsize=14)
    ax.annotate(s_sscale,
                xy=(x_sscale_pos, y_sscale_pos), xycoords='data', color='gray',
                horizontalalignment='left', verticalalignment='top', fontsize=14)

    at = AnchoredText(model.upper(), prop=dict(size=20), frameon=False, loc='upper left', )
    at.patch.set_boxstyle("round,pad=-0.3,rounding_size=0.2")
    ax.add_artist(at)

    ax.set_ylabel(r'Energy ($J~m^{-2}$)', fontsize=14)

    secax = ax.secondary_xaxis('top', functions=(kappa_from_lambda, kappa_from_lambda))

    ax.xaxis.set_major_formatter(ScalarFormatter())

    ax.set_xticks(1e3 * kappa_from_deg(xticks))
    ax.set_xticklabels(xticks)

    secax.xaxis.set_major_formatter(ScalarFormatter())

    # secax.set_xticks(1e-3 * lambda_from_deg(xticks))
    # secax.set_xticklabels((np.sqrt(2.) * 1e-3 * lambda_from_deg(xticks)).astype(int))

    ax.set_xlabel(r'Spherical harmonic degree', fontsize=14, labelpad=4)
    secax.set_xlabel(r'Spherical wavelength $(km)$', fontsize=14, labelpad=5)

    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.legend(title=r"  Troposphere  /  Stratosphere", loc='upper right', fontsize=12, ncol=2)

    plt.show()
    plt.close(fig)

    # ----------------------------------------------------------------------------------------------
    # Nonlinear transfer of Kinetic energy and Available potential energy
    # ----------------------------------------------------------------------------------------------
    kappa = 1e3 * budget.kappa_h

    # conversion of divergent to rotational kinetic energy
    cdr_w = cumulative_flux(budget.conversion_dke_rke_vertical())
    cdr_v = cumulative_flux(budget.conversion_dke_rke_vorticity())
    cdr_c = cumulative_flux(budget.conversion_dke_rke_coriolis())
    cdr = cdr_w + cdr_v + cdr_c

    # ----------------------------------------------------------------------------------------------
    # Load computed fluxes
    # ----------------------------------------------------------------------------------------------
    dataset_fluxes = budget.cumulative_energy_fluxes()

    # Perform vertical integration along last axis
    layers = {
        # 'Stratosphere': [50e2, 250e2],
        'Free troposphere': [250e2, 850e2],
        # 'Lower troposphere': [500e2, 850e2]
    }

    ke_limits = {'Stratosphere': [-0.1, 0.1],
                 'Free troposphere': [-0.08, 0.08],
                 'Lower troposphere': [-0.1, 0.1]}

    # perform vertical integration
    fluxes_layers = {}
    for i, (level, prange) in enumerate(layers.items()):
        fluxes_layers[level] = map_func(budget.vertical_integration, dataset_fluxes,
                                        pressure_range=prange).mean(dim='time')

    colors = ['green', 'magenta']
    if kappa.size < 1000:
        x_limits = 1e3 * kappa_from_deg(np.array([0, 1000]))
        xticks = np.array([1, 10, 100, 1000])
    else:
        x_limits = 1e3 * kappa_from_deg(np.array([0, 2048]))
        xticks = np.array([2, 20, 200, 2000])

    for i, (level, prange) in enumerate(layers.items()):
        # Integrate fluxes in layers
        pid_l = fluxes_layers[level].pi_dke.values
        pir_l = fluxes_layers[level].pi_rke.values
        pik_l = pid_l + pir_l

        cak_l = fluxes_layers[level].cka.values

        cdr_wl = budget.vertical_integration(cdr_w, pressure_range=prange).mean(-1)
        cdr_vl = budget.vertical_integration(cdr_v, pressure_range=prange).mean(-1)
        cdr_cl = budget.vertical_integration(cdr_c, pressure_range=prange).mean(-1)
        cdr_l = budget.vertical_integration(cdr, pressure_range=prange).mean(-1)

        # ------------------------------------------------------------------------------------------
        # Visualization of Kinetic energy budget
        # ------------------------------------------------------------------------------------------
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7.5, 5.8), constrained_layout=True)

        at = AnchoredText(model.upper(), prop=dict(size=20), frameon=False, loc='upper left', )
        at.patch.set_boxstyle("round,pad=-0.3,rounding_size=0.2")
        ax.add_artist(at)

        # ax.semilogx(kappa, cak_l, label=r'$C_{A\rightarrow D}$',
        #             linewidth=1.6, linestyle='-', color='green')

        ax.semilogx(kappa, pik_l, label=r'$\Pi_K$', linewidth=2., linestyle='-', color='k')

        # if model == 'ICON':
        ax.semilogx(kappa, pid_l, label=r'$\Pi_D$', linewidth=1.6, linestyle='-', color='green')
        ax.semilogx(kappa, pir_l, label=r'$\Pi_R$', linewidth=1.6, linestyle='-', color='red')

        ax.semilogx(kappa, cdr_l, label=r'$C_{D \rightarrow R}$',
                    linewidth=2., linestyle='-', color='blue')

        ax.semilogx(kappa, cdr_wl, label=r'Vertical motion', linewidth=1.6,
                    linestyle='-.', color='black')

        ax.semilogx(kappa, cdr_vl, label=r'Relative vorticity', linewidth=1.6,
                    linestyle='-.', color='red')

        # ax.semilogx(kappa, cdr_cl, label=r'Coriolis effect', linewidth=1.6,
        #             linestyle='-.', color='green')

        ax.set_ylabel(r'Cumulative energy flux ($W~m^{-2}$)', fontsize=16)

        ax.axhline(y=0.0, xmin=0, xmax=1, color='gray', linewidth=1., linestyle='dashed', alpha=0.5)

        secax = ax.secondary_xaxis('top', functions=(kappa_from_lambda, kappa_from_lambda))
        secax.xaxis.set_major_formatter(ScalarFormatter())

        ax.xaxis.set_major_formatter(ScalarFormatter())

        ax.set_xticks(1e3 * kappa_from_deg(xticks))
        ax.set_xticklabels(xticks)

        ax.set_xlabel(r'wavenumber', fontsize=16, labelpad=4)
        secax.set_xlabel(r'wavelength $(km)$', fontsize=16, labelpad=5)

        ax.set_xlim(*x_limits)
        ax.set_ylim(*ke_limits[level])

        prange_str = [int(1e-2 * p) for p in sorted(prange)]

        legend = ax.legend(title=r"{} ({:4d} - {:4d} hPa)".format(level, *prange_str),
                           loc='upper right', fontsize=14, ncol=2)
        ax.add_artist(legend)

        # fig.savefig('figures/{}_helmholtz_fluxes_{}_{}-{}_np.pdf'.format(
        #     model, resolution, *prange_str), dpi=300)

        plt.show()
        plt.close(fig)

    # ---------------------------------------------------------------------------------------
    # Visualize fluxes cross section
    # ---------------------------------------------------------------------------------------
    figure_name = '../figures/{}_wave_fluxes_section_{}.pdf'.format(model, resolution)

    fluxes_slices_by_models(dataset_fluxes, model=None, variables=['cdr', 'vf_dke'],
                            resolution='n1024', y_limits=[1000., 10.],
                            fig_name=figure_name)
