import warnings

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.ticker import ScalarFormatter

from seba import EnergyBudget
from spectral_analysis import kappa_from_deg, kappa_from_lambda
from tools import map_func
from visualization import AnchoredText

params = {'xtick.labelsize': 'medium',
          'ytick.labelsize': 'medium',
          'text.usetex': True, 'font.size': 14,
          'font.family': 'serif', 'font.weight': 'normal'}
plt.rcParams.update(params)
plt.rcParams['legend.title_fontsize'] = 12

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # Load dyamond dataset
    model = 'ICON'
    resolution = 'n256'
    data_path = '/home/yanm/PycharmProjects/AMSJAS_SEBA/data/'
    # data_path = '/mnt/levante/energy_budget/test_data/'

    date_time = '20[0]'
    file_names = data_path + '{}_atm_3d_inst_{}_gps_{}.nc'

    # # load earth topography and surface pressure
    dset_sfc = xr.open_dataset(data_path + 'ICON_sfcp_{}.nc'.format(resolution))
    sfc_pres = dset_sfc.pres_sfc.values

    dataset_dyn = xr.open_mfdataset(file_names.format(model, resolution, date_time))

    # load earth topography and surface pressure
    dset_sfc = xr.open_dataset(data_path + 'DYAMOND2_topography_{}.nc'.format(resolution))
    sfc_hgt = dset_sfc.topography_c.values

    # Create energy budget object
    budget = EnergyBudget(dataset_dyn, ghsl=sfc_hgt, ps=sfc_pres, jobs=1)

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

    fig.savefig('figures/icon_total_energy_spectra_{}.pdf'.format(resolution), dpi=300)
    plt.close(fig)

    # ----------------------------------------------------------------------------------------------
    # Nonlinear transfer of Kinetic energy and Available potential energy
    # ----------------------------------------------------------------------------------------------
    kappa = 1e3 * budget.kappa_h

    # Cumulative fluxes:
    # - Nonlinear energy fluxes
    # - linear spectral transfer due to coriolis
    # - Energy conversion from APE to KE
    # - Vertical energy fluxes
    dataset_fluxes = budget.cumulative_energy_fluxes()

    dataset_fluxes = xr.merge(dataset_fluxes, compat="no_conflicts").mean(dim='time')

    # Perform vertical integration along last axis
    layers = {
        # 'Stratosphere': [50e2, 250e2],
        'Free troposphere': [250e2, 500e2]
    }
    limits = {
        'Stratosphere': [-0.4, 0.4],
        'Free troposphere': [-0.5, 1.0],
    }

    for i, (level, prange) in enumerate(layers.items()):

        fluxes_level = map_func(budget.vertical_integration, dataset_fluxes, pressure_range=prange)

        pik_l = fluxes_level.pi_dke.values + fluxes_level.pi_rke.values
        pia_l = fluxes_level.pi_ape.values
        cka_l = fluxes_level.cka.values
        lct_l = fluxes_level.lc.values
        vfk_l = fluxes_level.vf_dke.values
        vfa_l = fluxes_level.vf_ape.values
        cdr_l = fluxes_level.cdr.values

        # Create figure
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7.5, 5.8), constrained_layout=True)

        pit_l = pik_l + pia_l

        y_min = 1.5 * np.nanmin([pik_l, cdr_l])
        y_max = 1.5 * np.nanmax([pit_l, vfk_l + vfa_l, cka_l])

        y_limits = limits[level]

        at = AnchoredText(model.upper(), prop=dict(size=20), frameon=False, loc='upper left', )
        at.patch.set_boxstyle("round,pad=-0.3,rounding_size=0.2")
        ax.add_artist(at)

        ax.semilogx(kappa, pit_l, label=r'$\Pi = \Pi_K + \Pi_A$',
                    linewidth=2.5, linestyle='-', color='k')
        ax.semilogx(kappa, pik_l, label=r'$\Pi_K$', linewidth=1.6, linestyle='-', color='red')
        ax.semilogx(kappa, pia_l, label=r'$\Pi_A$', linewidth=1.6, linestyle='-', color='navy')

        ax.semilogx(kappa, cka_l, label=r'$C_{A\rightarrow D}$',
                    linewidth=1.6, linestyle='--', color='green')
        ax.semilogx(kappa, cdr_l, label=r'$C_{D\rightarrow R}$',
                    linewidth=1.6, linestyle='-.', color='cyan')

        # ax.semilogx(kappa, lct_l, label=r'$L_c$', linewidth=1.6, linestyle='--', color='orange')
        ax.semilogx(kappa, vfk_l + vfa_l, label=r'$F_{\uparrow}(p_b) - F_{\uparrow}(p_t)$',
                    linewidth=1.6, linestyle='-.', color='magenta')

        ax.set_ylabel(r'Cumulative energy flux ($W~m^{-2}$)', fontsize=15)

        ax.axhline(y=0.0, xmin=0, xmax=1, color='gray', linewidth=1.2, linestyle='dashed',
                   alpha=0.5)

        secax = ax.secondary_xaxis('top', functions=(kappa_from_lambda, kappa_from_lambda))
        secax.xaxis.set_major_formatter(ScalarFormatter())

        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xticks(1e3 * kappa_from_deg(xticks))
        ax.set_xticklabels(xticks)

        ax.set_xlabel(r'Spherical harmonic degree', fontsize=14, labelpad=4)
        secax.set_xlabel(r'wavelength $(km)$', fontsize=14, labelpad=5)

        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)

        prange_str = [int(1e-2 * p) for p in sorted(prange)]

        ax.legend(title=r"{} ({:4d} - {:4d} hPa)".format(level, *prange_str),
                  loc='upper right', fontsize=14)
        plt.show()

        fig.savefig('figures/{}_nonlinear_fluxes_{}_{}-{}.pdf'.format(
            model, resolution, *prange_str), dpi=300)
        plt.close(fig)
