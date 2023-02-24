import warnings

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.ticker import ScalarFormatter

from seba import EnergyBudget
from spectral_analysis import kappa_from_deg, kappa_from_lambda
from visualization import AnchoredText

# , color_sequences

params = {'xtick.labelsize': 'medium',
          'ytick.labelsize': 'medium',
          'text.usetex': True, 'font.size': 14,
          'font.family': 'serif', 'font.weight': 'normal'}
plt.rcParams.update(params)
plt.rcParams['legend.title_fontsize'] = 12

warnings.filterwarnings('ignore')


def reduce_to_1d(func, data, dim="plev", **kwargs):
    res = xr.apply_ufunc(func, data, input_core_dims=[[dim]],
                         kwargs=kwargs, dask='allowed', vectorize=True)
    return res.mean(dim='time')


if __name__ == '__main__':
    # Load dyamond dataset
    model = 'ICON'
    resolution = 'n512'
    # data_path = '/home/yanm/PycharmProjects/AMSJAS_SEBA/data/'
    data_path = '/mnt/levante/energy_budget/test_data/'

    date_time = '200'
    file_names = data_path + '{}_atm_3d_inst_{}_gps_{}.nc'

    # # load earth topography and surface pressure
    # dset_sfc = xr.open_dataset(data_path + 'ICON_sfcp_{}.nc'.format(resolution))
    # sfc_pres = dset_sfc.pres_sfc.values

    dataset_dyn = xr.open_mfdataset(file_names.format(model, resolution, date_time))
    # dataset_tnd = xr.open_mfdataset('data/{}_atm_3d_tend_{}_{}.nc'.format(model, resolution,
    #                                                                       date_time))

    # load earth topography and surface pressure
    dset_sfc = xr.open_dataset(data_path + 'DYAMOND2_topography_{}.nc'.format(resolution))
    sfc_hgt = dset_sfc.topography_c.values
    sfc_pres = None

    # Create energy budget object
    budget = EnergyBudget(dataset_dyn, ghsl=sfc_hgt, ps=sfc_pres,
                          leveltype='pressure', filter_terrain=True,
                          jobs=1)

    # Compute diagnostics
    Ek = budget.horizontal_kinetic_energy()
    Ea = budget.available_potential_energy()
    Ew = budget.vertical_kinetic_energy()

    prange_trp = [250e2, 500e2]
    prange_stp = [50e2, 250e2]

    # Kinetic energy in vector form accumulate and integrate vertically
    # average over samples:
    # Ew_trp = AEB.vertical_integration(Ew, pressure_range=prange_trp).mean(-1)[1:-1]
    Ek_trp = reduce_to_1d(budget.vertical_integration, Ek, pressure_range=prange_trp)[1:-1]
    Ew_trp = reduce_to_1d(budget.vertical_integration, Ew, pressure_range=prange_trp)[1:-1]
    Ea_trp = reduce_to_1d(budget.vertical_integration, Ea, pressure_range=prange_trp)[1:-1]

    Ek_stp = reduce_to_1d(budget.vertical_integration, Ek, pressure_range=prange_stp)[1:-1]
    Ew_stp = reduce_to_1d(budget.vertical_integration, Ew, pressure_range=prange_stp)[1:-1]
    Ea_stp = reduce_to_1d(budget.vertical_integration, Ea, pressure_range=prange_stp)[1:-1]

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

    y_limits = [0.5e-3, 1e7]

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
    pik, lct, pia, cka, cdr, vfk, vfa = budget.cumulative_energy_fluxes()

    # Perform vertical integration along last axis
    layers = {'Stratosphere': [50e2, 250e2], 'Troposphere': [250e2, 500e2]}
    limits = [[-0.4, 0.4], [-0.5, 1.2]]

    for i, (level, prange) in enumerate(layers.items()):

        pik_l = reduce_to_1d(budget.vertical_integration, pik, pressure_range=prange)
        pia_l = reduce_to_1d(budget.vertical_integration, pia, pressure_range=prange)
        cka_l = reduce_to_1d(budget.vertical_integration, cka, pressure_range=prange)
        cdr_l = reduce_to_1d(budget.vertical_integration, cdr, pressure_range=prange)
        lct_l = reduce_to_1d(budget.vertical_integration, lct, pressure_range=prange)

        # Total vertical inflow from layer bottom to top
        vfk_l = reduce_to_1d(budget.vertical_integration, vfk, pressure_range=prange)
        vfa_l = reduce_to_1d(budget.vertical_integration, vfa, pressure_range=prange)

        # Create figure
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7.5, 5.8), constrained_layout=True)

        pit_l = pik_l + pia_l
        vft_l = vfk_l + vfa_l

        y_min = 1.5 * np.nanmin([pik_l, cdr_l])
        y_max = 1.5 * np.nanmax([pit_l, vft_l, cka_l])

        y_limits = limits[i]

        at = AnchoredText(model.upper(), prop=dict(size=20), frameon=False, loc='upper left', )
        at.patch.set_boxstyle("round,pad=-0.3,rounding_size=0.2")
        ax.add_artist(at)

        ax.semilogx(kappa, pit_l, label=r'$\Pi = \Pi_K + \Pi_A$', linewidth=2.5, linestyle='-',
                    color='k')
        ax.semilogx(kappa, pik_l, label=r'$\Pi_K$', linewidth=1.6, linestyle='-', color='red')
        ax.semilogx(kappa, pia_l, label=r'$\Pi_A$', linewidth=1.6, linestyle='-', color='navy')

        ax.semilogx(kappa, cka_l, label=r'$C_{A\rightarrow D}$',
                    linewidth=1.6, linestyle='--', color='green')
        ax.semilogx(kappa, cdr_l, label=r'$C_{D\rightarrow R}$',
                    linewidth=1.6, linestyle='--', color='cyan')

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

        ax.legend(title=r"{} ({:4d} $\leq p \leq$ {:4d} hPa)".format(level, *prange_str),
                  loc='upper right', fontsize=14)
        plt.show()

        fig.savefig('figures/{}_nonlinear_fluxes_{}_{}-{}.pdf'.format(
            model, resolution, *prange_str), dpi=300)
        plt.close(fig)

    # ----------------------------------------------------------------------------------------------
    # Compute APE tendency from parameterized processes
    # ----------------------------------------------------------------------------------------------
    # Perform vertical integration along last axis prange = [50e2, 950e2]
    #
    # ape_tendecies = {}
    # for name in ['ddt_temp_dyn', 'ddt_temp_radlw', 'ddt_temp_radsw',
    #              'ddt_temp_rad', 'ddt_temp_conv']:
    #
    #     pname = name.split('_')[-1].lower()
    #
    #     tend_grid = dataset_tnd.get(name)
    #     if tend_grid is not None:
    #         ape_tendecies[pname] = budget.get_ape_tendency(tend_grid, name="ddt_ape_" + pname,
    #                                                        cumulative=True)
    #
    # # Create figure
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7.5, 5.8), constrained_layout=True)
    # xlimits = 1e3 * kappa_from_deg(np.array([1, 1000]))
    #
    # at = AnchoredText(model.upper(), prop=dict(size=20), frameon=False, loc='upper left', )
    # at.patch.set_boxstyle("round,pad=-0.1,rounding_size=0.2")
    # ax.add_artist(at)
    #
    # colors = color_sequences['models'][8]
    # for i, (name, tend) in enumerate(ape_tendecies.items()):
    #     ax.semilogx(kappa, 1e6 * kappa * tend, label=name, linewidth=1.6,
    #                 linestyle='-', color=colors[i])
    #
    # ax.set_ylabel(r'APE tendency ($W / m^2$)', fontsize=14)
    #
    # ax.axhline(y=0.0, xmin=0, xmax=1, color='gray', linewidth=0.8, linestyle='dashed', alpha=0.25)
    #
    # secax = ax.secondary_xaxis('top', functions=(kappa_from_lambda, kappa_from_lambda))
    # secax.xaxis.set_major_formatter(ScalarFormatter())
    #
    # ax.xaxis.set_major_formatter(ScalarFormatter())
    # ax.set_xticks(1e3 * kappa_from_deg(xticks))
    # ax.set_xticklabels(xticks)
    #
    # ax.set_xlabel(r'wavenumber', fontsize=14, labelpad=4)
    # secax.set_xlabel(r'wavelength $(km)$', fontsize=14, labelpad=5)
    #
    # ax.set_xlim(*xlimits)
    # ax.set_ylim(-1, 2)
    #
    # prange_str = [int(1e-2 * p) for p in sorted(prange)]
    #
    # ax.legend(title=r"{:4d} $\leq p \leq$ {:4d} hPa ".format(*prange_str), loc='upper right',
    #           fontsize=15)
    # plt.show()
    #
    # fig.savefig('figures/{}_ape_tendencies_{}_{}-{}.pdf'.format(model, resolution, *prange_str),
    #             dpi=300)
    # plt.close(fig)
