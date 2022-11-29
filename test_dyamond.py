import warnings

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.ticker import ScalarFormatter

from AtmosphericEnergyBudget import EnergyBudget
from spectral_analysis import kappa_from_deg, kappa_from_lambda

params = {'xtick.labelsize': 'medium',
          'ytick.labelsize': 'medium',
          'text.usetex': False, 'font.size': 12,
          'font.family': 'serif', 'font.weight': 'normal'}
plt.rcParams.update(params)
plt.rcParams['legend.title_fontsize'] = 10

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # Load dyamond dataset
    resolution = 'n128'
    data_path = 'data/'  # '/mnt/levante/energy_budget/grid_data/'
    date_time = '20200127'
    file_names = data_path + 'ICON_atm_3d_inst_{}_PL_{}_{}.nc'

    # dset_dyn = xr.merge([
    #     xr.open_mfdataset(file_names.format(idv, resolution, date_time))
    #     for idv in ['uvt', 'pwe']])
    #
    # # load earth topography and surface pressure
    # dset_sfc = xr.merge([
    #     xr.open_dataset(data_path + 'ICON_sfcp_{}_{}.nc'.format(date_time, resolution)),
    #     xr.open_dataset(data_path + 'DYAMOND2_topography_{}.nc'.format(resolution))])
    #
    # sfc_hgt = dset_sfc.topography_c.values
    # sfc_pres = dset_sfc.pres_sfc.values
    #
    # # Create energy budget object
    # AEB = EnergyBudget(
    #     dset_dyn['u'].values, dset_dyn['v'].values, dset_dyn['omega'].values,
    #     dset_dyn['temp'].values, dset_dyn['plev'].values, ps=sfc_pres, ghsl=sfc_hgt,
    #     level_type='pressure', grid_type='gaussian', axes='tzyx', filter_terrain=False)

    dset_dyn = xr.open_mfdataset(data_path + 'IFS_atm_3d_inst_{}_000.nc'.format(resolution))

    # load earth topography and surface pressure
    dset_sfc = xr.open_dataset(data_path + 'DYAMOND2_topography_{}.nc'.format(resolution))

    sfc_hgt = dset_sfc.topography_c.values

    # Create energy budget object
    AEB = EnergyBudget(
        dset_dyn['u'].values, dset_dyn['v'].values, dset_dyn['omega'].values,
        dset_dyn['temp'].values, dset_dyn['plev'].values, ps=None, ghsl=sfc_hgt,
        leveltype='pressure', gridtype='gaussian', axes='tzyx', filter_terrain=False)

    # Compute diagnostics
    Ek = AEB.horizontal_kinetic_energy()
    Ea = AEB.available_potential_energy()
    Ew = AEB.vertical_kinetic_energy()

    prange_trp = [500e2, 950e2]
    prange_stp = [50e2, 500e2]

    # Kinetic energy in vector form accumulate and integrate vertically
    Ek_trp = AEB.vertical_integration(Ek, pressure_range=prange_trp).mean(-1)[1:-1]  # average over samples
    Ew_trp = AEB.vertical_integration(Ew, pressure_range=prange_trp).mean(-1)[1:-1]
    Ea_trp = AEB.vertical_integration(Ea, pressure_range=prange_trp).mean(-1)[1:-1]

    Ek_stp = AEB.vertical_integration(Ek, pressure_range=prange_stp).mean(-1)[1:-1]
    Ew_stp = AEB.vertical_integration(Ew, pressure_range=prange_stp).mean(-1)[1:-1]
    Ea_stp = AEB.vertical_integration(Ea, pressure_range=prange_stp).mean(-1)[1:-1]

    # -----------------------------------------------------------------------------------------------------------
    # Visualization of Kinetic energy and Available potential energy
    # -----------------------------------------------------------------------------------------------------------
    kappa = 1e3 * kappa_from_deg(np.arange(AEB.truncation + 1))[1:-1]  # km^-1

    xlimits = 1e3 * kappa_from_deg(np.array([0, 1000]))
    xticks = np.array([1, 10, 100, 1000])

    ylimits = [0.5e-3, 5e6]

    x_lscale = kappa_from_lambda(np.linspace(3200, 650., 2))
    x_sscale = kappa_from_lambda(np.linspace(500, 100., 2))

    y_lscale = 5.0e-4 * x_lscale ** (-3.0)
    y_sscale = 0.250 * x_sscale ** (-5.0 / 3.0)

    x_lscale_pos = x_lscale.min()
    x_sscale_pos = x_sscale.min()

    y_lscale_pos = 2.6 * y_lscale.max()
    y_sscale_pos = 2.6 * y_sscale.max()

    s_lscale = r'$l^{-3}$'
    s_sscale = r'$l^{-5/3}$'

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7.5, 5.8), constrained_layout=True)

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

    ax.set_ylabel(r'Energy ($J~m^{-2}$)', fontsize=14)

    secax = ax.secondary_xaxis('top', functions=(kappa_from_lambda, kappa_from_lambda))

    ax.xaxis.set_major_formatter(ScalarFormatter())

    ax.set_xticks(1e3 * kappa_from_deg(xticks))
    ax.set_xticklabels(xticks)

    ax.set_xlabel(r'Spherical wavenumber', fontsize=14, labelpad=4)
    secax.set_xlabel(r'Spherical wavelength $(km)$', fontsize=14, labelpad=5)

    ax.set_xlim(*xlimits)
    ax.set_ylim(*ylimits)
    ax.legend(title=r"  $p \geq 500$    $p < 500$ hPa ", loc='upper right', fontsize=12, ncol=2)

    plt.show()

    fig.savefig('figures/icon_total_energy_spectra_{}.pdf'.format(resolution), dpi=300)
    plt.close(fig)

    # -----------------------------------------------------------------------------------------------------------
    # Nonlinear transfer of Kinetic energy and Available potential energy
    # -----------------------------------------------------------------------------------------------------------
    kappa = 1e3 * kappa_from_deg(np.arange(AEB.truncation + 1))

    # Cumulative fluxes
    pik_lp, pia_lp = AEB.cumulative_energy_fluxes()

    # Energy conversion from APE to KE
    cka = AEB.energy_conversion()

    # linear spectral transfer due to coriolis
    lc = AEB.coriolis_linear_transfer()

    # Perform vertical integration along last axis
    prange = [50e2, 450e2]

    pik_l = AEB.vertical_integration(pik_lp, pressure_range=prange)
    pia_l = AEB.vertical_integration(pia_lp, pressure_range=prange)
    cka_l = AEB.vertical_integration(cka, pressure_range=prange).mean(-1)
    lc_l = AEB.vertical_integration(lc, pressure_range=prange).mean(-1)

    # Create figure
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7.5, 5.8), constrained_layout=True)
    xlimits = 1e3 * kappa_from_deg(np.array([0, 1000]))

    ax.semilogx(kappa, pik_l + pia_l, label=r'$\Pi = \Pi_K + \Pi_A$', linewidth=1.2, linestyle='-', color='k')
    ax.semilogx(kappa, pik_l, label=r'$\Pi_K$', linewidth=1., linestyle='-', color='red')
    ax.semilogx(kappa, pia_l, label=r'$\Pi_A$', linewidth=1., linestyle='-', color='navy')
    ax.semilogx(kappa, cka_l, label=r'$C_{AK}$', linewidth=1., linestyle='--', color='green')
    ax.semilogx(kappa, lc_l, label=r'$L_{c}$', linewidth=1., linestyle='-.', color='magenta')
    ax.set_ylabel(r'Cumulative energy flux ($W / m^2$)', fontsize=14)

    ax.axhline(y=0.0, xmin=0, xmax=1, color='gray', linewidth=0.8, linestyle='dashed', alpha=0.25)

    secax = ax.secondary_xaxis('top', functions=(kappa_from_lambda, kappa_from_lambda))

    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks(1e3 * kappa_from_deg(xticks))
    ax.set_xticklabels(xticks)

    ax.set_xlabel(r'wavenumber', fontsize=14, labelpad=4)
    secax.set_xlabel(r'wavelength $(km)$', fontsize=14, labelpad=5)

    ax.set_xlim(*xlimits)
    ax.set_ylim(-1., 1.5)

    prange_str = [int(1e-2 * p) for p in sorted(prange)]

    ax.legend(title=r"  {:4d} $\leq p \leq$ {:4d} hPa ".format(*prange_str), loc='upper right', fontsize=12)

    plt.show()

    fig.savefig('figures/icon_nonlinear_fluxes_{}_{}-{}.pdf'.format(resolution, *prange_str), dpi=300)
    plt.close(fig)
