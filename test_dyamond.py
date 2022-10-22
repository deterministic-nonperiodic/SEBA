import warnings

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.ticker import ScalarFormatter

from AtmosphericEnergyBudget import EnergyBudget
from tools import kappa_from_deg, kappa_from_lambda

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
    data_path = 'data/'

    dset_uvt = xr.open_dataset(data_path + 'DYAMONDwinter2_atm_3d_inst_uvt_PL_{}_20200128.nc'.format(resolution))
    dset_pwe = xr.open_dataset(data_path + 'DYAMONDwinter2_atm_3d_inst_pwe_PL_{}_20200128.nc'.format(resolution))

    # load earth topography and surface pressure
    ghsl = xr.open_dataset(data_path + 'DYAMOND2_topography_{}.nc'.format(resolution)).topography_c.values
    sfcp = xr.open_dataset(data_path + 'DYAMOND2_sfcp_{}.nc'.format(resolution)).pres_sfc.values.squeeze()

    # Create energy budget object
    AEB = EnergyBudget(
        dset_uvt['u'].values, dset_uvt['v'].values,
        dset_pwe['omega'].values, dset_uvt['temp'].values, dset_uvt['plev'].values,
        ps=sfcp, ghsl=ghsl, leveltype='pressure', gridtype='gaussian',
        truncation=None, legfunc='stored', axes=(1, 2, 3), sample_axis=0,
        filter_terrain=True)

    # Compute energy fluxes
    Tk = AEB.ke_nonlinear_transfer()
    Ta = AEB.ape_nonlinear_transfer()
    Cka = AEB.energy_conversion()

    # Compute diagnostics
    Ek = AEB.horizontal_kinetic_energy()
    Ea = AEB.available_potential_energy()
    Ew = AEB.vertical_kinetic_energy()

    nlat = AEB.nlat

    prange_trp = [400e2, 950e2]
    prange_stp = [100e2, 400e2]

    # Kinetic energy in vector form accumulate and integrate vertically
    Ek_trp = AEB.vertical_integration(Ek, prange=prange_trp).mean(-1)[1:-1]  # average over samples
    Ew_trp = AEB.vertical_integration(Ew, prange=prange_trp).mean(-1)[1:-1]  # average over samples
    Ea_trp = AEB.vertical_integration(Ea, prange=prange_trp).mean(-1)[1:-1]  # average over samples

    Ek_stp = AEB.vertical_integration(Ek, prange=prange_stp).mean(-1)[1:-1]  # average over samples
    Ew_stp = AEB.vertical_integration(Ew, prange=prange_stp).mean(-1)[1:-1]  # average over samples
    Ea_stp = AEB.vertical_integration(Ea, prange=prange_stp).mean(-1)[1:-1]  # average over samples

    # -----------------------------------------------------------------------------------------------------------
    # Visualization of Kinetic energy and Available potential energy
    # -----------------------------------------------------------------------------------------------------------
    kappa = 1e3 * kappa_from_deg(np.arange(AEB.truncation + 1))[1:-1]  # km^-1

    xlimits = 1e3 * kappa_from_deg(np.array([0, 1000]))
    xticks = np.array([1, 10, 100, 1000])

    x_lscale = kappa_from_lambda(np.linspace(2000, 400., 2))
    x_sscale = kappa_from_lambda(np.linspace(200, 30., 2))

    y_lscale = 5.0e-4 * x_lscale ** (-3.0)
    y_sscale = 1.5 * x_sscale ** (-5.0 / 3.0)

    x_lscale_pos = x_lscale.min()
    x_sscale_pos = x_sscale.min()

    y_lscale_pos = 2.6 * y_lscale.max()
    y_sscale_pos = 2.6 * y_sscale.max()

    s_lscale = r'$\kappa^{-3}$'
    s_sscale = r'$\kappa^{-5/3}$'

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7.0, 5.8), constrained_layout=True)

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

    ax.set_ylabel(r'Energy ($J~/~m^2$)', fontsize=14)

    secax = ax.secondary_xaxis('top', functions=(kappa_from_lambda, kappa_from_lambda))

    ax.xaxis.set_major_formatter(ScalarFormatter())

    ax.set_xticks(1e3 * kappa_from_deg(xticks))
    ax.set_xticklabels(xticks)

    ax.set_xlabel(r'Spherical wavenumber', fontsize=14, labelpad=4)
    secax.set_xlabel(r'Spherical wavelength $(km)$', fontsize=14, labelpad=5)

    ax.set_xlim(*xlimits)
    ax.legend(title=r"  $p \geq 400$    $p < 400$ hPa ", loc='upper right', fontsize=12, ncol=2)

    plt.show()

    fig.savefig('figures/icon_total_energy_spectra_{}.pdf'.format(resolution), dpi=300)
    plt.close(fig)

    # -----------------------------------------------------------------------------------------------------------
    # Nonlinear transfer of Kinetic energy and Available potential energy
    # -----------------------------------------------------------------------------------------------------------
    kappa = 1e3 * kappa_from_deg(np.arange(AEB.truncation + 1))

    prange = [200e2, 400e2]

    Tk_p = AEB.vertical_integration(Tk, prange=prange).mean(-1)
    Ta_p = AEB.vertical_integration(Ta, prange=prange).mean(-1)
    Cka_l = AEB.vertical_integration(Cka, prange=prange).mean(-1)

    # Accumulate
    Tk_l = np.cumsum(Tk_p[::-1])[::-1]
    Ta_l = np.cumsum(Ta_p[::-1])[::-1]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8.0, 5.8), constrained_layout=True)

    ax.semilogx(kappa, Tk_l + Ta_l, label=r'$\Pi = \Pi_K + \Pi_A$', linewidth=1.2, linestyle='-', color='k')
    ax.semilogx(kappa, Tk_l, label=r'$\Pi_K$', linewidth=1.2, linestyle='-', color='red')
    ax.semilogx(kappa, Ta_l, label=r'$\Pi_A$', linewidth=1.2, linestyle='-', color='navy')
    ax.semilogx(kappa, Cka_l, label=r'$C_{AK}$', linewidth=1.2, linestyle='-.', color='green')
    ax.set_ylabel(r'Cumulative energy flux ($W / m^2$)', fontsize=14)

    ax.axhline(y=0.0, xmin=0, xmax=1, color='gray', linewidth=0.8, linestyle='dashed', alpha=0.25)

    secax = ax.secondary_xaxis('top', functions=(kappa_from_lambda, kappa_from_lambda))

    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks(1e3 * kappa_from_deg(xticks))
    ax.set_xticklabels(xticks)

    ax.set_xlabel(r'Spherical wavenumber', fontsize=14, labelpad=4)
    secax.set_xlabel(r'Spherical wavelength $(km)$', fontsize=14, labelpad=5)

    ax.set_xlim(*xlimits)
    ax.set_ylim(-4.0, 4.)
    ax.legend(loc='upper right', fontsize=12)

    plt.show()

    fig.savefig('figures/icon_nonlinear_fluxes_{}_test.pdf'.format(resolution), dpi=300)
    plt.close(fig)
