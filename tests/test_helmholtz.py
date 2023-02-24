import warnings

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.ticker import ScalarFormatter
from tools import cumulative_flux

from seba import EnergyBudget
from spectral_analysis import kappa_from_deg, kappa_from_lambda
from visualization import AnchoredText

params = {'xtick.labelsize': 'medium',
          'ytick.labelsize': 'medium',
          'text.usetex': True, 'font.size': 14,
          'font.family': 'serif', 'font.weight': 'normal'}
plt.rcParams.update(params)
plt.rcParams['legend.title_fontsize'] = 12

warnings.filterwarnings('ignore')

if __name__ == '__main__':

    model = 'ICON'
    resolution = 'n256'
    data_path = '/home/yanm/PycharmProjects/AMSJAS_SEBA/data/'
    # '/mnt/levante/energy_budget/grid_data/'

    date_time = '20?'
    file_names = data_path + '{}_atm_3d_inst_{}_{}.nc'

    dataset_files = file_names.format(model, resolution, date_time)

    # load earth topography and surface pressure
    dset_sfc = xr.open_dataset(data_path + 'DYAMOND2_topography_{}.nc'.format(resolution))
    sfc_hgt = dset_sfc.topography_c.values
    sfc_pres = None

    # Create energy budget object
    budget = EnergyBudget(dataset_files, ghsl=sfc_hgt, ps=sfc_pres,
                          leveltype='pressure', filter_terrain=True, jobs=1)

    # ----------------------------------------------------------------------------------------------
    # Nonlinear transfer of Kinetic energy and Available potential energy
    # ----------------------------------------------------------------------------------------------
    kappa = 1e3 * budget.kappa_h

    # linear spectral transfer due to coriolis
    lck = cumulative_flux(budget.coriolis_linear_transfer())

    pik = cumulative_flux(budget.ke_nonlinear_transfer()) + lck
    pid = cumulative_flux(budget.dke_nonlinear_transfer())
    pir = cumulative_flux(budget.rke_nonlinear_transfer())

    # conversion of divergent to rotational kinetic energy
    cdr_w = cumulative_flux(budget.conversion_dke_rke_vertical())
    cdr_v = cumulative_flux(budget.conversion_dke_rke_vorticity())
    cdr_c = cumulative_flux(budget.conversion_dke_rke_coriolis())
    cdr = cdr_w + cdr_v + cdr_c

    cak = cumulative_flux(budget.conversion_ape_dke())

    # Perform vertical integration along last axis
    prange = [50e2, 450e2]

    pik_l = budget.vertical_integration(pik, pressure_range=prange, axis=-1).mean(-1)
    pid_l = budget.vertical_integration(pid, pressure_range=prange, axis=-1).mean(-1)
    pir_l = budget.vertical_integration(pir, pressure_range=prange, axis=-1).mean(-1)

    cdr_wl = budget.vertical_integration(cdr_w, pressure_range=prange, axis=-1).mean(-1)
    cdr_vl = budget.vertical_integration(cdr_v, pressure_range=prange, axis=-1).mean(-1)
    cdr_cl = budget.vertical_integration(cdr_c, pressure_range=prange, axis=-1).mean(-1)

    cdr_l = budget.vertical_integration(cdr, pressure_range=prange, axis=-1).mean(-1)
    cak_l = budget.vertical_integration(cak, pressure_range=prange, axis=-1).mean(-1)

    # Check consistency of the kinetic energy transfer of rotational and divergent components
    assert np.allclose(pir_l + pid_l, pik_l, atol=1e-2), "Inconsistent kinetic energy transfers"

    # ----------------------------------------------------------------------------------------------
    # Visualization of Kinetic energy budget
    # ----------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7.5, 5.8), constrained_layout=True)

    x_limits = 1e3 * kappa_from_deg(np.array([0, 1000]))
    xticks = np.array([1, 10, 100, 1000])

    xlimits = 1e3 * kappa_from_deg(np.array([0, 1000]))
    ylimits = [-0.6, 0.65]

    at = AnchoredText(model.upper(), prop=dict(size=20), frameon=False, loc='upper left', )
    at.patch.set_boxstyle("round,pad=-0.3,rounding_size=0.2")
    ax.add_artist(at)

    ax.semilogx(kappa, cak_l, label=r'$C_{A\rightarrow D}$',
                linewidth=1.6, linestyle='-', color='green')

    ax.semilogx(kappa, cdr_l, label=r'$C_{D\rightarrow R}$',
                linewidth=1.6, linestyle='-', color='magenta')

    ax.semilogx(kappa, pik_l, label=r'$\Pi_K$', linewidth=2., linestyle='-', color='k')
    ax.semilogx(kappa, pir_l, label=r'$\Pi_R$', linewidth=1.5, linestyle='--', color='red')
    ax.semilogx(kappa, pid_l, label=r'$\Pi_D$', linewidth=1.5, linestyle='--', color='green')

    ax.set_ylabel(r'Cumulative energy flux ($W~m^{-2}$)', fontsize=15)

    ax.axhline(y=0.0, xmin=0, xmax=1, color='gray', linewidth=1., linestyle='dashed', alpha=0.5)

    secax = ax.secondary_xaxis('top', functions=(kappa_from_lambda, kappa_from_lambda))
    secax.xaxis.set_major_formatter(ScalarFormatter())

    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks(1e3 * kappa_from_deg(xticks))
    ax.set_xticklabels(xticks)

    ax.set_xlabel(r'wavenumber', fontsize=14, labelpad=4)
    secax.set_xlabel(r'wavelength $(km)$', fontsize=14, labelpad=5)

    ax.set_xlim(*xlimits)
    ax.set_ylim(*ylimits)

    prange_str = [int(1e-2 * p) for p in sorted(prange)]

    ax.legend(title=r"{:4d} $\leq p \leq$ {:4d} hPa ".format(*prange_str),
              loc='upper right', fontsize=14)
    plt.show()
    plt.close(fig)

    # ----------------------------------------------------------------------------------------------
    # Visualization of Kinetic energy conversion
    # ----------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7.5, 5.8), constrained_layout=True)

    xlimits = 1e3 * kappa_from_deg(np.array([0, 1000]))
    ylimits = [-0.6, 0.65]

    at = AnchoredText(model.upper(), prop=dict(size=20), frameon=False, loc='upper left', )
    at.patch.set_boxstyle("round,pad=-0.3,rounding_size=0.2")
    ax.add_artist(at)

    ax.semilogx(kappa, cdr_l, label=r'$C_{D\rightarrow R}$',
                linewidth=1.6, linestyle='-', color='magenta')

    ax.semilogx(kappa, cdr_wl, label=r'Vertical motion', linewidth=2., linestyle='--', color='k')

    ax.semilogx(kappa, cdr_vl, label=r'Relative vorticity', linewidth=1.5, linestyle='--',
                color='red')
    ax.semilogx(kappa, cdr_cl, label=r'Coriolis effect', linewidth=1.5, linestyle='--',
                color='green')

    ax.set_ylabel(r'Cumulative energy flux ($W~m^{-2}$)', fontsize=15)

    ax.axhline(y=0.0, xmin=0, xmax=1, color='gray', linewidth=1., linestyle='dashed', alpha=0.5)

    secax = ax.secondary_xaxis('top', functions=(kappa_from_lambda, kappa_from_lambda))
    secax.xaxis.set_major_formatter(ScalarFormatter())

    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks(1e3 * kappa_from_deg(xticks))
    ax.set_xticklabels(xticks)

    ax.set_xlabel(r'wavenumber', fontsize=14, labelpad=4)
    secax.set_xlabel(r'wavelength $(km)$', fontsize=14, labelpad=5)

    ax.set_xlim(*xlimits)
    ax.set_ylim(*ylimits)

    prange_str = [int(1e-2 * p) for p in sorted(prange)]

    ax.legend(title=r"{:4d} $\leq p \leq$ {:4d} hPa ".format(*prange_str),
              loc='upper right', fontsize=14)
    plt.show()
