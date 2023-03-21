import warnings

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.ticker import ScalarFormatter

# from constants import cp
from seba import EnergyBudget
from spectral_analysis import kappa_from_deg, kappa_from_lambda
from tools import cumulative_flux
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

    model = 'IFS'
    resolution = 'n512'
    data_path = '/home/yanm/PycharmProjects/AMSJAS_SEBA/data/'
    # data_path = '/mnt/levante/energy_budget/test_data/'

    date_time = '20[12]'
    file_names = data_path + '{}_atm_3d_inst_{}_gps_{}.nc'

    dataset_files = file_names.format(model, resolution, date_time)

    # load earth topography and surface pressure
    dset_sfc = xr.open_dataset(data_path + 'DYAMOND2_topography_{}.nc'.format(resolution))
    sfc_hgt = dset_sfc.topography_c.values
    sfc_pres = None

    # Create energy budget object
    budget = EnergyBudget(dataset_files, ghsl=sfc_hgt, ps=sfc_pres, jobs=1)

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
    file_names = 'energy_budget/{}_energy_fluxes_{}_{}.nc'.format(model, resolution, date_time)
    dataset_fluxes = xr.open_mfdataset(data_path + file_names)

    file_names = 'energy_budget/combined_physics_tendencies_{}_{}.nc'.format(resolution, date_time)
    dataset_physic = xr.open_mfdataset(data_path + file_names)

    param_data = {'ddt_ke_conv': 260, 'ddt_ke_turb': 2 / 3600.}
    param_name = {'ddt_ke_conv': "Convection", 'ddt_ke_turb': "Turbulent dissipation"}
    param_fluxes = {}

    # if model == 'IFS':
    #     for name, factor in param_data.items():
    #         flux = dataset_physic.get(name)
    #         if isinstance(flux, xr.DataArray):
    #             param_fluxes[name] = flux.values.clip(None, 0.0) / factor
    #         else:
    #             param_fluxes[name] = None

    lck = dataset_fluxes.lc.values

    pik = dataset_fluxes.pi_ke.values + lck
    pid = dataset_fluxes.pi_dke.values
    pir = dataset_fluxes.pi_rke.values

    cak = dataset_fluxes.pi_ape.values

    # Perform vertical integration along last axis
    layers = {
        # 'Stratosphere': [50e2, 250e2],
        'Free troposphere': [250e2, 850e2],
        # 'Lower troposphere': [500e2, 850e2]
    }

    ke_limits = {'Stratosphere': [-0.4, 0.4],
                 'Free troposphere': [-0.4, 0.4],
                 'Lower troposphere': [-0.4, 0.4]}

    cd_limits = {'Stratosphere': [-0.4, 0.4],
                 'Free troposphere': [-0.4, 0.4],
                 'Lower troposphere': [-0.4, 0.4]}

    if kappa.size < 1000:
        x_limits = 1e3 * kappa_from_deg(np.array([0, 1000]))
        xticks = np.array([1, 10, 100, 1000])
    else:
        x_limits = 1e3 * kappa_from_deg(np.array([0, 2048]))
        xticks = np.array([2, 20, 200, 2000])

    colors = ['green', 'magenta']

    for i, (level, prange) in enumerate(layers.items()):

        pik_l = budget.vertical_integration(pik, pressure_range=prange, vertical_axis=1).mean(0)
        pid_l = budget.vertical_integration(pid, pressure_range=prange, vertical_axis=1).mean(0)
        pir_l = budget.vertical_integration(pir, pressure_range=prange, vertical_axis=1).mean(0)
        cak_l = budget.vertical_integration(cak, pressure_range=prange, vertical_axis=1).mean(0)

        flux_dict = {}
        for flux, value in param_fluxes.items():
            flux_dict[param_name[flux]] = budget.vertical_integration(value,
                                                                      pressure_range=prange,
                                                                      vertical_axis=1).mean(0)

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

        param_lines = []
        for p, (flux, value) in enumerate(flux_dict.items()):

            if value is not None:
                pline, = ax.semilogx(kappa, kappa * value, linewidth=2.,
                                     linestyle='-', color=colors[p])
                param_lines.append(pline)

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

        ax.legend(param_lines, flux_dict.keys(), title='Parametrized fluxes',
                  loc='lower right', fontsize=15, ncol=1)

        ax.add_artist(legend)

        fig.savefig('figures/{}_helmholtz_fluxes_{}_{}-{}_np.pdf'.format(
            model, resolution, *prange_str), dpi=300)

        plt.show()
        plt.close(fig)

    # ---------------------------------------------------------------------------------------
    # Visualize fluxes cross section
    # ---------------------------------------------------------------------------------------
    figure_name = '{}_wave_fluxes_section_{}.pdf'.format(model, resolution)

    fluxes_slices_by_models(dataset_fluxes, model=None, variables=['cdr', 'ke_vf'],
                            resolution='n1024', y_limits=[1000., 100.],
                            fig_name=figure_name)
