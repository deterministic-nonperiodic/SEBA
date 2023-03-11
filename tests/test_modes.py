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
        '/home/yanm/PycharmProjects/AMSJAS_SEBA/data/DYAMOND2_topography_{}.nc'.format(resolution))

    sfc_hgt = dset_sfc.topography_c.values
    sfc_pres = dataset_dyn.ps.values

    p_levels = np.linspace(1000e2, 50e2, 26)

    # Create energy budget object
    budget = EnergyBudget(dataset_dyn, ghsl=sfc_hgt, ps=sfc_pres, p_levels=p_levels,
                          filter_terrain=True, jobs=1)

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

    dataset_fluxes = xr.merge(dataset_fluxes, compat="no_conflicts")

    # Perform vertical integration along last axis
    layers = {
        'Stratosphere': [50e2, 250e2],
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

        cdr_wl = budget.vertical_integration(cdr_w, pressure_range=prange, axis=-1).mean(-1)
        cdr_vl = budget.vertical_integration(cdr_v, pressure_range=prange, axis=-1).mean(-1)
        cdr_cl = budget.vertical_integration(cdr_c, pressure_range=prange, axis=-1).mean(-1)
        cdr_l = budget.vertical_integration(cdr, pressure_range=prange, axis=-1).mean(-1)

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
    figure_name = '{}_wave_fluxes_section_{}.pdf'.format(model, resolution)

    fluxes_slices_by_models(dataset_fluxes, model=None, variables=['cdr', 'ke_vf'],
                            resolution='n1024', y_limits=[1000., 100.],
                            fig_name=figure_name)
