import warnings

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.ticker import ScalarFormatter

from src.io_tools import parse_dataset
from src.seba import EnergyBudget
from src.spectral_analysis import kappa_from_deg, kappa_from_lambda
from src.visualization import AnchoredText, fluxes_slices_by_models

params = {'xtick.labelsize': 'medium',
          'ytick.labelsize': 'medium',
          'text.usetex': True, 'font.size': 14,
          'font.family': 'serif', 'font.weight': 'normal'}
plt.rcParams.update(params)
plt.rcParams['legend.title_fontsize'] = 15

warnings.filterwarnings('ignore')

if __name__ == '__main__':

    model = 'ICON'
    data_path = '/media/yanm/Data/DYAMOND/data/'

    date_time = '2020020[2]'

    p_levels = np.linspace(1000e2, 10e2, 41)

    dataset = {}
    dataset_fluxes = {}
    for mode in ['IG', 'RO']:
        file_names = data_path + f"{model}_{mode}_inst_{date_time}_n256.nc"

        # parse dataset so interpolation is done only once
        dataset[mode] = parse_dataset(file_names, p_levels=p_levels)

        # compute cumulative energy fluxes
        dataset_fluxes[mode] = EnergyBudget(dataset[mode]).nonlinear_energy_fluxes()
        dataset_fluxes[mode] = dataset_fluxes[mode].cumulative_sum(dim='kappa')

    # Create dataset with full winds
    add_var = ['u_wind', 'v_wind', 'omega']
    keep_var = ['temperature', 'pressure', 'geopotential', 'ps', 'ts']

    # replace dataset with full fields data
    dataset = xr.merge([dataset['RO'][add_var] + dataset['IG'][add_var], dataset['IG'][keep_var]])

    # Create energy budget object
    budget = EnergyBudget(dataset)

    # compute cumulative energy fluxes
    dataset_fluxes['FF'] = budget.nonlinear_energy_fluxes().cumulative_sum(dim='kappa')

    wave_mean_fluxes = dataset_fluxes['FF'] - dataset_fluxes['IG'] - dataset_fluxes['RO']

    # ---------------------------------------------------------------------------------------
    # Visualize fluxes cross section
    # ---------------------------------------------------------------------------------------
    variables = ['cdr', 'vfd_dke', 'pi_hke']

    fluxes_slices_by_models(dataset_fluxes['FF'], model=None, variables=variables,
                            resolution='n1024', y_limits=[1000., 10.],
                            fig_name=f'../figures/{model}_total_fluxes_section_n256.pdf')

    fluxes_slices_by_models(wave_mean_fluxes, model=None, variables=variables,
                            resolution='n1024', y_limits=[1000., 10.],
                            fig_name=f'../figures/{model}_wave_vortex_fluxes_section_n256.pdf')

    # ----------------------------------------------------------------------------------------------
    # Nonlinear transfer of Kinetic energy and Available potential energy
    # ----------------------------------------------------------------------------------------------
    kappa = 1e3 * budget.kappa_h

    layers = {
        'Free troposphere': [250e2, 850e2],
        # 'Lower troposphere': [500e2, 950e2]
    }

    ke_limits = {
        "IG": {'Free troposphere': [-0.08, 0.08], 'Lower troposphere': [-0.6, 0.6]},
        "RO": {'Free troposphere': [-0.60, 0.60], 'Lower troposphere': [-0.6, 0.6]},
    }

    if kappa.size < 1000:
        x_limits = 1e3 * kappa_from_deg(np.array([0, 1000]))
        xticks = np.array([1, 10, 100, 1000])
    else:
        x_limits = 1e3 * kappa_from_deg(np.array([0, 2048]))
        xticks = np.array([2, 20, 200, 2000])

    for i, (level, prange) in enumerate(layers.items()):
        # Integrate fluxes in layers
        data = dataset_fluxes['FF'].integrate_range(coord_range=prange).mean(dim='time')

        cad = data.cad.values
        pid = data.pi_dke.values
        pir = data.pi_rke.values

        cdr_w = data.cdr_w.values
        cdr_v = data.cdr_v.values
        cdr_c = data.cdr_c.values
        cdr = data.cdr.values - cdr_c

        # ------------------------------------------------------------------------------------------
        # Visualization of Kinetic energy budget
        # ------------------------------------------------------------------------------------------
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7.5, 5.8), constrained_layout=True)

        at = AnchoredText(model.upper(), prop=dict(size=20), frameon=False, loc='upper left', )
        at.patch.set_boxstyle("round,pad=-0.3,rounding_size=0.2")
        ax.add_artist(at)

        # ax.semilogx(kappa, cad, label=r'$C_{A\rightarrow D}$',
        #             linewidth=1.6, linestyle='-', color='green')

        ax.semilogx(kappa, pid + pir, label=r'$\Pi_K$', linewidth=2., linestyle='-', color='k')

        ax.semilogx(kappa, pid, label=r'$\Pi_D$', linewidth=1.6, linestyle='-', color='green')
        ax.semilogx(kappa, pir, label=r'$\Pi_R$', linewidth=1.6, linestyle='-', color='red')

        ax.semilogx(kappa, cdr, label=r'$C_{D \rightarrow R}$',
                    linewidth=2., linestyle='-', color='blue')

        ax.semilogx(kappa, cdr_w, label=r'Vertical motion', linewidth=1.6,
                    linestyle='-.', color='black')

        ax.semilogx(kappa, cdr_v, label=r'Relative vorticity', linewidth=1.6,
                    linestyle='-.', color='red')

        # ax.semilogx(kappa, cdr_c, label=r'Coriolis effect', linewidth=1.6,
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

        y_limits = ke_limits[mode][level]

        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)

        prange_str = [int(1e-2 * p) for p in sorted(prange)]

        legend = ax.legend(title=r"{} ({:4d} - {:4d} hPa)".format(level, *prange_str),
                           loc='upper right', fontsize=14, ncol=2)
        ax.add_artist(legend)

        # fig.savefig('figures/{}_helmholtz_fluxes_{}_{}-{}_np.pdf'.format(
        #     model, resolution, *prange_str), dpi=300)

        plt.show()
        plt.close(fig)
