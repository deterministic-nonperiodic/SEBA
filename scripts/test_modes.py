import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

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

    mode = "IG"
    model = 'ICON'
    resolution = 'n256'
    data_path = '/media/yanm/Data/DYAMOND/data/'

    date_time = '20200202'
    file_names = data_path + f"{model}_{mode}_inst_{date_time}_{resolution}.nc"

    p_levels = np.linspace(1000e2, 10e2, 21)

    # Create energy budget object
    budget = EnergyBudget(file_names, p_levels=p_levels)

    # Compute diagnostics
    dataset_energy = budget.energy_diagnostics()

    layers = {
        'Troposphere': [250e2, 500e2],
        'Stratosphere': [50e2, 250e2]
    }

    # ----------------------------------------------------------------------------------------------
    # Visualization of Kinetic energy and Available potential energy
    # ----------------------------------------------------------------------------------------------
    kappa = 1e3 * dataset_energy.kappa.values  # km^-1

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

    ls = ['-', '--']
    for i, (layer, prange) in enumerate(layers.items()):
        data = dataset_energy.integrate_range(coord_range=prange).mean(dim='time')

        ax.loglog(kappa, data.hke, label=r'$E_K$', linewidth=1.2, linestyle=ls[i], color='red')
        ax.loglog(kappa, data.ape, label=r'$E_A$', linewidth=1.2, linestyle=ls[i], color='navy')
        ax.loglog(kappa, data.vke, label=r'$E_w$', linewidth=1.2, linestyle=ls[i], color='black')

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

    ax.set_xlabel(r'Spherical harmonic degree', fontsize=14, labelpad=4)
    secax.set_xlabel(r'Spherical wavelength $(km)$', fontsize=14, labelpad=5)

    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.legend(title=" / ".join(layers.keys()), loc='upper right', fontsize=12, ncol=2)

    plt.show()
    plt.close(fig)

    # ----------------------------------------------------------------------------------------------
    # Nonlinear transfer of Kinetic energy and Available potential energy
    # ----------------------------------------------------------------------------------------------
    kappa = 1e3 * budget.kappa_h

    # ----------------------------------------------------------------------------------------------
    # Load computed fluxes
    # ----------------------------------------------------------------------------------------------
    dataset_fluxes = budget.nonlinear_energy_fluxes().cumulative_sum(dim='kappa').mean(dim='time')

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
        data = dataset_fluxes.integrate_range(coord_range=prange)

        pid = data.pi_dke.values
        pir = data.pi_rke.values

        cdr_w = data.cdr_w.values
        cdr_v = data.cdr_v.values
        cdr_c = data.cdr_c.values
        cdr = data.cdr.values

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

    # ---------------------------------------------------------------------------------------
    # Visualize fluxes cross section
    # ---------------------------------------------------------------------------------------
    figure_name = '../figures/{}_wave_fluxes_section_{}.pdf'.format(model, resolution)

    fluxes_slices_by_models(dataset_fluxes, model=None, variables=['cdr', 'vfd_dke'],
                            resolution='n1024', y_limits=[1000., 10.], fig_name=figure_name)
