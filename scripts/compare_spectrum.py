import warnings

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from seba.seba import EnergyBudget

params = {'xtick.labelsize': 'medium',
          'ytick.labelsize': 'medium',
          'text.usetex': False, 'font.size': 15,
          'font.family': 'serif', 'font.weight': 'normal'}
plt.rcParams['legend.title_fontsize'] = 14
plt.rcParams.update(params)

warnings.filterwarnings('ignore')

if __name__ == '__main__':

    # Load dyamond dataset
    model = 'ICON'
    resolution = 'n1536'
    data_path = '/media/yanm/Data/DYAMOND/simulations/'

    date_time = '20200125_000001'
    file_names = data_path + f"{model}_atm_3d_inst_{resolution}_gps_{date_time}.nc"

    # # load earth topography and surface pressure
    dataset_sfc = xr.open_dataset(data_path + 'ICON_sfcp_{}.nc'.format(resolution))
    sfc_pres = dataset_sfc.pres_sfc

    # Create energy budget object
    p_levels = np.linspace(1000e2, 100e2, 10)
    budget = EnergyBudget(file_names, ps=sfc_pres, p_levels=p_levels)

    # compute mask
    beta = (~budget.theta_prime.mask).astype(float)

    # no mode-coupling assumption
    f_sky = budget.representative_mean(beta)

    # visualize profiles
    variables = ['omega', 'theta_prime', 'wind_rot', 'wind_div']
    vars_info = {
        'omega': ('scalar', r'Pressure vertical velocity $/~Pa^{2}~s^{-2}$'),
        'phi': ('scalar', r'${\phi}^{2}~(K^{2})$'),
        'theta_prime': ('scalar', r'${\theta^{\prime}}^{2}~/~K^{2}$'),
        'wind': ('vector', r'Horizontal kinetic energy  $/~m^{2}~s^{-2}$'),
        'wind_div': ('vector', r'Divergent kinetic energy  $/~m^{2}~s^{-2}$'),
        'wind_rot': ('vector', r'Rotational kinetic energy  $/~m^{2}~s^{-2}$'),
    }
    pressure = 1e-2 * budget.pressure

    n_cols = len(variables)
    fig, axes = plt.subplots(nrows=1, ncols=n_cols, figsize=(n_cols * 5, 10.0),
                             constrained_layout=True)

    results = {}
    lines = []
    # Compute spectrum of scalar variables
    for i, variable in enumerate(variables):
        ax = axes[i]
        data = budget.__dict__[variable]

        if vars_info[variable][0] == 'vector':
            # The global average of the dot product of two vectors must equal the sum
            # of the vectors' cross-spectrum along all spherical harmonic degrees.
            data_gs = budget.representative_mean(np.ma.sum(data ** 2, axis=0)).mean(0)
            data_clm = budget._vector_spectrum(data)
        else:
            data_gs = budget.representative_mean(data ** 2).mean(0)
            data_clm = budget._scalar_spectrum(data)

        masked = np.ma.is_masked(data)

        data_sp = budget.cumulative_spectrum(data_clm, mask_correction=masked)
        data_nc = budget.cumulative_spectrum(data_clm, mask_correction=False)

        # sum over all spherical harmonic degrees
        data_sp = np.nansum(data_sp, axis=0).mean(0)
        data_nc = np.nansum(data_nc, axis=0).mean(0)

        # plot vertical profiles of reconstructed mean
        lines = ax.plot(data_gs.T, pressure, '-b',
                        data_sp.T, pressure, '--k',
                        data_nc.T, pressure, '-.g', lw=2.5)

        ax.set_ylim(1020, 50)
        ax.set_xlabel(vars_info[variable][1])

    labels = [
        'physical space',
        'corrected spectrum',
        'uncorrected spectrum'
    ]
    axes[0].legend(lines, labels, title='Integrated global variance', loc='best')
    axes[0].set_ylabel('Pressure / hPa')

    plt.show()
    plt.close(fig)
