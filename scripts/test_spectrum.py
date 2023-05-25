import warnings

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from seba import EnergyBudget

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
    resolution = 'n1024'
    data_path = '../data/'
    # data_path = '/mnt/levante/energy_budget/test_data/'

    date_time = '20[0]'
    file_names = data_path + f"{model}_atm_3d_inst_{resolution}_gps_{date_time}.nc"

    # # load earth topography and surface pressure
    dataset_sfc = xr.open_dataset(data_path + 'ICON_sfcp_{}.nc'.format(resolution))
    sfc_pres = dataset_sfc.pres_sfc

    # Create energy budget object
    budget = EnergyBudget(file_names, ps=sfc_pres)

    # compute mask
    beta = (~budget.theta_prime.mask).astype(float)

    # no mode-coupling assumption
    f_sky = budget.representative_mean(beta)

    # visualize profiles
    variables = ['omega', 'wind', 'theta_prime', 'theta']
    vars_info = {
        'omega': ('scalar', r'Pressure velocity $(Pa^{2}~s^{-2})$'),
        'theta': ('scalar', r'${\theta}^{2}~(K^{2})$'),
        'theta_prime': ('scalar', r'${\theta^{\prime}}^{2}~(K^{2})$'),
        'wind': ('vector', r'Horizontal kinetic energy  $(m^{2}~s^{-2})$')
    }
    pressure = 1e-2 * budget.pressure

    n_cols = len(variables)
    fig, axes = plt.subplots(nrows=1, ncols=n_cols, figsize=(n_cols * 5, 10.0),
                             constrained_layout=True)

    results = {}
    # Compute spectrum of scalar variables
    for i, variable in enumerate(variables):
        ax = axes[i]
        data = budget.__dict__[variable]

        if vars_info[variable][0] == 'vector':
            # The global average of the dot product of two vectors must equal the sum
            # of the vectors' cross-spectrum along all spherical harmonic degrees.
            data_sqd = np.sum(data ** 2, axis=0)
            data_gs = budget.representative_mean(data_sqd).mean(0)
            data_sp = budget.cumulative_spectrum(budget._vector_spectrum(data))
        else:
            data_sqd = data ** 2
            data_gs = budget.representative_mean(data_sqd).mean(0)
            data_sp = budget.cumulative_spectrum(budget._scalar_spectrum(data))

        # sum over all spherical harmonic degrees
        data_sp = np.nansum(data_sp, axis=0).mean(0)

        # plot vertical profiles of reconstructed mean
        lines = ax.plot(data_gs.T, pressure, '-b',
                        data_sp.T, pressure, '-.k', lw=2.5)
        labels = [
            'global mean',
            'recovered',
        ]
        ax.legend(lines, labels, title=vars_info[variable][1], loc='best')
        ax.set_ylim(1020, 20)

    axes[0].set_ylabel('Pressure (hPa)')

    plt.show()
    plt.close(fig)
