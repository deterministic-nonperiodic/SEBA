import warnings

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from src.io_tools import parse_dataset
from src.seba import EnergyBudget
from src.visualization import fluxes_slices_by_models

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

    p_levels = np.linspace(1000e2, 10e2, 21)

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
