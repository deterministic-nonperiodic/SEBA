import numpy as np
import xarray as xr

from io_tools import SebaDataset
from seba import EnergyBudget
from seba import parse_dataset

if __name__ == '__main__':

    model = 'ICON'
    data_path = '/media/yanm/Data/DYAMOND/data/'

    date_time = '2020020[2]'

    p_levels = np.linspace(1000e2, 10e2, 31)

    dataset = {}
    dataset_fluxes = {}
    for mode in ['IG', 'RO']:
        file_names = data_path + f"{model}_{mode}_inst_{date_time}_n256.nc"

        # parse dataset so interpolation is done only once
        dataset[mode] = parse_dataset(file_names, p_levels=p_levels)

        # compute cumulative energy fluxes
        dataset_fluxes[mode] = EnergyBudget(dataset[mode]).cumulative_energy_fluxes()

    # Create dataset with full winds
    add_var = ['u_wind', 'v_wind', 'omega']
    keep_var = ['temperature', 'pressure', 'geopotential', 'ps', 'ts']

    # replace dataset with full fields data
    dataset = xr.merge([dataset['RO'][add_var] + dataset['IG'][add_var], dataset['IG'][keep_var]])

    # Create energy budget object
    budget = EnergyBudget(dataset)

    # compute cumulative energy fluxes
    dataset_fluxes['FF'] = budget.cumulative_energy_fluxes()

    wave_mean_fluxes = SebaDataset(
        dataset_fluxes['FF'] - dataset_fluxes['IG'] - dataset_fluxes['RO']
    )

    # ---------------------------------------------------------------------------------------
    # Visualize fluxes cross section
    # ---------------------------------------------------------------------------------------
    variables = ['cdr', 'vfd_dke', 'pi_hke']

    fig_name = f'../figures/papers/{model}_total_fluxes_section_n256.pdf'
    dataset_fluxes['FF'].visualize_slices(model=None, variables=variables,
                                          y_limits=[1000., 10.], fig_name=fig_name)

    fig_name = f'../figures/papers/{model}_wave_vortex_fluxes_section_n256.pdf'
    wave_mean_fluxes.visualize_slices(model=None, variables=variables,
                                      y_limits=[1000., 10.], fig_name=fig_name)

    # ----------------------------------------------------------------------------------------------
    # Nonlinear transfer of Kinetic energy and Available potential energy
    # ----------------------------------------------------------------------------------------------

    # Perform vertical integration along last axis
    layers = {'Stratosphere': [20e2, 250e2], 'Free troposphere': [250e2, 450e2]}

    y_limits = {'Stratosphere': [-0.8, 1.2],
                'Free troposphere': [-0.5, 1.5],
                'Lower troposphere': [-1.0, 1.5]
                }

    figure_name = f'../figures/papers/{model}_total_energy_fluxes_n256.pdf'

    dataset_fluxes['FF'].visualize_fluxes(model=model,
                                          variables=['pi_hke+pi_ape', 'pi_hke',
                                                     'pi_ape', 'cad', 'cdr', 'vfd_tot'],
                                          layers=layers, y_limits=y_limits,
                                          fig_name=figure_name)
