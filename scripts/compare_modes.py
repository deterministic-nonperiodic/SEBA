import warnings

import numpy as np

from seba.seba import EnergyBudget

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    mode = "IG"
    model = 'IFS'
    resolution = 'n256'
    data_path = '/media/yanm/Data/DYAMOND/data/'

    date_time = '20200202'
    file_names = data_path + f"{model}_{mode}_inst_{date_time}_{resolution}.nc"

    p_levels = np.linspace(1000e2, 10e2, 21)

    # Create energy budget object
    budget = EnergyBudget(file_names, p_levels=p_levels, truncation=511)

    # Compute diagnostics
    dataset_energy = budget.energy_diagnostics()

    # ----------------------------------------------------------------------------------------------
    # Visualization of Kinetic energy and Available potential energy
    # ----------------------------------------------------------------------------------------------
    layers = {'Troposphere': [250e2, 450e2], 'Stratosphere': [50e2, 250e2]}

    figure_name = f'../figures/tests/{model}_{mode}_energy_spectra_{resolution}.pdf'

    dataset_energy.visualize_energy(model=model, layers=layers, fig_name=figure_name)

    # ----------------------------------------------------------------------------------------------
    # Nonlinear transfer of Kinetic energy and Available potential energy
    # ----------------------------------------------------------------------------------------------
    dataset_fluxes = budget.cumulative_energy_fluxes()

    # Perform vertical integration along last axis
    layers = {'Free troposphere': [250e2, 450e2], 'Stratosphere': [20e2, 250e2]}

    y_limits = {'Stratosphere': [-0.4, 0.8],
                'Free troposphere': [-0.4, 0.8],
                'Lower troposphere': [-1.0, 1.5]}

    figure_name = f'../figures/tests/{model}_{mode}_energy_fluxes_{resolution}.pdf'

    dataset_fluxes.visualize_fluxes(model=model,
                                    variables=['pi_hke+pi_ape', 'pi_hke',
                                               'pi_ape', 'cad', 'cdr', 'vfd_tot'],
                                    layers=layers, y_limits=y_limits, fig_name=figure_name)

    # ----------------------------------------------------------------------------------------------
    # Nonlinear transfer of Kinetic energy and Available potential energy
    # ----------------------------------------------------------------------------------------------
    figure_name = f'../figures/tests/{model}_{mode}_hke_fluxes_{resolution}.pdf'

    layers = {'Free troposphere': [250e2, 450e2], 'Lower troposphere': [500e2, 850e2]}

    y_limits = {
        'IG': {'Free troposphere': [-0.04, 0.04], 'Lower troposphere': [-0.08, 0.08]},
        'RO': {'Free troposphere': [-0.3, 0.3], 'Lower troposphere': [-0.3, 0.3]}
    }

    # perform vertical integration
    dataset_fluxes.visualize_fluxes(model=model,
                                    variables=['pi_dke+pi_rke', 'pi_rke', 'pi_dke',
                                               'cdr', 'cdr_w', 'cdr_v'],
                                    layers=layers, y_limits=y_limits[mode], fig_name=figure_name)

    # ---------------------------------------------------------------------------------------
    # Visualize fluxes cross section
    # ---------------------------------------------------------------------------------------
    figure_name = f'../figures/tests/{model}_{mode}_dke_fluxes_section_{resolution}.pdf'

    dataset_fluxes.visualize_sections(model=None, variables=['cdr', 'vfd_dke'],
                                      y_limits=[1000., 100.], fig_name=figure_name)

    figure_name = f'../figures/tests/{model}_{mode}_hke_fluxes_section_{resolution}.pdf'

    dataset_fluxes.visualize_sections(model=model, share_cbar=True,
                                      variables=['pi_hke', 'pi_rke', 'pi_dke',
                                                 'cdr', 'cdr_v', 'cdr_w'],
                                      y_limits=[1000., 100.], fig_name=figure_name)
