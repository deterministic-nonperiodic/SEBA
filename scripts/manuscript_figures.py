import os
import warnings

import xarray as xr

from io_tools import SebaDataset

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # Load dyamond dataset
    model = 'ICON'
    resolution = 'n1024'
    data_path = '../data'

    date_time = '20200125-30'

    file_names = os.path.join(data_path, f"{model}_energy_budget_{resolution}_{date_time}.nc")

    dataset_budget = SebaDataset(xr.open_mfdataset(file_names)).truncate(2047)

    # ----------------------------------------------------------------------------------------------
    # Visualization of Kinetic energy and Available potential energy
    # ----------------------------------------------------------------------------------------------
    layers = {'Free troposphere': [250e2, 450e2], 'Stratosphere': [50e2, 250e2]}

    figure_name = f'../figures/manuscript/{model}_energy_spectra_{resolution}.pdf'

    dataset_budget.visualize_energy(model=model, layers=layers, fig_name=figure_name)

    # ----------------------------------------------------------------------------------------------
    # Nonlinear transfer of Kinetic energy and Available potential energy
    # ----------------------------------------------------------------------------------------------
    # Perform vertical integration along last axis
    layers = {'Free troposphere': [250e2, 450e2], 'Stratosphere': [50e2, 250e2]}

    y_limits = {'Free troposphere': [-0.5, 1.2], 'Stratosphere': [-0.5, 1.2]}

    figure_name = f'../figures/manuscript/{model}_energy_budget_{resolution}.pdf'

    dataset_budget.visualize_fluxes(model=model,
                                    variables=['pi_hke+pi_ape', 'pi_hke',
                                               'pi_ape', 'cad', 'cdr', 'vfd_tot'],
                                    layers=layers, y_limits=y_limits, fig_name=figure_name)

    # ----------------------------------------------------------------------------------------------
    # Nonlinear transfer of Kinetic energy and Available potential energy
    # ----------------------------------------------------------------------------------------------
    figure_name = f'../figures/manuscript/{model}_hke_fluxes_{resolution}.pdf'

    layers = {'Free troposphere': [250e2, 450e2], 'Lower troposphere': [500e2, 850e2]}
    y_limits = {'Free troposphere': [-0.4, 0.4], 'Lower troposphere': [-0.4, 0.4]}

    # perform vertical integration
    dataset_budget.visualize_fluxes(model=model, show_injection=True,
                                    variables=['pi_dke+pi_rke', 'pi_rke', 'pi_dke',
                                               'cdr', 'cdr_w', 'cdr_v'], layers=layers,
                                    y_limits=y_limits, fig_name=figure_name)

    # ---------------------------------------------------------------------------------------
    # Visualize fluxes cross sections
    # ---------------------------------------------------------------------------------------
    figure_name = f'../figures/manuscript/{model}_dke_fluxes_section_{resolution}.pdf'

    dataset_budget.visualize_slices(variables=['cdr', 'vfd_dke'],
                                    y_limits=[1000., 98.], fig_name=figure_name)

    figure_name = f'../figures/manuscript/{model}_hke_fluxes_section_{resolution}.pdf'

    dataset_budget.visualize_slices(variables=['pi_hke', 'pi_rke', 'pi_dke',
                                               'cdr', 'cdr_v', 'cdr_w'],
                                    y_limits=[1000., 98.], fig_name=figure_name)
