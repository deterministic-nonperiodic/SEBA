import os
import warnings

import xarray as xr

from seba import EnergyBudget

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # Load dyamond dataset
    model = 'ICON'
    resolution = 'n1024'
    data_path = '../data/'

    date_time = '20[0]'
    file_names = data_path + f"{model}_atm_3d_inst_{resolution}_gps_{date_time}.nc"

    # load surface pressure if given externally
    sfc_file = data_path + '{}_sfcp_{}.nc'.format('ICON', resolution)
    if os.path.exists(sfc_file):
        sfc_pres = xr.open_dataset(sfc_file).get('pres_sfc')
    else:
        print("No surface pressure file found!")
        sfc_pres = None

    # Create energy budget object
    budget = EnergyBudget(file_names, ps=sfc_pres)

    # ----------------------------------------------------------------------------------------------
    # Visualization of Kinetic energy and Available potential energy
    # ----------------------------------------------------------------------------------------------
    dataset_energy = budget.energy_diagnostics()

    layers = {'Free troposphere': [250e2, 450e2], 'Stratosphere': [50e2, 250e2]}

    figure_name = f'../figures/papers/{model}_energy_spectra_{resolution}.pdf'

    dataset_energy.visualize_energy(model=model, layers=layers, fig_name=figure_name)

    # ----------------------------------------------------------------------------------------------
    # Nonlinear transfer of Kinetic energy and Available potential energy
    # ----------------------------------------------------------------------------------------------
    dataset_fluxes = budget.cumulative_energy_fluxes()

    # Perform vertical integration along last axis
    layers = {'Free troposphere': [250e2, 450e2], 'Stratosphere': [20e2, 250e2]}

    y_limits = {'Free troposphere': [-0.5, 1.2], 'Stratosphere': [-0.5, 1.2]}

    figure_name = f'../figures/papers/{model}_energy_fluxes_{resolution}.pdf'

    dataset_fluxes.visualize_fluxes(model=model,
                                    variables=['pi_hke+pi_ape', 'pi_hke',
                                               'pi_ape', 'cad', 'cdr', 'vfd_tot'],
                                    layers=layers, y_limits=y_limits, fig_name=figure_name)

    # ----------------------------------------------------------------------------------------------
    # Nonlinear transfer of Kinetic energy and Available potential energy
    # ----------------------------------------------------------------------------------------------
    figure_name = f'../figures/papers/{model}_hke_fluxes_{resolution}.pdf'

    layers = {'Free troposphere': [250e2, 450e2], 'Lower troposphere': [500e2, 850e2]}
    y_limits = {'Free troposphere': [-0.4, 0.4], 'Lower troposphere': [-0.4, 0.4]}

    # perform vertical integration
    dataset_fluxes.visualize_fluxes(model=model, show_injection=True,
                                    variables=['pi_dke+pi_rke', 'pi_rke', 'pi_dke',
                                               'cdr', 'cdr_w', 'cdr_v'], layers=layers,
                                    y_limits=y_limits, fig_name=figure_name)

    # ---------------------------------------------------------------------------------------
    # Visualize fluxes cross sections
    # ---------------------------------------------------------------------------------------
    figure_name = f'../figures/papers/{model}_fluxes_section_{resolution}.pdf'

    dataset_fluxes.visualize_slices(variables=['cdr', 'vfd_dke'], y_limits=[1000., 100.],
                                    fig_name=figure_name)
