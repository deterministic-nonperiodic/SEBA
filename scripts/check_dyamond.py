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

    # load earth topography and surface pressure
    dataset_sfc = xr.open_dataset(data_path + 'ICON_sfcp_{}.nc'.format(resolution))
    sfc_pres = dataset_sfc.pres_sfc

    # Create energy budget object
    budget = EnergyBudget(file_names, ps=sfc_pres)

    # Compute diagnostics
    dataset_energy = budget.energy_diagnostics()

    # ----------------------------------------------------------------------------------------------
    # Visualization of Kinetic energy and Available potential energy
    # ----------------------------------------------------------------------------------------------
    layers = {'Troposphere': [250e2, 450e2], 'Stratosphere': [50e2, 250e2]}

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
    # Visualize fluxes cross section
    # ---------------------------------------------------------------------------------------
    figure_name = f'../figures/papers/{model}_fluxes_section_{resolution}.pdf'

    dataset_fluxes.visualize_slices(model=None, variables=['cdr', 'vfd_dke'],
                                    y_limits=[1000., 100.], fig_name=figure_name)
