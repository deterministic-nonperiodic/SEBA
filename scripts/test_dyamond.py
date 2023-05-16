import warnings

import xarray as xr

from src.seba import EnergyBudget

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # Load dyamond dataset
    model = 'IFS'
    resolution = 'n512'
    data_path = '../data/'
    # data_path = '/mnt/levante/energy_budget/test_data/'

    date_time = '20[01]'
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
    layers = {
        'Troposphere': [250e2, 450e2],
        'Stratosphere': [50e2, 250e2]
    }

    figure_name = f'../figures/papers/{model}_energy_spectra_{resolution}.pdf'

    dataset_energy.visualize_energy(model=model, layers=layers,
                                    resolution='n2048', fig_name=figure_name)

    # ----------------------------------------------------------------------------------------------
    # Nonlinear transfer of Kinetic energy and Available potential energy
    # ----------------------------------------------------------------------------------------------
    # get nonlinear energy fluxes. Compute time-averaged cumulative fluxes
    dataset_fluxes = budget.nonlinear_energy_fluxes().cumulative_sum(dim='kappa')

    # Perform vertical integration along last axis
    layers = {'Stratosphere': [20e2, 250e2], 'Free troposphere': [250e2, 450e2]}

    y_limits = {'Stratosphere': [-0.8, 0.8],
                'Free troposphere': [-0.5, 1.0],
                'Lower troposphere': [-1.0, 1.5]
                }

    figure_name = f'../figures/papers/{model}_energy_fluxes_{resolution}.pdf'

    dataset_fluxes.visualize_fluxes(model=model,
                                    variables=['pi_hke+pi_ape', 'pi_hke',
                                               'pi_ape', 'cad', 'cdr', 'vfd_tot'],
                                    layers=layers, y_limits=y_limits,
                                    resolution='n2048', fig_name=figure_name)

    # ----------------------------------------------------------------------------------------------
    # Nonlinear transfer of Kinetic energy and Available potential energy
    # ----------------------------------------------------------------------------------------------
    figure_name = f'../figures/papers/{model}_hke_fluxes_{resolution}.pdf'

    layers = {'Free troposphere': [250e2, 450e2], 'Lower troposphere': [500e2, 850e2]}
    y_limits = {'Free troposphere': [-0.6, 0.6], 'Lower troposphere': [-0.6, 0.6]}

    # perform vertical integration
    dataset_fluxes.visualize_fluxes(model=model,
                                    variables=['pi_dke+pi_rke', 'pi_rke', 'pi_dke',
                                               'cdr', 'cdr_v', 'cdr_w'],
                                    layers=layers, y_limits=y_limits,
                                    resolution='n2048', fig_name=figure_name)

    # ---------------------------------------------------------------------------------------
    # Visualize fluxes cross section
    # ---------------------------------------------------------------------------------------
    figure_name = f'../figures/papers/{model}_fluxes_section_{resolution}.pdf'

    dataset_fluxes.visualize_slices(model=None, variables=['cdr', 'vfd_dke'],
                                    resolution='n1024', y_limits=[1000., 100.],
                                    fig_name=figure_name)
