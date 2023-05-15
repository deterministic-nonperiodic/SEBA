import warnings

from src.seba import EnergyBudget
from src.visualization import fluxes_slices_by_models

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # Load dyamond dataset
    model = 'ERA5'
    resolution = '025deg'
    data_path = '../data/'

    date_time = '20200128'
    file_names = data_path + f"{model}_atm_3d_inst_{resolution}_gps_{date_time}.nc"

    # Create energy budget object. Set the truncation
    budget = EnergyBudget(file_names, truncation=480)

    # Compute diagnostics
    dataset_energy = budget.energy_diagnostics()

    # ----------------------------------------------------------------------------------------------
    # Visualization of Kinetic energy and Available potential energy
    # ----------------------------------------------------------------------------------------------
    layers = {
        'Troposphere': [250e2, 450e2],
        'Stratosphere': [50e2, 250e2]
    }

    dataset_energy.visualize_energy(model=model, layers=layers,
                                    fig_name=f'../figures/{model}_energy_spectra_{resolution}.pdf')

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

    dataset_fluxes.visualize_fluxes(model=model,
                                    variables=['pi_hke+pi_ape', 'pi_hke',
                                               'pi_ape', 'cad', 'cdr', 'vfd_tot'],
                                    layers=layers, y_limits=y_limits, resolution='n1024')

    # ----------------------------------------------------------------------------------------------
    # Nonlinear transfer of Kinetic energy and Available potential energy
    # ----------------------------------------------------------------------------------------------

    layers = {'Free troposphere': [250e2, 450e2], 'Lower troposphere': [500e2, 850e2]}
    y_limits = {'Free troposphere': [-0.8, 0.8], 'Lower troposphere': [-0.8, 0.8]}

    # perform vertical integration
    dataset_fluxes.visualize_fluxes(model=model,
                                    variables=['pi_dke+pi_rke', 'pi_rke', 'pi_dke',
                                               'dis_hke', 'cdr', 'cdr_w', 'cdr_v', 'cdr_c'],
                                    layers=layers, y_limits=y_limits, resolution='n1024')

    # ---------------------------------------------------------------------------------------
    # Visualize fluxes cross section
    # ---------------------------------------------------------------------------------------
    figure_name = '../figures/{}_fluxes_section_{}.pdf'.format(model, resolution)

    fluxes_slices_by_models(dataset_fluxes, model=None, variables=['cdr', 'vfd_dke'],
                            resolution='n1024', y_limits=[1000., 100.], fig_name=figure_name)
