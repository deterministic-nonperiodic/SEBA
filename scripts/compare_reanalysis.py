import os
import warnings

import xarray as xr

from seba import EnergyBudget

warnings.filterwarnings('ignore')

print("-------------------------- Memory usage -----------------------------")
memory_usage = os.popen('free -th').readlines()[-1].split()[1:]
print('Total: {} -- Used: {} -- Free: {}'.format(*memory_usage))
print("---------------------------------------------------------------------")

if __name__ == '__main__':
    # Load dyamond dataset
    model = 'ERA5'
    resolution = '025deg'
    data_path = '/media/yanm/Data/DYAMOND/simulations/'

    date_time = '20200128'
    file_names = data_path + f"{model}_atm_3d_inst_{resolution}_gps_{date_time}.nc"

    # load surface pressure
    sfc_pres = xr.open_dataset(data_path + 'reanalysis/ERA5_atm_ps_025deg_gps_202001.nc')
    sfc_pres = sfc_pres.sp.sel(time=date_time)

    # Create energy budget object. Set the truncation
    budget = EnergyBudget(file_names, truncation=511, ps=sfc_pres)

    # Compute diagnostics
    dataset_energy = budget.energy_diagnostics()

    # ----------------------------------------------------------------------------------------------
    # Visualization of Kinetic energy and Available potential energy
    # ----------------------------------------------------------------------------------------------
    layers = {'Troposphere': [250e2, 450e2], 'Stratosphere': [50e2, 250e2]}

    figure_name = f'../figures/tests/{model}_energy_spectra_{resolution}.pdf'

    dataset_energy.visualize_energy(model=model, layers=layers, fig_name=figure_name)

    # ----------------------------------------------------------------------------------------------
    # Nonlinear transfer of Kinetic energy and Available potential energy
    # ----------------------------------------------------------------------------------------------
    dataset_fluxes = budget.cumulative_energy_fluxes()

    # Perform vertical integration along last axis
    layers = {'Lower troposphere': [450e2, 850e2],
              'Free troposphere': [250e2, 450e2], 'Stratosphere': [20e2, 250e2]}

    y_limits = {'Stratosphere': [-0.6, 1.0],
                'Free troposphere': [-0.6, 1.0],
                'Lower troposphere': [-0.9, 1.5]}

    figure_name = f'../figures/tests/{model}_energy_fluxes_{resolution}_masked.pdf'

    dataset_fluxes.visualize_fluxes(model=model,
                                    variables=['pi_hke+pi_ape', 'pi_hke',
                                               'pi_ape', 'cad', 'cdr', 'vfd_tot'],
                                    layers=layers, y_limits=y_limits, fig_name=figure_name)
    # ----------------------------------------------------------------------------------------------
    # Nonlinear transfer of Kinetic energy and Available potential energy
    # ----------------------------------------------------------------------------------------------
    figure_name = f'../figures/tests/{model}_hke_fluxes_{resolution}_masked.pdf'

    layers = {'Free troposphere': [250e2, 450e2], 'Lower troposphere': [450e2, 850e2]}
    y_limits = {'Free troposphere': [-0.6, 0.6], 'Lower troposphere': [-0.6, 0.6]}

    # perform vertical integration
    dataset_fluxes.visualize_fluxes(model=model,
                                    variables=['pi_dke+pi_rke', 'pi_rke', 'pi_dke', 'cdr',
                                               'cdr_w', 'cdr_v'],
                                    layers=layers, y_limits=y_limits, fig_name=figure_name)
    # ---------------------------------------------------------------------------------------
    # Visualize fluxes cross section
    # ---------------------------------------------------------------------------------------
    figure_name = f'../figures/tests/{model}_fluxes_section_{resolution}_masked.pdf'

    dataset_fluxes.visualize_sections(model=None, variables=['cdr', 'vfd_dke'],
                                      y_limits=[1000., 98.], fig_name=figure_name)
