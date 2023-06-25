import os
import warnings

import xarray as xr

from io_tools import SebaDataset
from visualization import compare_model_fluxes, compare_model_energy

warnings.filterwarnings('ignore')

data_path = '../data'


def create_figures(model, resolution, date_time='20200125-30'):
    # Load dyamond dataset
    file_names = os.path.join(data_path, f"{model}_energy_budget_{resolution}_{date_time}.nc")

    dataset_budget = SebaDataset(xr.open_mfdataset(file_names)).truncate(2047)

    # ----------------------------------------------------------------------------------------------
    # Visualization of Kinetic energy and Available potential energy
    # ----------------------------------------------------------------------------------------------
    layers = {'Troposphere': [250e2, 450e2], 'Stratosphere': [100e2, 250e2]}

    figure_name = f'../figures/manuscript/{model}_energy_spectra_{resolution}.pdf'

    dataset_budget.visualize_energy(model=model, layers=layers, fig_name=figure_name)

    # ----------------------------------------------------------------------------------------------
    # Nonlinear transfer of Kinetic energy and Available potential energy
    # ----------------------------------------------------------------------------------------------

    layers = {'': [10e2, 1000e2]}
    y_limits = {'': [-1., 3.4]}

    figure_name = f'../figures/manuscript/{model}_total_energy_budget_{resolution}.pdf'

    dataset_budget.visualize_fluxes(model=model,
                                    variables=['pi_hke+pi_ape', 'pi_hke',
                                               'pi_ape', 'cad', 'cdr', 'vfd_tot'],
                                    layers=layers, y_limits=y_limits,
                                    fig_name=figure_name)

    layers = {'Troposphere': [250e2, 450e2], 'Stratosphere': [100e2, 250e2]}
    y_limits = {'Troposphere': [-0.6, 1.2], 'Stratosphere': [-0.6, 1.2]}

    figure_name = f'../figures/manuscript/{model}_energy_budget_{resolution}.pdf'

    dataset_budget.visualize_fluxes(model=model,
                                    variables=['pi_hke+pi_ape', 'pi_hke',
                                               'pi_ape', 'cad', 'cdr', 'vfd_tot'],
                                    layers=layers, y_limits=y_limits,
                                    fig_name=figure_name)

    # ----------------------------------------------------------------------------------------------
    # Nonlinear transfer of Kinetic energy and Available potential energy
    # ----------------------------------------------------------------------------------------------
    figure_name = f'../figures/manuscript/{model}_hke_fluxes_{resolution}.pdf'

    # 'Lower troposphere': [500e2, 850e2],
    # 'Lower troposphere': [-0.5, 0.5]
    layers = {'Troposphere': [250e2, 450e2], 'Stratosphere': [100e2, 250e2]}
    y_limits = {'Troposphere': [-0.6, 0.8], 'Stratosphere': [-0.6, 0.8]}

    # perform vertical integration
    dataset_budget.visualize_fluxes(model=model, show_injection=True,
                                    variables=['pi_rke+pi_dke', 'pi_rke', 'pi_dke',
                                               'cdr', 'cdr_v', 'cdr_w'], layers=layers,
                                    y_limits=y_limits, fig_name=figure_name)

    # ---------------------------------------------------------------------------------------
    # Visualize fluxes cross sections
    # ---------------------------------------------------------------------------------------
    # figure_name = f'../figures/manuscript/{model}_dke_fluxes_section_{resolution}.pdf'
    #
    # dataset_budget.visualize_sections(variables=['vfd_dke', 'cdr'], share_cbar=True,
    #                                   y_limits=[1000., 100.], fig_name=figure_name)

    figure_name = f'../figures/manuscript/{model}_hke_fluxes_section_{resolution}.pdf'

    dataset_budget.visualize_sections(model=model, share_cbar=True,
                                      variables=['pi_hke', 'pi_rke', 'pi_dke',
                                                 'cdr', 'cdr_v', 'cdr_w'],
                                      y_limits=[1000., 100.], fig_name=figure_name)


def compare_models(model_datasets, date_time='20200125-29'):
    # Load dyamond dataset
    model_dataset = {}
    model_names = list(model_datasets.keys())
    for model_name, resolution in model_datasets.items():
        file_name = f"{model_name}_energy_budget_{resolution}_{date_time}.nc"
        file_name = os.path.join(data_path, file_name)
        dataset = SebaDataset(xr.open_dataset(file_name)).truncate(2047)

        for varname in dataset:
            if 'pi' in varname:
                dataset[varname].values[..., :2] = 0.0

        model_dataset[model_name] = dataset

    # -----------------------------------------------------------------------------------------------
    # Compare resolved hke fluxes among models
    # -----------------------------------------------------------------------------------------------
    layers = {'Troposphere': [250e2, 450e2], 'Stratosphere': [100e2, 250e2]}
    y_limits = {'Troposphere': [-0.4, 0.6], 'Stratosphere': [-0.4, 0.6]}

    figure = compare_model_fluxes(model_dataset, models=model_names,
                                  show_injection=True,
                                  variables=['pi_rke+pi_dke', 'pi_rke', 'pi_dke',
                                             'cdr_c', 'cdr_v', 'cdr_w'],
                                  layers=layers, y_limits=y_limits)
    figure.show()

    figure_name = '../figures/manuscript/{}_hke_fluxes_n1024.pdf'.format('-'.join(model_names))
    figure.savefig(figure_name, dpi=300)

    # -----------------------------------------------------------------------------------------------
    # Compare resolved fluxes among models
    # -----------------------------------------------------------------------------------------------
    layers = {'Troposphere': [250e2, 450e2], 'Stratosphere': [100e2, 250e2]}
    y_limits = {'Troposphere': [-0.6, 1.2], 'Stratosphere': [-0.6, 0.8]}

    figure = compare_model_fluxes(model_dataset, models=model_names,
                                  show_injection=True,
                                  variables=['pi_hke+pi_ape', 'pi_hke',
                                             'pi_ape', 'cad', 'cdr', 'vfd_tot'],
                                  layers=layers, y_limits=y_limits)
    figure.show()

    figure_name = '../figures/manuscript/{}_energy_budget_n1024.pdf'.format('-'.join(model_names))
    figure.savefig(figure_name, dpi=300)

    # ------------------------------------------------------------------------------------------
    # Total atmosphere integrated energy
    # ------------------------------------------------------------------------------------------
    layers = {'': [100e2, 1000e2]}

    figure = compare_model_energy(model_dataset, models=model_names,
                                  show_crossing=False, layers=layers,
                                  y_limits={'': [7e-5, 1.2e8]},
                                  start_index='a')
    figure.show()

    figure_name = '../figures/manuscript/{}_total_kinetic_energy_n1024.pdf'.format(
        '-'.join(model_names))
    figure.savefig(figure_name, dpi=300)

    figure = compare_model_fluxes(model_dataset, models=model_names,
                                  show_injection=True,
                                  variables=['pi_hke+pi_ape', 'pi_hke',
                                             'pi_ape', 'cad', 'cdr', 'vfd_tot'],
                                  layers=layers, y_limits={'': [-1.0, 3.2]},
                                  start_index='d')
    figure.show()

    figure_name = '../figures/manuscript/{}_total_energy_budget_n1024.pdf'.format(
        '-'.join(model_names))
    figure.savefig(figure_name, dpi=300)


if __name__ == '__main__':
    models = {
        'ERA5': 'n256',
        'IFS': 'n1024',
        'ICON': 'n1024',
        # 'ICON_IG': 'n256',
        # 'ICON_RO': 'n256',
        # 'ICON_FF': 'n256',
        # 'IFS_IG': 'n256',
        # 'IFS_RO': 'n256',
        # 'IFS_FF': 'n256',
    }

    # for model_name, grid in models.items():
    #     # create figures for each model independently
    #     create_figures(model=model_name, resolution=grid, date_time='20200125-30')

    compare_models(models, date_time='20200125-29')
