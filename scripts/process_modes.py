import os
import warnings
from datetime import date

import numpy as np
import xarray as xr
from tqdm import tqdm

from seba.io_tools import parse_dataset
from seba.seba import EnergyBudget

warnings.filterwarnings('ignore')

DATA_PATH = '/media/yanm/Data/DYAMOND/data/'
OUTPUT_PATH = "../data/"


def _process_model(model, resolution='n256', date_time='*'):
    # Load dyamond dataset
    date_time_op = date_time.replace("*", "").replace("?", "")
    suffix = "_".join([resolution, date_time_op])

    # target horizontal truncation and vertical levels
    truncation = 511
    p_levels = np.linspace(1000e2, 10e2, 31)

    dataset = {}
    fluxes = {}
    for mode in ['IG', 'RO']:
        # load atmospheric dataset
        file_name = f"{model}_{mode}_inst_{resolution}_gps_{date_time}.nc"
        file_name = os.path.join(DATA_PATH, file_name)

        output_file = os.path.join(OUTPUT_PATH, f"{model}_{mode}_energy_fluxes_{suffix}.nc")

        if os.path.exists(output_file):
            print(f"File {output_file}, already exists")
            return

        # parse dataset so interpolation is done only once
        dataset[mode] = parse_dataset(file_name, p_levels=p_levels)

        # ------------------------------------------------------------------------------------
        # Nonlinear transfer of Kinetic energy and Available potential energy
        # ------------------------------------------------------------------------------------
        # - Nonlinear energy transfers for DKE and RKE
        # - Energy conversion terms APE --> DKE and DKE --> RKE
        # - Vertical pressure and turbulent fluxes
        budget = EnergyBudget(dataset[mode], truncation=truncation)
        fluxes[mode] = budget.cumulative_energy_fluxes()

        hke = budget.horizontal_kinetic_energy()
        fluxes[mode]['rke'], fluxes[mode]['dke'], fluxes[mode]['hke'] = hke
        fluxes[mode]['ape'] = budget.available_potential_energy()
        fluxes[mode]['vke'] = budget.vertical_kinetic_energy()

        for varname in fluxes[mode]:
            if 'pi' in varname:
                fluxes[mode][varname].values[..., :2] = 0.0
        # ------------------------------------------------------------------------------------
        # Export datasets to netcdf files
        # ------------------------------------------------------------------------------------
        attrs = {
            'title': '{}: Spectral Energy Budget of the Atmosphere.'.format(model.upper()),
            'history': date.today().strftime('Created on %c')
        }

        fluxes[mode].attrs.update(attrs)
        fluxes[mode].to_netcdf(output_file)
        fluxes[mode].close()

    # Create dataset with full winds
    add_var = ['u_wind', 'v_wind', 'omega']
    keep_var = ['temperature', 'pressure', 'geopotential', 'ps', 'ts']

    # replace dataset with full fields data
    dataset = xr.merge([dataset['RO'][add_var] + dataset['IG'][add_var], dataset['IG'][keep_var]])

    # Create energy budget object
    budget = EnergyBudget(dataset, truncation=truncation)

    # compute cumulative energy fluxes for total fields
    fluxes = budget.cumulative_energy_fluxes()

    fluxes['rke'], fluxes['dke'], fluxes['hke'] = budget.horizontal_kinetic_energy()
    fluxes['ape'] = budget.available_potential_energy()
    fluxes['vke'] = budget.vertical_kinetic_energy()

    # remove small truncation errors from fluxes at wavenumber l=0
    for varname in fluxes:
        if 'pi' in varname:
            fluxes[varname].values[..., :2] = 0.0
    # ------------------------------------------------------------------------------------
    # Export datasets to netcdf files
    # ------------------------------------------------------------------------------------
    output_file = os.path.join(OUTPUT_PATH, f"{model}_FF_energy_fluxes_{suffix}.nc")

    attrs = {
        'title': '{}: Spectral Energy Budget of the Atmosphere.'.format(model.upper()),
        'history': date.today().strftime('Created on %c')
    }
    fluxes.attrs.update(attrs)
    fluxes.to_netcdf(output_file)
    fluxes.close()


def main():
    # Processing models
    for model in ['ICON', 'IFS']:
        for i in tqdm(range(25, 26), desc=f"{model} spectral energy budget"):
            _process_model(model, resolution='n256', date_time='202001{:02d}???'.format(i))
        print('-----------------------------------------------------------------------------')


if __name__ == '__main__':
    main()
