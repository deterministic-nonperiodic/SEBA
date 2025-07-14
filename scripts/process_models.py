import os
import warnings
from datetime import date

import xarray as xr
from tqdm import tqdm

from seba.seba import EnergyBudget

warnings.filterwarnings('ignore')

DATA_PATH = "../data/"
OUTPUT_PATH = "../data/"


def _process_model(model, resolution, date_time):
    # Load dyamond dataset
    date_time_op = date_time.replace("*", "")
    date_time_op = date_time_op.replace("?", "")
    suffix = "_".join([resolution, date_time_op])

    file_names = DATA_PATH + f'{model}_atm_3d_inst_{resolution}_gps_{date_time}.nc'

    # load surface pressure if given externally
    sfc_file = DATA_PATH + '{}_sfcp_{}.nc'.format(model, resolution)
    if os.path.exists(sfc_file):
        sfc_pres = xr.open_dataset(sfc_file).get('pres_sfc')
    else:
        print("No surface pressure file found!")
        sfc_pres = None

    # Initialize energy budget
    budget = EnergyBudget(file_names, ps=sfc_pres)

    # ----------------------------------------------------------------------------------------------
    # Nonlinear transfer of Kinetic energy and Available potential energy
    # ----------------------------------------------------------------------------------------------
    # - Nonlinear energy transfers for DKE and RKE
    # - Energy conversion terms APE --> DKE and DKE --> RKE
    # - Vertical pressure and turbulent fluxes
    dataset = budget.cumulative_energy_fluxes()

    # Compute spectral energy diagnostics and combine with fluxes
    dataset['rke'], dataset['dke'], dataset['hke'] = budget.horizontal_kinetic_energy()
    dataset['ape'] = budget.available_potential_energy()
    dataset['vke'] = budget.vertical_kinetic_energy()

    # ----------------------------------------------------------------------------------------------
    # Export dataset to netcdf file
    # ----------------------------------------------------------------------------------------------
    attrs = {  # update dataset name and creation date/time
        'title': '{}: Spectral Energy Budget of the Atmosphere.'.format(model.upper()),
        'history': date.today().strftime('Created on %c')
    }
    dataset.attrs.update(attrs)

    dataset.to_netcdf(os.path.join(OUTPUT_PATH, f"{model}_energy_fluxes_{suffix}.nc"))
    dataset.close()


def main():
    # Processing models
    for i in tqdm(range(20, 30), desc="Computing ERA5 spectral energy fluxes"):
        _process_model(model='ERA5', resolution='025deg', date_time='202001{:02d}'.format(i))

    for i in tqdm(range(40), desc="Computing IFS spectral energy fluxes"):
        _process_model(model='IFS', resolution='n1024', date_time='2{:02d}'.format(i))

    for i in tqdm(range(20, 30), desc="Computing ICON spectral energy fluxes"):
        _process_model(model='ICON', resolution='n1024', date_time='202001{:02d}*'.format(i))


if __name__ == '__main__':
    main()
