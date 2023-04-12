import os.path
import warnings
from datetime import date

import numpy as np
import xarray as xr
from tqdm import tqdm

from src.seba import EnergyBudget

warnings.filterwarnings('ignore')

DATA_PATH = "data/"  # '/mnt/levante/energy_budget/test_data/'


def _process_model(model, resolution, date_time):
    # Load dyamond dataset
    date_time_op = date_time.replace("*", "")
    date_time_op = date_time_op.replace("?", "")
    suffix = "_".join([resolution, date_time_op])

    file_names = DATA_PATH + '{}_atm_3d_inst_{}_gps_{}.nc'
    dataset_dyn = file_names.format(model, resolution, date_time)

    dataset_tnd = xr.open_mfdataset(DATA_PATH + '{}_atm_3d_tend_{}_gps_{}.nc'.format(model,
                                                                                     resolution,
                                                                                     date_time))
    # load earth topography and surface pressure
    dset_sfc = xr.open_dataset(DATA_PATH + 'DYAMOND2_topography_{}.nc'.format(resolution))

    sfc_file = DATA_PATH + '{}_sfcp_{}_{}.nc'.format(model, date_time, resolution)
    if os.path.exists(sfc_file):
        dset_sfc = dset_sfc.update(xr.open_dataset(sfc_file))
    else:
        print("No surface pressure file found!")

    sfcp = dset_sfc.get('pres_sfc')

    # Create energy budget object
    budget = EnergyBudget(dataset_dyn, ps=sfcp, jobs=1)

    # ----------------------------------------------------------------------------------------------
    # Nonlinear transfer of Kinetic energy and Available potential energy
    # ----------------------------------------------------------------------------------------------
    # - Nonlinear energy transfers for DKE and RKE
    # - Energy conversion terms APE --> DKE and DKE --> RKE
    # - Vertical pressure and turbulent fluxes
    fluxes = budget.nonlinear_energy_fluxes()

    # Compute spectral energy diagnostics
    fluxes['hke'] = budget.horizontal_kinetic_energy()
    fluxes['ape'] = budget.available_potential_energy()
    fluxes['vke'] = budget.vertical_kinetic_energy()

    # ----------------------------------------------------------------------------------------------
    # Combine results into a dataset and export to netcdf
    # ----------------------------------------------------------------------------------------------
    fluxes.attrs.clear()  # clear global attributes

    attrs = {'Conventions': 'CF-1.6',
             'source': 'git@github.com:deterministic-nonperiodic/SEBA.git',
             'institution': 'Max Planck Institute for Meteorology',
             'title': '{}: Spectral Energy Budget of the Atmosphere.'.format(model.upper()),
             'history': date.today().strftime('Created on %c'),
             'references': ''}
    fluxes.attrs.update(attrs)

    fluxes.to_netcdf("data/energy_budget/{}_energy_fluxes_{}.nc".format(model, suffix))
    fluxes.close()

    # ----------------------------------------------------------------------------------------------
    # Compute APE tendency from parameterized processes
    # ----------------------------------------------------------------------------------------------
    ape_tendencies = []
    tendecies = ['ddt_temp_dyn', 'ddt_temp_radlw', 'ddt_temp_radsw',
                 'ddt_temp_rad', 'ddt_temp_conv', 'ddt_temp_vd',
                 'ddt_temp_gwd', 'ddt_temp_turb', 'ddt_temp_gscp']

    for name in tendecies:

        pname = name.split('_')[-1].lower()
        tend_grid = dataset_tnd.get(name)
        if tend_grid is not None:
            ape_tendencies.append(budget.get_ape_tendency(tend_grid, name="ddt_ape_" + pname,
                                                          cumulative=True))

    # ----------------------------------------------------------------------------------------------
    # Compute APE tendency from parameterized processes
    # ----------------------------------------------------------------------------------------------
    ke_tendencies = []
    tendecies = ['ddt_*_conv', 'ddt_*_vd', 'ddt_*_gwd', 'ddt_*_turb']

    for name in tendecies:

        pname = name.split('_')[-1].lower()

        tend_u_grid = dataset_tnd.get(name.replace("*", "u"))
        tend_v_grid = dataset_tnd.get(name.replace("*", "v"))

        if tend_u_grid is not None:
            tend_grid = np.moveaxis(np.array([tend_u_grid.values,
                                              tend_v_grid.values]), (1, 2), (-2, -1))

            ke_tendency = budget.get_ke_tendency(tend_grid, cumulative=True)

            ke_tendency = budget.add_field(ke_tendency, "ddt_ke_" + pname,
                                           gridtype='spectral', units='W m**-2')

            ke_tendencies.append(ke_tendency)

    dataset_tnd.close()
    # ----------------------------------------------------------------------------------------------
    # Combine results into a dataset and export to netcdf
    # ----------------------------------------------------------------------------------------------
    dataset = xr.merge(ape_tendencies + ke_tendencies, compat="no_conflicts")
    dataset.attrs.clear()  # clear global attributes
    attrs = {'Conventions': 'CF-1.6',
             'source': 'git@github.com:deterministic-nonperiodic/SEBA.git',
             'institution': 'Max Planck Institute for Meteorology',
             'title': '{}: Parameterized Spectral Energy Fluxes.'.format(model.upper()),
             'history': date.today().strftime('Created on %c'),
             'references': ''}
    dataset.attrs.update(attrs)

    dataset.to_netcdf(
        DATA_PATH + "/energy_budget/{}_physics_tendencies_{}.nc".format(model, suffix))
    dataset.close()


if __name__ == '__main__':

    for i in tqdm(range(6), desc="Computing spectral energy fluxes"):
        date_id = '20{:d}'.format(i)
        _process_model(model='IFS', resolution='n512', date_time=date_id)
        _process_model(model='ICON', resolution='n512', date_time=date_id)
