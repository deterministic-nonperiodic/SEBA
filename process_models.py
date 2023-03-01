import os.path
import warnings

import numpy as np
import xarray as xr
from tqdm import tqdm

from seba import EnergyBudget
from tools import cumulative_flux

warnings.filterwarnings('ignore')

DATA_PATH = "data/"  # '/mnt/levante/energy_budget/test_data/'


def reduce_to_1d(func, data, dim="plev", **kwargs):
    res = xr.apply_ufunc(func, data, input_core_dims=[[dim]],
                         kwargs=kwargs, dask='allowed', vectorize=True)
    return res.mean(dim='time')


def _process_model(model, resolution, date_time):
    # Load dyamond dataset
    date_time_op = date_time.replace("*", "")
    date_time_op = date_time_op.replace("?", "")
    suffix = "_".join([resolution, date_time_op])

    file_names = DATA_PATH + '{}_atm_3d_inst_{}_gps_{}.nc'

    dataset_dyn = xr.open_mfdataset(file_names.format(model, resolution, date_time))
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

    sfc_hgt = dset_sfc.get('topography_c')
    sfc_pres = dset_sfc.get('pres_sfc')

    # Create energy budget object
    budget = EnergyBudget(dataset_dyn, ghsl=sfc_hgt, ps=sfc_pres, filter_terrain=False, jobs=1)

    # Compute diagnostics
    ek = budget.horizontal_kinetic_energy()
    ea = budget.available_potential_energy()
    ew = budget.vertical_kinetic_energy()

    # ----------------------------------------------------------------------------------------------
    # Nonlinear transfer of Kinetic energy and Available potential energy
    # ----------------------------------------------------------------------------------------------
    # - Nonlinear energy fluxes
    # - linear spectral transfer due to coriolis
    # - Energy conversion from APE to KE
    # - Vertical energy fluxes
    fluxes = budget.cumulative_energy_fluxes()

    pi_r = budget._add_metadata(cumulative_flux(budget.rke_nonlinear_transfer()),
                                'pi_rke', gridtype='spectral',
                                units='W m**-2', standard_name='nonlinear_rke_flux',
                                long_name='cumulative spectral flux of rotational kinetic energy')

    pi_d = budget._add_metadata(cumulative_flux(budget.dke_nonlinear_transfer()),
                                'pi_dke', gridtype='spectral',
                                units='W m**-2', standard_name='nonlinear_dke_flux',
                                long_name='cumulative spectral flux of divergent kinetic energy')

    # ----------------------------------------------------------------------------------------------
    # Combine results into a dataset and export to netcdf
    # ----------------------------------------------------------------------------------------------
    dataset = xr.merge([ek, ea, ew, pi_r, pi_d] + list(fluxes), compat="no_conflicts")
    dataset.attrs.clear()  # clear global attributes
    dataset.attrs.update(dataset_dyn.attrs)
    dataset_dyn.close()

    dataset.to_netcdf("data/energy_budget/{}_energy_fluxes_{}.nc".format(model, suffix))
    dataset.close()

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

            ke_tendency = budget._add_metadata(ke_tendency, "ddt_ke_" + pname,
                                               gridtype='spectral', units='W m**-2')

            ke_tendencies.append(ke_tendency)

    # ----------------------------------------------------------------------------------------------
    # Combine results into a dataset and export to netcdf
    # ----------------------------------------------------------------------------------------------
    dataset = xr.merge(ape_tendencies + ke_tendencies, compat="no_conflicts")
    dataset.attrs.clear()  # clear global attributes
    dataset.attrs.update(dataset_tnd.attrs)
    dataset_tnd.close()

    dataset.to_netcdf(
        DATA_PATH + "/energy_budget/{}_physics_tendencies_{}.nc".format(model, suffix))
    dataset.close()


if __name__ == '__main__':

    for i in tqdm(range(6), desc="Computing spectral energy fluxes"):
        date_id = '20{:d}'.format(i)
        _process_model(model='IFS', resolution='n512', date_time=date_id)
        _process_model(model='ICON', resolution='n512', date_time=date_id)
