import os.path
import warnings

import xarray as xr

from seba import EnergyBudget

warnings.filterwarnings('ignore')

DATA_PATH = 'data/'  # '/mnt/levante/energy_budget/grid_data/'


def reduce_to_1d(func, data, dim="plev", **kwargs):
    res = xr.apply_ufunc(func, data, input_core_dims=[[dim]],
                         kwargs=kwargs, dask='allowed', vectorize=True)
    return res.mean(dim='time')


def _process_model(model, resolution, date_time):
    # Load dyamond dataset
    # # load earth topography and surface pressure
    # sfc_pres = dset_sfc.pres_sfc.values

    date_time_op = date_time.replace("*", "")
    date_time_op = date_time_op.replace("?", "")
    suffix = "_".join([resolution, date_time_op])

    file_names = DATA_PATH + '{}_atm_3d_inst_{}_{}.nc'

    dataset_dyn = xr.open_mfdataset(file_names.format(model, resolution, date_time))
    dataset_tnd = xr.open_mfdataset('data/{}_atm_3d_tend_{}_{}.nc'.format(model, resolution,
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
    budget = EnergyBudget(dataset_dyn, ghsl=sfc_hgt, ps=sfc_pres, filter_terrain=False, jobs=None)

    # Compute diagnostics
    ek = budget.horizontal_kinetic_energy()
    ea = budget.available_potential_energy()
    ew = budget.vertical_kinetic_energy()

    # ----------------------------------------------------------------------------------------------
    # Nonlinear transfer of Kinetic energy and Available potential energy
    # ----------------------------------------------------------------------------------------------
    pik, pia, cka, vfk, vfa = budget.cumulative_energy_fluxes()

    # linear spectral transfer due to Coriolis
    lct = budget.coriolis_linear_transfer()

    # ----------------------------------------------------------------------------------------------
    # Combine results into a dataset and export to netcdf
    # ----------------------------------------------------------------------------------------------
    dataset = xr.merge([ek, ea, ew, pik, pia, cka, vfk, vfa, lct], compat="no_conflicts")
    dataset.attrs.clear()  # clear global attributes
    dataset.attrs.update(dataset_dyn.attrs)
    dataset_dyn.close()

    dataset.to_netcdf("data/energy_budget/{}_energy_fluxes_{}.nc".format(model, suffix))
    dataset.close()

    # ----------------------------------------------------------------------------------------------
    # Compute APE tendency from parameterized processes
    # ----------------------------------------------------------------------------------------------
    ape_tendecies = {}
    tendecies = ['ddt_temp_dyn', 'ddt_temp_radlw', 'ddt_temp_radsw',
                 'ddt_temp_rad', 'ddt_temp_conv', 'ddt_temp_vd',
                 'ddt_temp_gwd', 'ddt_temp_turb', 'ddt_temp_gscp']

    for name in tendecies:

        pname = name.split('_')[-1].lower()

        tend_grid = dataset_tnd.get(name)
        if tend_grid is not None:
            ape_tendecies[pname] = budget.get_ape_tendency(tend_grid, name="ddt_ape_" + pname,
                                                           cumulative=True)

    # ----------------------------------------------------------------------------------------------
    # Combine results into a dataset and export to netcdf
    # ----------------------------------------------------------------------------------------------
    dataset = xr.merge(ape_tendecies.values(), compat="no_conflicts")
    dataset.attrs.clear()  # clear global attributes
    dataset.attrs.update(dataset_tnd.attrs)
    dataset_tnd.close()

    dataset.to_netcdf(
        DATA_PATH + "/energy_budget/{}_physics_tendencies_{}.nc".format(model, suffix))
    dataset.close()


if __name__ == '__main__':
    #
    _process_model(model='IFS', resolution='n256', date_time='20?')
