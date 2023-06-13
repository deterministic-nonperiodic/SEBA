import os
import warnings

import numpy as np
import xarray as xr
from joblib import Parallel, delayed, cpu_count
from pandas import read_table
from tqdm import tqdm
from xarray import set_options

# from io_tools import get_surface_elevation
from spherical_harmonics import Spharmt

warnings.filterwarnings('ignore')
set_options(keep_attrs=True)

print("-------------------------- Memory usage ---------------------------------------------------")
print('Total: {} -- Used: {} -- Free: {}'.format(*os.popen('free -th').readlines()[-1].split()[1:]))
print("-------------------------------------------------------------------------------------------")

data_path = '/media/yanm/Data/DYAMOND/modes'
output_path = '/media/yanm/Data/DYAMOND/modes/WINTER/'

# 6-hourly data from 2020-01-25 to 2020-01-30
date_register = {
    'nwp2.5': [39, 41, 43, 46, 48, 50, 53, 55, 57, 60, 62, 64, 67, 69, 71],
    'ifs4': range(21, 42),
    'era5': range(96, 192, 2)
}

model_alias = {
    'nwp2.5': "ICON",
    'ifs4': "IFS",
    'geos3': "GEOS",
    'era5': "ERA5",
}

# model levels index corresponding to IFS L137 vertical grid
level_mask = np.array([29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 56, 58, 60, 61, 63,
                       64, 66, 67, 68, 70, 71, 72, 74, 75, 76, 78, 79, 80, 82, 83, 84, 86, 87, 89,
                       90, 91, 93, 94, 95, 97, 98, 99, 101, 102, 103, 105, 106, 107, 109, 110,
                       112, 113, 115, 116, 118, 120, 122, 124, 126, 128, 131, 133, 135, 137]) - 1
start_level = level_mask[0]

# surface geopotential from external file (constant in time for all analysis).
phi_s = xr.open_dataset(os.path.join(data_path, 'topography.nc')).z.values
phi_s = np.expand_dims(phi_s, 1)

# Load IFS coordinate parameters:
reference_coordinates = read_table(
    # '/work/mh1119/m300794/dyamond_dataset/reduced_grid/grids/ifs_vertical_levels_rd.dat',
    os.path.join(data_path, 'ifs_vertical_levels_rd.dat'),
    names=['level_id', 'a', 'b', 'ph', 'pf', 'geop', 'z', 't', 'rho'], index_col=0,
    sep=', ', engine='python')

# Integration indices:
ai = np.insert(reference_coordinates['a'].values, 0, 0.0)
bi = np.insert(reference_coordinates['b'].values, 0, 0.0)

am = 0.5 * (ai[1:] + ai[:-1])  # A coefficients at full levels
bm = 0.5 * (bi[1:] + bi[:-1])  # B coefficients at full levels

delta_a = np.diff(ai)[start_level:]
delta_b = np.diff(bi)[start_level:]


def sigma_hybrid_coordinate(a, b):
    return b + a / 101325.


def geopotential(temperature, ps, dim='level'):
    # compute geopotential from temperature
    rd = 287.058  # gas constant for dry air (J / kg / K)

    temperature = temperature.sortby(dim, ascending=True)
    level = temperature[dim]

    # Averaged temperature at full levels excluding the surface
    full_levels = sigma_hybrid_coordinate(ai[start_level:-1], bi[start_level:-1])

    temperature = temperature.interp({dim: full_levels}, method='linear',
                                     kwargs={"fill_value": "extrapolate"})

    # Compute pressure at mid levels
    half_levels = sigma_hybrid_coordinate(am[start_level - 1:], bm[start_level - 1:])

    # hybrid sigma/pressure coefficients at half levels
    a = xr.DataArray(am[start_level - 1:], coords={dim: half_levels})
    b = xr.DataArray(bm[start_level - 1:], coords={dim: half_levels})

    # pressure variation in log space (assign mid-level coordinates)
    delta_lnp = np.log(a + b * ps).diff(dim).assign_coords({dim: full_levels})

    # compute depth of each geopotential layer
    depth = rd * (delta_lnp * temperature).sortby(dim, ascending=False)

    # geopotential each level from the cumulative integral from the surface
    phi = phi_s + depth.cumsum(dim=dim, skipna=False)

    # reassign original coordinate and add attributes
    phi = phi.sortby(dim, ascending=True).assign_coords({dim: level})
    phi.attrs.update(dict(standard_name='geopotential',
                          long_name='Geopotential',
                          units='m**2 s**-2'))

    return phi.transpose(..., 'lev', 'lat', 'lon')


def parse_model_data(model_name, component='IG', date=None):
    """
    This function combines the input dataset to MODES and the resulting dataset after
    the modal decomposition. The resulting dataset contains the inverse wind components
    along with the thermodynamic fields.
    """
    mode = component.replace('RO', 'ROT')
    date = date or 0

    base_path = os.path.join(data_path, f'{model_name}')

    file_name = '{}_all_2020012000000{:02d}.nc'.format(mode, date)
    file_name = os.path.join(base_path, 'inverse', file_name)
    dataset_mode = xr.open_mfdataset(file_name).drop_vars('Z')

    # open dataset and extract coordinates
    file_name = os.path.join(base_path, 'infile_2020012000000{:02d}.nc'.format(date))
    dataset_input = xr.open_mfdataset(file_name)

    # use sigma-pressure levels
    dataset_mode['lev'].attrs.update(
        dict(standard_name="sigma",
             long_name="sigma hybrid level at layer midpoints",
             positive="down", axis="Z"))

    levels = dataset_mode.lev

    dataset_input = dataset_input.assign_coords(lev=levels)

    # use the time coordinate from input dataset (MODES messed it up)
    dataset_mode = dataset_mode.assign_coords(time=dataset_input.time)

    # collect thermodynamic variables from input files
    temperature = dataset_input.t.values
    dataset_mode['temperature'] = xr.DataArray(temperature, dims=dataset_input.t.dims)
    dataset_mode['temperature'].attrs['units'] = 'K'

    # add surface pressure (remove dummy level first):
    if 'lev_2' in dataset_input.lnsp.dims:
        lnsp = dataset_input.lnsp.isel(lev_2=0).values
    else:
        lnsp = dataset_input['lnsp'].values

    dataset_mode['ps'] = xr.DataArray(np.exp(lnsp), dims=('time', 'lat', 'lon'))
    dataset_mode['ps'].attrs['units'] = 'Pa'

    # compute surface temperature by linear extrapolation of temperature to sigma=1.0
    dataset_mode['ts'] = dataset_mode.temperature.interp(lev=1.0, method='linear',
                                                         kwargs={"fill_value": "extrapolate"})

    # ----------------------------------------------------------------------------------------------
    # Compute full pressure field at mid-levels: using 68-level resolution [Pa]
    # ----------------------------------------------------------------------------------------------
    # parameters of vertical coordinate transformation
    a = xr.DataArray(am[level_mask], coords={'lev': levels})
    b = xr.DataArray(bm[level_mask], coords={'lev': levels})

    dataset_mode['pressure'] = (a + b * dataset_mode['ps']).transpose(..., 'lev', 'lat', 'lon')
    dataset_mode['pressure'].attrs['units'] = 'Pa'

    return dataset_mode


def compute_omega(ps, lnsp_adv, div, correct=0):
    """
        Compute Omega using IFS vertical discretization Eq (2.18)
        documentation (https://www.ecmwf.int/node/20197)
    """

    # Correction term for model top pressure > 0.0
    mass_deficit = correct * 608.23356  # Pa #- np.nansum(delta_a) / 2.0

    # pressure layer difference
    scale_p = delta_b * ps[:, np.newaxis]
    delta_p = delta_a + scale_p

    # Compute advection of total pressure: nabla(v dp/dn)
    mass_flux = scale_p * lnsp_adv + delta_p * div

    # Vertical cumulative integral of the total mass flux
    integral_flux = np.cumsum(mass_flux, axis=1)

    # Negative surface pressure tendency: d(ps)/dt = ps d(lnsp)/dt
    dps_dt = integral_flux[:, -1:] - mass_deficit * div[:, 0]

    # Compute omega at mid levels:
    return bm[start_level:] * dps_dt + 0.5 * mass_flux - integral_flux


def compute_omega_mp(dataset, axis=-1, nprocs=12):
    # organize arrays dimensions:
    data_shape = list(dataset['u'].shape)
    nlevels = data_shape.pop(axis)

    # Unpack and reshape 2d fields
    vars_nd = {'ps': dataset.ps.values.reshape(-1), 'div': dataset['div'].values}

    # Compute horizontal advection of total pressure: nabla(v dp/dn)
    lnsp = np.log(dataset.ps.values)

    vars_nd['lnsp_adv'] = scalar_advection(dataset['u'].values, dataset['v'].values,
                                           vars_nd['div'], lnsp, spatial_axes=(2, 3))

    # Unpack and reshape 4d fields
    for varname in ['div', 'lnsp_adv']:
        vars_nd[varname] = np.moveaxis(vars_nd[varname], axis, -1).reshape(-1, nlevels)

    # Create chunks of arrays along first axis for the mp mapping ...
    varnames = ['ps', 'lnsp_adv', 'div']

    n_chunks = vars_nd['ps'].size
    packed_arrays = [np.array_split(vars_nd[vname], n_chunks) for vname in varnames]

    # wrap function passing static arguments
    pool = Parallel(n_jobs=nprocs, backend="threading")

    # applying lanczos filter in parallel
    result = pool(delayed(compute_omega)(*args) for args in zip(*packed_arrays))

    # concatenate individual procs results:
    result = np.concatenate(result, axis=0)

    # back to original array shape:
    return np.moveaxis(result.reshape(data_shape + [nlevels, ]), -1, axis)


def compute_vrtdiv(u, v, gridtype='gaussian', spatial_axes=None):
    """
        Computes vorticity and divergence of the horizontal wind

    *truncation*
        Truncation limit (triangular truncation) for the spherical
        harmonic computation.

    **Returns:**
    *uchi*, *vchi*, *upsi*, *vpsi*


    **Examples:**

    Compute the irrotational and non-divergent components of the
    vector wind::
        uchi, vchi, upsi, vpsi = w.helmholtz()
    """
    assert u.shape == v.shape, 'Components u and v must be the same shape'

    if spatial_axes is None:
        spatial_axes = (-2, -1)
    else:
        assert len(spatial_axes) == 2, ValueError('spatial_axes must be a 2-tuple')

    u = np.moveaxis(u, spatial_axes, (0, 1))
    v = np.moveaxis(v, spatial_axes, (0, 1))

    # get dimensions
    nlat, nlon, *extra_dims = u.shape

    # Create a Spharmt object
    sphere = Spharmt(nlat, nlon, gridtype=gridtype)

    # compute spectral coefficients
    u = u.reshape((nlat, nlon, -1))
    v = v.reshape((nlat, nlon, -1))

    vrt_spec, div_spec = sphere.getvrtdivspec(u, v)

    # convert to physical space
    div_grid = sphere.synthesis(div_spec)
    vrt_grid = sphere.synthesis(vrt_spec)

    div_grid = np.moveaxis(div_grid.reshape([nlat, nlon] + extra_dims), (0, 1), spatial_axes)
    vrt_grid = np.moveaxis(vrt_grid.reshape([nlat, nlon] + extra_dims), (0, 1), spatial_axes)

    return div_grid, vrt_grid


def scalar_advection(u, v, div, scalar, spatial_axes=None):
    r"""
    Compute the horizontal advection of a scalar field on the sphere.

    Advection is computed in flux form for better conservation properties.
    This approach also has higher performance than computing horizontal gradients.

    .. math:: \mathbf{u}\cdot\nabla_h\phi = \nabla_h\cdot(\phi\mathbf{u}) - \delta\phi

    where :math:`\phi` is an arbitrary scalar, :math:`\mathbf{u}=(u, v)` is the horizontal
    wind vector, and :math:`\delta` is the horizontal wind divergence.

    Parameters:
    -----------
        scalar: `np.ndarray`
            scalar field to be advected
    Returns:
    --------
        advection: `np.ndarray`
            Array containing the advection of a scalar field.
    """
    # Same as but faster than: np.ma.sum(self.wind * self.horizontal_gradient(scalar), axis=0)

    # Spectral coefficients of the scalar flux divergence: ∇⋅(φ u)
    scalar_flux_divergence = compute_vrtdiv(u * scalar, v * scalar, spatial_axes=spatial_axes)[0]

    # recover scalar advection: u⋅∇φ = ∇⋅(φu) - δφ
    return scalar_flux_divergence - div * scalar


def compute_modes_fluctuations(model_name=None, mode='IG'):
    var_dims = ('time', 'lev', 'lat', 'lon')

    # compute target vertical coordinate fro interpolation
    target_levels = sigma_hybrid_coordinate(am[start_level:], bm[start_level:])

    # ----------------------------------------------------------------------
    # Create datasets for each time step and export to netcdf file        :
    # ----------------------------------------------------------------------
    description = 'Processing model ' + model_alias[model_name].upper()

    for time_id in tqdm(date_register[model_name], desc=description):
        # Load datasets:
        dataset = parse_model_data(model_name=model_name, component=mode, date=time_id)
        date = str(dataset.time[0].dt.strftime("%Y%m%dT%H").values)

        # store original levels
        levels = dataset.lev.values
        # ------------------------------------------------------------------------------------------
        # Compute divergence at mid-levels: using 68-level resolution
        # ------------------------------------------------------------------------------------------
        div, vrt = compute_vrtdiv(dataset.u.values, dataset.v.values, spatial_axes=(2, 3))

        dataset['div'] = xr.DataArray(div, dims=var_dims)
        dataset['vrt'] = xr.DataArray(vrt, dims=var_dims)

        # perform vertical interpolation to integrate continuous levels
        dataset = dataset.interp(lev=target_levels, method='cubic',
                                 kwargs={"fill_value": "extrapolate"})

        # ------------------------------------------------------------------------------------------
        # Calculate pressure vertical velocity [Ps/s] using IFS discretization Eqs. (2.18-3.17)
        # ------------------------------------------------------------------------------------------
        omega = compute_omega_mp(dataset, axis=1, nprocs=cpu_count())

        dataset['omega'] = xr.DataArray(omega, dims=var_dims)
        dataset['omega'].attrs.update(dict(standard_name='pressure velocity', units='Pa/s'))

        # ------------------------------------------------------------------------------------------
        # Calculate geopotential depth [m**2/s**2] using IFS discretization Eqs. (2.21)
        # ------------------------------------------------------------------------------------------
        dataset['geopotential'] = geopotential(dataset.temperature, dataset.ps, dim='lev')

        # --------------------------------------------------------------------------------------
        # Export dataset to netcdf file
        # --------------------------------------------------------------------------------------
        # recovering the original 68 sigma levels
        dataset = dataset.sel(lev=levels, method='nearest')

        file_name = '_'.join([model_alias[model_name], mode, 'inst', 'n256', date]) + '_test.nc'
        file_name = os.path.join(output_path, file_name)

        dataset.astype('float32').to_netcdf(file_name)
        dataset.close()


if __name__ == '__main__':
    compute_modes_fluctuations(model_name='ifs4', mode='IG')

    # combined = [(model, mode) for model in ['nwp2.5', 'ifs4'] for mode in ['IG', 'RO']]
    #
    # for model, mode_id in combined:
    #     compute_modes_fluctuations(model_name=model, mode=mode_id)
    #     print('-----------------------------------------------------------------------------')
