import os
import re
from typing import Optional

import numpy as np
import pint
from xarray import apply_ufunc, Dataset, DataArray, open_dataset

from thermodynamics import pressure_vertical_velocity
from tools import interpolate_1d, inspect_gridtype

path_global_topo = "../data/topo_global_n1250m.nc"

CF_variable_conventions = {
    "temperature": {
        "standard_name": ("t", "ta", "temp", "air_temperature", 'temperature'),
        "units": ('K', 'kelvin', 'Kelvin', 'degree_C')
    },
    "pressure": {
        "standard_name": ("p", "air_pressure", "pressure"),
        "units": ('Pa', 'hPa', 'mb', 'millibar')
    },
    "omega": {
        "standard_name": ('omega', 'lagrangian_tendency_of_air_pressure',
                          'pressure_velocity', 'pressure velocity',
                          'vertical_velocity', 'vertical velocity'),
        "units": ("Pa s-1", 'Pa s**-1', 'Pa/s')
    },
    "u_wind": {
        "standard_name": ('u', 'ua', 'zonal_wind', 'zonal wind',
                          "eastward_wind", 'zonal wind component'),
        "units": ("m s-1", "m s**-1", "m/s")
    },
    "v_wind": {
        "standard_name": ('v', 'va', 'meridional_wind', 'Meridional wind',
                          'northward_wind', 'meridional wind component'),
        "units": ("m s-1", "m s**-1", "m/s")
    },
    'w_wind': {
        'standard_name': ('w', 'wa', 'vertical_velocity', 'vertical velocity',
                          "upward_air_velocity", "vertical wind component"),
        'units': ("m s-1", "m s**-1", "m/s")
    },
    'ts': {
        "standard_name": ('ts', 'surface_temperature', 'surface_air_temperature'),
        "units": ('K', 'kelvin', 'Kelvin', 'degree_C')
    },
    "ps": {
        "standard_name": ('ps', 'surface_pressure', 'surface pressure', 'surface_air_pressure'),
        "units": ('Pa', 'hPa', 'mb', 'millibar'),
    }
}

CF_coordinate_conventions = {
    "latitude": {
        "standard_name": ("latitude", "lat", "lats"),
        "units": ("degree_north", "degree_N", "degreeN",
                  "degrees_north", "degrees_N", "degreesN"),
        'axis': ('Y',)
    },
    "longitude": {
        "standard_name": ("longitude", "lon", "lons"),
        "units": ("degree_east", "degree_E", "degreeE",
                  "degrees_east", "degrees_E", "degreesE"),
        'axis': ('X',)
    },
    "level": {
        "standard_name": ('level', 'lev', 'pressure', 'air_pressure',
                          'altitude', 'depth', 'height', 'geopotential_height',
                          'height_above_geopotential_datum',
                          'height_above_mean_sea_level',
                          'height_above_reference_ellipsoid'),
        "units": ('meter', 'm', 'gpm', 'Pa', 'hPa', 'mb', 'millibar'),
        "axis": ('Z', 'vertical')
    }
}

expected_units = {
    "u_wind": "m s**-1",
    "v_wind": "m s**-1",
    "w_wind": "m s**-1",
    "temperature": "K",
    "pressure": "Pa",
    "omega": "Pa s**-1"
}

expected_range = {
    "u_wind": [-350., 350.],
    "v_wind": [-350., 350.],
    "w_wind": [-100., 100.],
    "temperature": [120, 350.],
    "pressure": [0.0, 2000e2],
    "omega": [-100, 100]
}

cmd = re.compile(r'(?<=[A-Za-z)])(?![A-Za-z)])(?<![0-9\-][eE])(?<![0-9\-])(?=[0-9\-])')


def _parse_power_units(unit_str):
    return cmd.sub('**', unit_str)


def map_func(func, data, dim="plev", **kwargs):
    # map function to all variables in dataset along axis
    res = apply_ufunc(func, data, input_core_dims=[[dim]],
                      kwargs=kwargs, dask='allowed',
                      vectorize=True)

    if 'pressure_range' in kwargs.keys() and isinstance(data, Dataset):
        res = res.assign_coords({'layer': kwargs['pressure_range']})

    return res


def check_and_convert_units(ds):
    # Define the expected units for each variable

    # Create a pint UnitRegistry object
    reg = pint.UnitRegistry()

    # Check and convert the units of each variable
    for varname in ds.variables:
        var = ds[varname]
        if varname not in expected_units:
            continue

        expected_unit = expected_units[varname]

        var_units = var.attrs.get("units")
        if var_units is None:
            print(f"Warning: Units not found for variable {varname}")

            if np.nanmax(var) <= expected_range[varname][1] and \
                    np.nanmin(var) >= expected_range[varname][0]:

                var_units = expected_unit
            else:
                raise ValueError(f"Variable '{varname}' has no units and values are outside "
                                 f"admitted range: {expected_range[varname]}")

        if var_units != expected_unit:
            try:
                # Convert the values to the expected units
                fixed_units = reg(_parse_power_units(var_units))
                var.values = (var.values * fixed_units).to(expected_unit).magnitude

            except pint.errors.DimensionalityError:
                raise ValueError(f"Cannot convert {varname} units to {expected_unit}.")

        # Update the units attribute of the variable
        var.attrs["units"] = var_units


def get_coordinate_names(dataset):
    """
    Given a Dataset object, returns a list of coordinate names in the same order
    as they appear in the data variables.

    Parameters:
        dataset (xr.Dataset): The xarray Dataset object to extract the coordinate names from.

    Returns:
        list: A list of coordinate names in the same order as they appear in the data variables.
    """
    # Get the list of data variable names
    var_names = list(dataset.data_vars.keys())

    # Create an empty list to store the coordinate names
    coord_names = []

    # Iterate over each data variable
    for var_name in var_names:
        # Get the variable object
        var = dataset[var_name]
        # Iterate over each dimension of the variable
        for dim in var.dims:
            # If the dimension is not already in the coordinate names list, add it
            if dim not in coord_names:
                coord_names.append(dim)

    return coord_names


def ordered_dims(dataset):
    # returns dimensions in dataset ordered according to how they appear in data
    return [dataset.dims[name] for name in get_coordinate_names(dataset)]


def _find_coordinate(ds, name):
    """
    Find a dimension coordinate in an `xarray.DataArray` that satisfies
    a predicate function.
    """
    if name not in CF_coordinate_conventions.keys():
        raise ValueError("Wrong coordinate name: {!s}".format(name))

    criteria = CF_coordinate_conventions[name]

    predicate = lambda c: (c.name in criteria['standard_name'] or
                           c.attrs.get('units') in criteria['units'] or
                           c.attrs.get('axis') in criteria['axis'])

    candidates = [coord for coord in [ds.coords[n] for n in ds.dims] if predicate(coord)]

    if not candidates:
        raise ValueError('Cannot find a {!s} coordinate'.format(name))
    if len(candidates) > 1:
        msg = 'Multiple {!s} coordinates are not allowed.'
        raise ValueError(msg.format(name))
    coord = candidates[0]
    dim = ds.dims[coord.name]
    return coord, dim


def reindex_coordinate(coord, data):
    # Reindex dataset coordinate
    return coord.reindex({coord.name: data})


def is_standard_variable(data, standard_name):
    var_attrs = data.attrs

    cf_attrs = CF_variable_conventions[standard_name]

    has_name = np.any([var_attrs.get(name) in cf_attrs["standard_name"]
                       for name in ["name", "standard_name", "long_name"]]) or \
               data.name in cf_attrs["standard_name"]

    units = var_attrs.get("units")

    if units is not None:
        return has_name and (units in cf_attrs["units"])
    else:
        return has_name


def _find_variable(dataset, variable_name, raise_notfound=True) -> Optional[DataArray]:
    """
    Given a xarray dataset and a variable name, identifies whether the variable
    is present in the dataset and conforms to CF conventions.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset to identify the variable in.
    variable_name : str, dict
        The name of the variable to identify.
    raise_notfound: bool, optional, default True
        If raise_notfound=true, raises error if variable is not found, or returns None otherwise.
    Returns
    -------
    DataArray or None,
        A DataArray if variable is found in the dataset or None if not found and
        raise_notfound=False. Raises ValueError otherwise.
    """
    if isinstance(variable_name, dict):
        (standard_name, variable_name), = variable_name.items()
    elif isinstance(variable_name, (list, tuple)):
        standard_name, variable_name = variable_name
    elif isinstance(variable_name, str):
        standard_name = variable_name
    else:
        raise ValueError("Wrong value for variable name: {}".format(variable_name))

    # try to get variable by literal name
    array = dataset.get(variable_name)

    # Array was found by explicit name (no checks needed)
    if array is not None:
        return array

    candidates = []

    for var_name in dataset.variables:
        # Check if the variable is in the dataset and consistent with the CF conventions
        if is_standard_variable(dataset[var_name], standard_name):
            candidates.append(dataset[var_name])

    if not candidates:
        if raise_notfound:
            # variable is not in the dataset or not consistent with the CF conventions
            msg = "The variable {} is not in the dataset or is " \
                  "inconsistent with the CF conventions."
            raise ValueError(msg.format(variable_name))
        else:
            return None

    return candidates[0]


def parse_dataset(dataset, variables=None):
    """
        Parse input xarray dataset

        Returns
        _______
        arrays: a list of requested DataArray objects

    """
    default_variables = ['u_wind', 'v_wind', 'omega', 'temperature', 'pressure']

    if variables is None:
        variables = {name: name for name in default_variables}

    if isinstance(variables, dict):
        merged_variables = {}
        # Loop through each variable in the default list
        for var_name in default_variables:
            # If the variable exists in the input dict, use its value
            if var_name in variables:
                merged_variables[var_name] = variables[var_name]
            # Otherwise, use the default value
            else:
                merged_variables[var_name] = var_name
        variables = merged_variables
    else:
        raise ValueError("Unknown type for 'variables', must be a dictionary")

    # find variables by name or candidates variables based on CF conventions
    arrays = {name_map[0]: _find_variable(dataset, name_map, raise_notfound=True)
              for name_map in variables.items()}

    if len(arrays) != len(variables):
        raise ValueError("Missing variables!")

    # check sanity of 3D fields
    for name, values in arrays.items():

        if np.isnan(values).any():
            raise ValueError('Array {} contain missing values'.format(name))

        # Make sure the shapes of the two components match.
        if (name != 'pressure') and values.ndim < 3:
            raise ValueError('Fields must be at least 3D.'
                             'Variable {} has {} dimensions'.format(name, values.ndim))

    # find surface data arrays
    for name in ['ts', 'ps']:

        # silence error for flexible variables.
        sfc_var = _find_variable(dataset, name, raise_notfound=False)

        if sfc_var is None:
            print("Warning: surface variable {} not found!".format(name))
            continue
        # Surface data must be exactly one dimension less than the total number of dimensions
        # in dataset (Not sure if this is a good enough condition!?)
        if np.ndim(sfc_var) == len(dataset.dims) - 1:
            arrays[name] = sfc_var

    # Check units and convert to standard if needed before computing vertical velocity
    check_and_convert_units(dataset)

    # Check for pressure velocity in dataset. If not present, it is estimated from
    # height based vertical velocity. If neither is found raises ValueError.
    if arrays.get('omega') is None:
        pressure = arrays.get('pressure').values
        temperature = arrays.get('temperature').values
        w_wind = _find_variable(dataset, 'w_wind', raise_notfound=True).values

        arrays['omega'] = pressure_vertical_velocity(pressure, w_wind, temperature)

    # create dataset and fill nans with zeros for spectral computations
    return Dataset(arrays, coords=dataset.coords).fillna(0.0)


def interpolate_pressure_levels(dataset, p_levels=None):
    """
    Interpolate all variables in dataset to pressure levels.

    Parameters
    ----------
    dataset : xarray.Dataset
        The input dataset to be interpolated.

    p_levels : list, array, optional
        The target levels to interpolate to. If not given, it is created from the number of
        vertical levels of the variables in dataset and the pressure field.

    Returns
    -------
    xarray.Dataset
        The interpolated dataset.

    Raises
    ------
    ValueError
        If the input dataset is not on pressure levels and the pressure field is missing.

    Notes
    -----
    This function checks if the input dataset is on pressure levels by examining its coordinate
    variables. If the pressure coordinate variable is not present, or if it is not consistent
    with CF conventions (standard name, units), this function assumes that the dataset is not on
    pressure levels and attempts to interpolate all variables to pressure levels using linear
    interpolation. If the pressure field is not present, or it is consistent with CF convention
    raises a `ValueError`.
    """

    # find vertical coordinate
    levels, nlevels = _find_coordinate(dataset, "level")

    # Finding pressure array in dataset. Silence error if not found (try to use coordinate)
    pressure = _find_variable(dataset, 'pressure', raise_notfound=False)

    # If the vertical coordinate is consistent with pressure standards, then no interpolation is
    # needed: Data is already in pressure coordinates.
    if is_standard_variable(levels, "pressure"):
        # If pressure was not found create a new variable 'pressure' from levels.
        if pressure is None:
            dataset["pressure"] = levels

        return dataset

    # Prepare data for interpolation to constant pressure levels:
    if pressure is None:
        raise ValueError("Pressure not found in dataset. If the data is not in pressure "
                         "levels, a 3D pressure field must be provided!")

    data_shape = tuple(ordered_dims(dataset))

    # check pressure dimensions
    if np.shape(pressure) != data_shape:
        raise ValueError("Pressure field must match the dimensionality of the other "
                         "dynamic fields for the interpolation. Expecting {}, "
                         "but got {} instead.".format(data_shape, np.shape(pressure)))

    if p_levels is None:
        # creating levels from pressure range and number of levels if no pressure levels
        # are given (set bottom level to 1000 hPa)
        p_levels = np.linspace(1000e2, np.min(pressure), nlevels)  # [Pa]
    else:
        p_levels = np.array(p_levels)
        assert p_levels.ndim == 1, "'p_levels' must be a one-dimensional array."

    print("Interpolating data to {} isobaric levels ...".format(p_levels.size))

    dims = get_coordinate_names(dataset)
    coords = [dataset.coords[name] for name in dims]
    z_axis = ''.join([coord.axis for coord in coords]).lower().find('z')

    # Replace vertical coordinate with new pressure levels
    dims[z_axis] = 'plev'
    coords[z_axis] = DataArray(data=p_levels, name='plev', dims=["plev"],
                               coords=dict(plev=("plev", p_levels)),
                               attrs=dict(standard_name="pressure",
                                          long_name="pressure", positive="up",
                                          units="Pa", axis="Z"))
    # create coordinate dictionary for dataset
    coords = {coord.name: coord for coord in coords}

    # Start vertical interpolation of required fields in dataset.
    excluded_vars = []
    result_dataset = {}
    for name, data in dataset.data_vars.items():
        # exclude pressure and all variables with the wrong dimensionality
        if not data.equals(pressure) and data.ndim == pressure.ndim:
            result, = interpolate_1d(p_levels, pressure.values,
                                     data.values, scale='log',
                                     fill_value=np.nan, axis=z_axis)

            result_dataset[name] = (dims, result, data.attrs)
        else:
            excluded_vars.append(name)

    print("Interpolation successfully completed.")

    # replace pressure field in dataset with the new coordinate
    attrs = {'standard_name': "pressure", 'units': "Pa"}
    result_dataset['pressure'] = ('plev', p_levels, attrs)

    # Add fields in dataset that were not interpolated except 'pressure'.
    for svar in excluded_vars:
        sdata = dataset[svar]
        if not sdata.equals(pressure):
            result_dataset[svar] = (sdata.dims, sdata.values, sdata.attrs)

    # Create new dataset with interpolated data
    return Dataset(data_vars=result_dataset, coords=coords)


def get_surface_elevation(latitude, longitude):
    """
        Interpolates global topography data to desired grid defined by arrays lats and lons.
        The data source is SRTM15Plus (Global SRTM15+ V2.1) from https://opentopography.org/
    """

    # convert longitudes to range (-180, 180) if needed
    longitude = np.where(longitude > 180, longitude - 360, longitude)

    # infer grid type
    grid_type, _, _ = inspect_gridtype(latitude)
    grid_prefix = "n" if grid_type == 'gaussian' else "r"

    grid_id = grid_prefix + str(latitude.size // 2)

    file_name = f"topo_global_{grid_id}.nc"

    expected_path, _ = os.path.split(path_global_topo)
    expected_path = os.path.join(expected_path, file_name)

    if os.path.exists(expected_path):
        # opening existing dataset
        ds = open_dataset(expected_path)

        # sorting according to required longitudes. Latitudes are sorted (north-to-south)
        if np.allclose(longitude, np.sort(longitude)):
            ds = ds.sortby('lon').sortby('lat', ascending=False)

        if np.allclose(ds.lon.values, longitude):
            # return DataArray with surface elevation
            return ds.elevation
        else:
            print("Warning: could not determine the ordering of coordinate 'longitude'.")
            remap = True
    else:
        print(f"Warning: Surface elevation data with required resolution "
              f"not found! Interpolating global data to grid {grid_id}.")
        remap = True

    if remap:
        # Dataset with required resolution does not exist! We need to interpolate...
        # This step could take a while since it is a large dataset containing global
        # topography data with ~1.25 km resolution.
        ds = open_dataset(path_global_topo)

        ds = ds.interp(lat=latitude, lon=longitude,
                       method="nearest", kwargs={"fill_value": 0.0})

        # export interpolated dataset to netcdf for future use...
        ds.to_netcdf(expected_path)

        # return DataArray with surface elevation
        return ds.elevation
