import os
import re
from typing import Optional

import numpy as np
import pint
from xarray import Dataset, DataArray, open_dataset, open_mfdataset

import constants as cn
from fortran_libs import numeric_tools
from thermodynamics import pressure_vertical_velocity
from tools import interpolate_1d, inspect_gridtype, prepare_data, surface_mask, is_sorted

path_global_topo = "./data/topo_global_n1250m.nc"

CF_variable_conventions = {
    "geopotential": {
        "standard_name": ("geop", "geopotential", "air_geopotential",),
        "units": ('joule / kilogram',)
    },
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
    "time": {
        "standard_name": ("time", "date", "t"),
        "units": (),
        "calendar": ('proleptic_gregorian', 'julian', 'gregorian', 'standard'),
        'axis': ('T',)
    },
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
        "standard_name": ('level', 'lev', 'zlev', 'eta', 'plev',
                          'pressure', 'air_pressure', 'altitude',
                          'depth', 'height', 'geopotential_height',
                          'height_above_geopotential_datum',
                          'height_above_mean_sea_level',
                          'height_above_reference_ellipsoid',
                          'atmosphere_hybrid_height_coordinate',
                          'atmosphere_hybrid_height_coordinate',
                          'atmosphere_ln_pressure_coordinate',
                          'atmosphere_sigma_coordinate',
                          'atmosphere_sleve_coordinate',
                          ''),
        "units": ('meter', 'm', 'gpm', 'Pa', 'hPa', 'mb', 'millibar'),
        "axis": ('Z', 'vertical')
    }
}

expected_units = {
    "u_wind": "m/s",
    "v_wind": "m/s",
    "w_wind": "m/s",
    "temperature": "K",
    "pressure": "Pa",
    "omega": "Pa/s",
    "geopotential": "m**2 s**-2",
}

expected_range = {
    "u_wind": [-350., 350.],
    "v_wind": [-350., 350.],
    "w_wind": [-100., 100.],
    "temperature": [120, 350.],
    "pressure": [0.0, 2000e2],
    "omega": [-100, 100],
    "geopotential": [0., ],
}

# from Metpy
cmd = re.compile(r'(?<=[A-Za-z)])(?![A-Za-z)])(?<![0-9\-][eE])(?<![0-9\-])(?=[0-9\-])')

# Create a pint UnitRegistry object
reg = pint.UnitRegistry()


def _parse_power_units(unit_str):
    return cmd.sub('**', unit_str)


def _parse_units():
    return


def equivalent_units(unit_1, unit_2):
    unit_1 = reg.Unit(_parse_power_units(unit_1))
    unit_2 = reg.Unit(_parse_power_units(unit_2))

    ratio = (1 * unit_1 / (1 * unit_2)).to_base_units()

    return ratio.dimensionless and np.isclose(ratio.magnitude, 1.0)


def compatible_units(unit_1, unit_2):
    unit_1 = reg.Unit(_parse_power_units(unit_1))
    unit_2 = reg.Unit(_parse_power_units(unit_2))

    return unit_1.is_compatible_with(unit_2)


class SebaDataset(Dataset):
    __slots__ = ()

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        return instance

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def find_coordinate(self, coord_name=None):
        # Use the instance variables to perform the necessary computations
        return _find_coordinate(self, coord_name)

    def find_variable(self, name=None, raise_notfound=True):
        return _find_variable(self, name, raise_notfound=raise_notfound)

    def check_convert_units(self, other=None):

        is_array = False
        if other is None:
            ds = self.copy()
        elif isinstance(other, Dataset):
            ds = other.copy()
        elif isinstance(other, DataArray):
            is_array = True
            ds = other.to_dataset()
        else:
            raise ValueError("Illegal type for parameter 'other'")

        # Check and convert the units of each variable
        for varname in ds.data_vars:
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
                if compatible_units(var_units, expected_unit):
                    # Convert the values to the expected units
                    fixed_units = reg(_parse_power_units(var_units))
                    var.values = (var.values * fixed_units).to(
                        _parse_power_units(expected_unit)).magnitude

                else:
                    raise ValueError(f"Cannot convert {varname} units"
                                     f"{var_units} to {expected_unit}.")

            # Update the units attribute of the variable
            var.attrs["units"] = var_units

        if is_array:
            # get rid of extra coordinates and return original DataArray object
            ds = ds.to_array().squeeze('variable').drop_vars('variable')

        return ds

    def coordinate_names(self):
        return get_coordinate_names(self)

    def coordinates_by_axes(self):
        cds = self.coords
        return {cds[name].axis.lower(): cds[name] for name in self.coordinate_names()}

    def data_shape(self):
        return tuple(self.dims[name] for name in self.coordinate_names())

    def interpolate_levels(self, p_levels=None):
        return interpolate_pressure_levels(self, p_levels=p_levels)

    def get_field(self, name):
        """ Returns a field in dataset after preprocessing:
            - Exclude extrapolated data below the surface (p >= ps).
            - Ensure latitude axis is oriented north-to-south.
            - Reverse vertical axis from the surface to the model top.
        """

        if name not in self.data_vars:
            return None

        # Calculate mask to exclude extrapolated data below the surface (p >= ps).
        # beta should be computed just once
        if hasattr(self, 'pressure') and hasattr(self, 'ps'):
            mask = surface_mask(self.pressure.values, self.ps.values, smooth=False)
            mask = DataArray(mask, dims=["latitude", "longitude", "level"])
        else:
            # mask all values as valid
            mask = True

        # Mask data pierced by the topography, if not already masked during interpolation.
        data = self[name].where(mask, np.nan)

        info_coords = "".join(data.coords[name].axis.lower() for name in data.dims)
        data = data.to_masked_array()

        # masked elements are filled with zeros before the spectral analysis
        data.set_fill_value(0.0)

        # Move dimensions (nlat, nlon) forward and vertical axis last
        # (Useful for cleaner vectorized operations)
        return prepare_data(data, info_coords)

    def integrate_levels(self, variable=None, coord_name="level", coord_range=None):

        if variable is None:
            # create a copy and integrate the entire dataset.
            data = self.copy()
        elif variable in self:
            data = self[variable]
        else:
            raise ValueError(f"Variable {variable} not found in dataset!")

        # Find vertical coordinate and sort 'coord_range' accordingly
        coord = self.find_coordinate(coord_name=coord_name)

        _range = (coord.values.min(), coord.values.max())
        if coord_range is None:
            coord_range = _range
        elif isinstance(coord_range, (list, tuple)):
            # Replaces None in 'coord_range' and sort the values. If None appears in the first
            # position, e.g. [None, b], the integration is done from the [coord.min, b].
            # Similarly for [a, None] , the integration is done from the [a, coord.max].
            coord_range = [r if c is None else c for c, r in zip(coord_range, _range)]
            coord_range = np.sort(coord_range)
        else:
            raise ValueError("Illegal type of parameter 'coord_range'. Expecting a scalar, "
                             "a list or tuple of length two defining the integration limits.")

        # Sorting dataset in ascending order along coordinate before selection. This ensures
        # that the selection of integration limits is inclusive (behaves like method='nearest')
        if is_sorted(coord.values, ascending=False):
            data = data.sortby(coord.name)

        # Integrate over coordinate 'coord_name' using the trapezoidal rule
        data = data.sel({coord.name: slice(*coord_range)}).integrate(coord.name)

        # convert to height coordinate if data is pressure levels (hydrostatic approximation)
        if is_standard(coord, "pressure"):
            data /= cn.g

        return data


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

    # get coordinate by explicit names
    coord = ds.coords.get(name)

    if coord is not None:
        return coord

    if name not in CF_coordinate_conventions.keys():
        raise ValueError("Wrong coordinate name: {!s}".format(name))

    criteria = CF_coordinate_conventions[name]

    predicate = lambda c: (c.name in criteria['standard_name'] or
                           c.attrs.get('units') in criteria['units'] or
                           c.attrs.get('axis') in criteria['axis'])

    candidates = [coord for coord in [ds.coords[n] for n in ds.dims] if predicate(coord)]

    if len(candidates) > 1:
        msg = 'Multiple {!s} coordinates are not allowed.'
        raise ValueError(msg.format(name))

    if not candidates:
        raise ValueError('Cannot find a {!s} coordinate'.format(name))

    return candidates[0]


def reindex_coordinate(coord, data):
    # Reindex dataset coordinate
    return coord.reindex({coord.name: data})


def is_standard(data, standard_name):
    if not isinstance(data, (DataArray, Dataset)):
        return False

    var_attrs = data.attrs

    cf_attrs = CF_variable_conventions[standard_name]

    has_name = np.any([var_attrs.get(name) in cf_attrs["standard_name"]
                       for name in ["name", "standard_name", "long_name"]]) or \
               data.name in cf_attrs["standard_name"]

    var_units = var_attrs.get("units")

    if var_units is not None:
        return has_name and var_units in cf_attrs["units"]
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
    if variable_name is None:
        return None

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
        if is_standard(dataset[var_name], standard_name):
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


def parse_dataset(dataset, variables=None, surface_data=None, p_levels=None):
    """
        Parse input xarray dataset

        Parameters
        ----------
        :param dataset: xarray.Dataset or str indicating the path to a dataset.

            The dataset must contain the following analysis fields:
            - The horizontal wind component in the zonal direction      (u)
            - The horizontal wind component in the meridional direction (v)
            - Height/pressure vertical velocity depending on leveltype (inferred from dataset)
            - Temperature
            - Atmospheric pressure: A 1D array for isobaric levels or a ND array for arbitrary
               vertical coordinate. Data is interpolated to pressure levels before the analysis.

        :param variables: dict, optional,
            A dictionary mapping of the field names in the dataset to the internal variable names.
            The default names are: ['u_wind', 'v_wind', 'omega', 'temperature', 'pressure'].
            Ensures all variables needed for the analysis are found. If not given, variables are
            looked up based on standard CF conventions of variable names, units and typical value
            ranges. Example: variables = {'u_wind': 'U', 'temperature': 'temp'}. Note that often
            used names 'U' and 'temp' are not conventional names.

        :param surface_data: a dictionary or xr.Dataset containing surface variables

        :param p_levels: iterable, optional
            Contains the pressure levels in (Pa) for vertical interpolation.
            Ignored if the data is already in pressure coordinates.

        Returns
        _______
        SebaDataset object containing the required analysis fields

    """
    # check if input dataset is a path to file
    if isinstance(dataset, str):
        dataset = open_mfdataset(dataset, combine='by_coords', parallel=False)

    if not isinstance(dataset, Dataset):
        raise TypeError("Input parameter 'dataset' must be xarray.Dataset instance"
                        "or a string containing the path to a netcdf file.")

    if surface_data is None:
        surface_data = {'': None}

    # Define list of mandatory variables in dataset. Raise error if missing. Pressure vertical
    # velocity 'omega' is estimated from height-based vertical velocity if not present.
    default_variables = ['u_wind', 'v_wind', 'temperature', 'pressure']

    if variables is None:
        variables = {name: name for name in default_variables}

    if not isinstance(variables, dict):
        raise ValueError("Unknown type for 'variables', must be a dictionary")

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

    # Create a SebaDataset instance for convenience
    dataset = SebaDataset(dataset)

    # rename spatial coordinates to standard names: (level, latitude, longitude)
    map_dims = {dataset.find_coordinate(name).name: name
                for name in ("level", "longitude", "latitude")}
    dataset = dataset.rename(map_dims)

    dataset.swap_dims({dataset.find_coordinate('time').name: 'time'})

    # find variables by name or candidates variables based on CF conventions
    arrays = {name_map[0]: dataset.find_variable(name_map, raise_notfound=True)
              for name_map in variables.items()}

    if len(arrays) != len(variables):
        raise ValueError("Missing variables!")

    # check dimensionality of variables
    for name, values in arrays.items():

        # Make sure the shapes of the two components match.
        if (name != 'pressure') and values.ndim < 3:
            raise ValueError('Fields must be at least 3D.'
                             'Variable {} has {} dimensions'.format(name, values.ndim))

    grid_shape = (dataset.latitude.size, dataset.longitude.size)

    # Find surface data arrays and assign to dataset if given externally
    for name in ['ts', 'ps']:

        # silence error: these variables are flexible.
        surface_var = dataset.find_variable(name, raise_notfound=False)

        if surface_var is None and surface_data is not None:
            print("Warning: surface variable {} not found!".format(name))

            if name in surface_data and is_standard(surface_data[name], name):

                surface_var = surface_data[name]

                if isinstance(surface_var, DataArray):
                    # inputs are DataArray
                    if 'time' in surface_var.dims:
                        surface_var = surface_var.mean('time')

                    # Renaming spatial coordinates. Raise error if no coordinate
                    # is consistent with latitude and longitude standards.
                    surface_var = surface_var.rename(
                        {_find_coordinate(surface_var, c).name: c
                         for c in ("latitude", "longitude")})

                elif isinstance(surface_var, np.ndarray):

                    if surface_var.shape != grid_shape:
                        raise ValueError(f'Given surface variable {name} must be a '
                                         f'scalar or a 2D array with shape (nlat, nlon).'
                                         f'Expected shape {grid_shape}, '
                                         f'but got {np.shape(surface_var)}')
                    surface_var = DataArray(surface_var, dims=("latitude", "longitude"))

                else:
                    raise ValueError(f'Illegal type for surface variable {name}!')

                arrays[name] = surface_var
        else:
            if 'time' in surface_var.dims:
                surface_var = surface_var.mean('time')

            # Surface data must be exactly one dimension less than the total number of dimensions
            # in dataset (Not sure if this is a good enough condition!?)
            if not np.all([dim in surface_var.dims for dim in ("latitude", "longitude")]):
                raise ValueError(f"Variable {name} must have at least latitude "
                                 f"and longitude dimensions.")

            arrays[name] = surface_var

    # Create output dataset with required fields.
    data = SebaDataset(arrays).check_convert_units()

    # Check for pressure velocity: If not present in dataset, it is estimated from
    # height-based vertical velocity using the hydrostatic approximation.
    data['omega'] = dataset.find_variable('omega', raise_notfound=False)

    if not data['omega'].all():
        # Searching for vertical velocity. Raise error if 'w_wind' cannot be found!
        w_wind = dataset.find_variable('w_wind', raise_notfound=True)

        # Checking 'w_wind' units and converting to standard before computing omega.
        w_wind = data.check_convert_units(w_wind)

        # compute omega from pressure [Pa], temperature [K] and vertical velocity [m/s]
        data['omega'] = pressure_vertical_velocity(w_wind, data.temperature, data.pressure)

        # add standard units [Ps/s]
        data.omega.attrs["units"] = expected_units['omega']

    # Perform interpolation to constant pressure levels as needed (after all dynamic fields added)
    # Ensures latitude dimension is ordered north-to-south and starting from the surface.
    data = data.interpolate_levels(p_levels=p_levels).sortby(["level", 'latitude'], ascending=False)

    # Compute geopotential if not present in dataset
    data['geopotential'] = dataset.find_variable('geopotential', raise_notfound=False)

    if not data['geopotential'].all():

        print("Computing geopotential ...")

        # reshape temperature to pass it to fortran subroutine
        target_dims = ('time', 'latitude', 'longitude', 'level')
        coord_order = [data.temperature.dims.index(dim) for dim in target_dims]

        data_shape = tuple(data.temperature.shape[dim] for dim in coord_order)
        proc_shape = (data_shape[0], -1, data_shape[-1])

        # get temperature field
        temperature = data.temperature.to_masked_array()
        temperature = np.transpose(temperature, axes=coord_order).reshape(proc_shape)
        temperature = np.ma.filled(temperature, fill_value=0.0)

        # If surface pressure is not given it is approximated by the pressure level
        # closest to the surface.
        if 'ps' not in data:
            sfcp = np.broadcast_to(np.max(data.pressure) + 1e2, grid_shape)
            data['ps'] = DataArray(sfcp, dims=("latitude", "longitude"))
            data['ps'].attrs['units'] = 'Pa'

        sfcp = data.ps.values.ravel()

        if 'ts' not in data:
            print("Computing surface temperature ...")

            # compute surface temperature by linear interpolation
            sfct = numeric_tools.surface_temperature(sfcp, temperature, data.pressure.values)

            # assign surface variables to dataset
            data['ts'] = DataArray(sfct.reshape(data_shape[:-1]), dims=target_dims[:-1])
            data['ts'].attrs['units'] = 'K'
        else:
            coord_order = [data.ts.dims.index(dim) for dim in ("time", "latitude", "longitude")]
            # noinspection PyTypeChecker
            sfct = np.transpose(data.ts.values, axes=coord_order).reshape(data.dims["time"], -1)

        # get surface elevation for the given grid.
        sfch = get_surface_elevation(data.latitude, data.longitude)
        sfch = sfch.values.ravel()

        # Compute geopotential from the temperature field: d(phi)/d(log p) = - Rd * T(p)
        phi = numeric_tools.geopotential(data.pressure.values, temperature, sfch, sfcp, sfct=sfct)
        phi = DataArray(phi.reshape(data_shape), dims=target_dims)

        data['geopotential'] = phi.transpose("time", "level", "latitude", "longitude")
        data['geopotential'].attrs['units'] = 'joule / kg'

    print("Data processing completed successfully!")

    # return dataset with proper variables and units (remove dask arrays)
    return data.compute()


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
    levels = _find_coordinate(dataset, "level")

    # Finding pressure array in dataset. Silence error if not found (try to use coordinate)
    pressure = _find_variable(dataset, 'pressure', raise_notfound=False)

    # If the vertical coordinate is consistent with pressure standards, then no interpolation is
    # needed: Data is already in pressure coordinates.
    if is_standard(levels, "pressure"):
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
        p_levels = np.linspace(1000e2, np.min(pressure), levels.size)  # [Pa]
    else:
        p_levels = np.array(p_levels)
        assert p_levels.ndim == 1, "'p_levels' must be a one-dimensional array."

    print("Interpolating data to {} isobaric levels ...".format(p_levels.size))

    dims = get_coordinate_names(dataset)
    coords = [dataset.coords[name] for name in dims]
    z_axis = ''.join([coord.axis for coord in coords]).lower().find('z')

    # Replace vertical coordinate with new pressure levels
    dims[z_axis] = "level"
    coords[z_axis] = DataArray(data=p_levels, name="level", dims=["level"],
                               coords=dict(plev=("level", p_levels)),
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

    print("Interpolation finished successfully.")

    # replace pressure field in dataset with the new coordinate
    attrs = {'standard_name': "pressure", 'units': "Pa"}
    result_dataset['pressure'] = ("level", p_levels, attrs)

    # Add fields in dataset that were not interpolated except 'pressure'.
    for svar in excluded_vars:
        sdata = dataset[svar]
        if not sdata.equals(pressure):
            result_dataset[svar] = (sdata.dims, sdata.values, sdata.attrs)

    # Create new dataset with interpolated data
    return SebaDataset(data_vars=result_dataset, coords=coords)


def get_surface_elevation(latitude, longitude):
    """
        Interpolates global topography data to desired grid defined by arrays lats and lons.
        The data source is SRTM15Plus (Global SRTM15+ V2.1) which can be accessed at
        https://opentopography.org/
    """

    # convert longitudes to range (-180, 180) if needed
    longitude = np.where(longitude > 180, longitude - 360, longitude)

    # infer the grid type from latitude points (raises error if grid is not regular or gaussian)
    grid_type, _, _ = inspect_gridtype(latitude)
    grid_prefix = "n" if grid_type == 'gaussian' else "r"

    grid_id = grid_prefix + str(latitude.size // 2)

    expected_path, _ = os.path.split(path_global_topo)
    expected_file = os.path.join(expected_path, f"topo_global_{grid_id}.nc")

    remap = True  # assume remap is needed
    if os.path.exists(expected_file):

        # load global elevation dataset
        ds = open_dataset(expected_file)

        # Sort according to required longitudes (west-to-east)
        # Latitudes are always sorted from north to south.
        if is_sorted(longitude, ascending=True):
            ds = ds.sortby('lon').sortby('lat', ascending=False)

        # Check if the longitudes in the file match the required longitudes.
        remap = not np.allclose(ds.lon.values, longitude)

    if remap:
        print(f"Warning: Surface elevation data with required resolution "
              f"not found! Interpolating global data to grid {grid_id}.")

        # Interpolate the global topography dataset to the required latitudes and longitudes
        # using the nearest neighbor method.
        ds = open_dataset(path_global_topo).interp(lat=latitude, lon=longitude,
                                                   method="nearest", kwargs={"fill_value": 0})

        # Export the interpolated dataset to a netcdf file for future use.
        ds.to_netcdf(expected_file)

    # Return the DataArray with the surface elevation.
    # noinspection PyUnboundLocalVariable
    return ds.elevation
