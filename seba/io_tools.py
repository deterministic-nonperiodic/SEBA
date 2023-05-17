import os
import re
from typing import Optional

import numpy as np
import pint
from xarray import Dataset, DataArray, open_dataset, open_mfdataset
from xarray import set_options

import constants as cn
from fortran_libs import numeric_tools
from thermodynamics import pressure_vertical_velocity
from tools import interpolate_1d, inspect_gridtype, gradient_1d
from tools import is_sorted, gaussian_lats_wts
from visualization import energy_spectra_by_levels
from visualization import fluxes_slices_by_models
from visualization import fluxes_spectra_by_levels

# Keep attributes after operations (why isn't this the default behaviour anyways...)
set_options(keep_attrs=True)

path_global_topo = "../data/topo_global_n1250m.nc"

CF_variable_conventions = {
    "geopotential": {
        "standard_name": ("geop", "geopotential", "air_geopotential",),
        "units": ('joule / kilogram', 'm**2 s**-2')
    },
    "temperature": {
        "standard_name": ("t", "ta", "temp", "air_temperature", 'temperature'),
        "units": ('K', 'kelvin', 'Kelvin', 'degree_C')
    },
    "pressure": {
        "standard_name": ("p", "air_pressure", "pressure", "pressure_level"),
        "units": ('Pa', 'hPa', 'mb', 'millibar', 'millibars')
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
    'divergence': {
        'standard_name': ('div', 'sd', 'divergence', 'horizontal divergence',
                          "horizontal_divergence", "wind_divergence", "divergence_of_wind"),
        'units': ("s-1", "s**-1", "1/s")
    },
    'vorticity': {
        'standard_name': ('vrt', 'svo', 'vor', 'vorticity', 'vertical vorticity',
                          "vertical_vorticity", "wind_vorticity", "atmosphere_relative_vorticity"),
        'units': ("s-1", "s**-1", "1/s")
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
                          'pressure', 'pressure_level', 'air_pressure', 'altitude',
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
    "divergence": "1/s",
    "vorticity": "1/s",
    "temperature": "K",
    "pressure": "Pa",
    "ps": "Pa",
    "ts": "K",
    "level": "Pa",
    "omega": "Pa/s",
    "geopotential": "m**2 s**-2",
}

expected_range = {
    "u_wind": [-350., 350.],
    "v_wind": [-350., 350.],
    "w_wind": [-100., 100.],
    "divergence": [-10., 10.],
    "vorticity": [-10., 10.],
    "temperature": [120, 350.],
    "pressure": [0.0, 2000e2],
    "ts": [120, 350.],
    "ps": [0.0, 2000e2],
    "omega": [-100, 100],
    "geopotential": [0., ],
}

# from Metpy
# Create a pint UnitRegistry object
UNITS_REG = pint.UnitRegistry()

# from Metpy
cmd = re.compile(r'(?<=[A-Za-z)])(?![A-Za-z)])(?<![0-9\-][eE])(?<![0-9\-])(?=[0-9\-])')


def _parse_units(unit_str):
    if isinstance(unit_str, (pint.Quantity, pint.Unit)):
        return unit_str
    else:
        return UNITS_REG(cmd.sub('**', unit_str))


def equivalent_units(unit_1, unit_2):
    ratio = (_parse_units(unit_1) / _parse_units(unit_2)).to_base_units()
    return ratio.dimensionless and np.isclose(ratio.magnitude, 1.0)


def compatible_units(unit_1, unit_2):
    return _parse_units(unit_1).is_compatible_with(_parse_units(unit_2))


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

    def coordinate_names(self):
        return get_coordinate_names(self)

    def coordinates_by_axes(self):
        cds = self.coords
        return {cds[name].axis.lower(): cds[name] for name in self.coordinate_names()}

    def coordinates_by_names(self):
        cds = self.coords
        return {name: cds[name] for name in self.coordinate_names()}

    def data_shape(self):
        return tuple(self.dims[name] for name in self.coordinate_names())

    def interpolate_levels(self, p_levels=None):
        return interpolate_pressure_levels(self, p_levels=p_levels)

    def check_convert_units(self, other=None):

        is_array = False
        if other is None:
            ds = self.copy()
        elif isinstance(other, Dataset):
            ds = other.copy()
        elif isinstance(other, DataArray):
            is_array = True
            ds = other.to_dataset(promote_attrs=True)
        else:
            raise ValueError("Illegal type for parameter 'other'")

        # Check and convert the units of each variable
        for varname in ds.data_vars:

            var = ds[varname]

            if varname not in expected_units:
                continue

            # define variable units
            var_units = var.attrs.get("units")

            # define expected units
            expected_unit = expected_units[varname]

            if var_units is None:
                print(f"Warning: Units not found for variable {varname}")

                if np.nanmax(var) <= expected_range[varname][1] and \
                        np.nanmin(var) >= expected_range[varname][0]:
                    # Update the units attribute of the variable
                    ds[varname].attrs["units"] = expected_unit
                    var_units = expected_unit
                else:
                    raise ValueError(f"Variable '{varname}' has no units and values are outside "
                                     f"admitted range: {expected_range[varname]}")

            if not equivalent_units(var_units, expected_unit):
                if compatible_units(var_units, expected_unit):
                    # Convert the values to the expected units
                    fixed_units = _parse_units(var_units)

                    values = (var.values * fixed_units).to(_parse_units(expected_unit)).magnitude

                    ds[varname] = (var.dims, values, var.attrs)

                    # Update the units attribute of the variable
                    ds[varname].attrs["units"] = expected_unit
                else:
                    raise ValueError(f"Cannot convert {varname} incompatible units"
                                     f"{var_units} to {expected_unit}!")

        if is_array:
            # get rid of extra coordinates and return original DataArray object
            return ds.to_array().squeeze('variable').drop_vars('variable')
        else:
            return ds

    def add_surface_data(self, surface_data=None):

        if surface_data is None:
            surface_data = {'': None}

        # spatial shape of variables in dataset
        grid_shape = (self.latitude.size, self.longitude.size)

        # Find surface data arrays and assign to dataset if given externally
        for name in ['ts', 'ps']:

            # silence error: these variables are flexible.
            surface_var = self.find_variable(name, raise_notfound=False)

            if surface_var is None and surface_data is not None:
                print("Warning: surface variable {} not found!".format(name))

                if name in surface_data and is_standard(surface_data[name], name):

                    surface_var = surface_data[name]
                    surface_var_attrs = {}
                    if isinstance(surface_var, DataArray):

                        # inputs are DataArray
                        if 'time' in surface_var.dims:
                            surface_var = surface_var.mean('time', keep_attrs=True)

                        # Renaming spatial coordinates. Raise error if no coordinate
                        # is consistent with latitude and longitude standards.
                        surface_var = surface_var.rename(
                            {_find_coordinate(surface_var, c).name: c
                             for c in ("latitude", "longitude")})

                        surface_var = surface_var.transpose('latitude', 'longitude')

                        # check units
                        surface_var = self.check_convert_units(surface_var)
                        surface_var_attrs = surface_var.attrs

                        # check dimensions match (not much precision needed), interpolate otherwise
                        same_size = surface_var.latitude.size == self.latitude.size

                        if not same_size or not np.allclose(surface_var.latitude,
                                                            self.latitude, atol=1e-4):
                            print(f'Interpolating surface data: {name}')

                            # extrapolate missing values usually points near the poles
                            surface_var = surface_var.interp(latitude=self.latitude,
                                                             longitude=self.longitude,
                                                             kwargs={"fill_value": "extrapolate"})
                        # convert to masked array
                        surface_var = surface_var.to_masked_array()

                    if isinstance(surface_var, np.ndarray):

                        if surface_var.shape != grid_shape:
                            raise ValueError(f'Given surface variable {name} must be a '
                                             f'scalar or a 2D array with shape (nlat, nlon).'
                                             f'Expected shape {grid_shape}, '
                                             f'but got {np.shape(surface_var)}')

                        # create DataArray and assign to dataset
                        surface_var = DataArray(surface_var, name=name,
                                                dims=("latitude", "longitude"),
                                                attrs=surface_var_attrs)
                    else:
                        raise ValueError(f'Illegal type for surface variable {name}! '
                                         f'If given it must be one of [xr.DataArray, np.ndarray].')

                    self[name] = surface_var
            else:
                if 'time' in surface_var.dims:
                    surface_var = surface_var.mean('time', keep_attrs=True)

                # Surface data must be exactly one dimension less than the total number of
                # dimensions in dataset (Not sure if this is a good enough condition!?)
                if not np.all([dim in surface_var.dims for dim in ("latitude", "longitude")]):
                    raise ValueError(f"Variable {name} must have at least latitude "
                                     f"and longitude dimensions.")

                self[name] = self.check_convert_units(surface_var)

        # If surface pressure is not found, it is approximated by the pressure
        # level closest to the surface. This is a fallback option for unmasked terrain.
        if 'ps' not in self and 'pressure' in self:
            pressure = self.check_convert_units(self.pressure)
            sfcp = np.broadcast_to(np.max(pressure) + 1e2, grid_shape)
            self['ps'] = DataArray(sfcp, dims=("latitude", "longitude"))
            self['ps'].attrs['units'] = 'Pa'

        return self

    def get_field(self, name, default=None, masked=True):
        """ Returns a field in dataset after processing to be used by seba.EnergyBudget:
            - Exclude extrapolated data below the surface (p >= ps).
            - Ensure latitude axis is oriented north-to-south.
            - Reverse vertical axis from the surface to the model top.
        """

        if name not in self.data_vars:
            return default

        # Calculate mask to exclude extrapolated data below the surface (p >= ps).
        # beta should be computed just once
        data = self[name]  # assume all values as valid

        if masked:
            if 'pressure' in self and 'ps' in self:
                # Mask data pierced by the topography, if not already masked during interpolation.
                data = data.where(self.pressure < self.ps.broadcast_like(self.pressure), np.nan)

        # sort data north-south and starting at the surface
        data = data.sortby(
            [dim for dim in ["latitude", "level"] if dim in data.coords], ascending=False
        )

        # Transpose dimensions for cleaner vectorized operations.
        # - moving dimensions 'lat' and 'lon' forward and 'levels' to last dimension
        target_dims = ('latitude', 'longitude', 'time', 'level')
        coord_order = [data.dims.index(dim) for dim in target_dims]

        # info_coords = "".join(data.coords[name].axis.lower() for name in data.dims)
        data = data.to_masked_array()

        # masked elements are filled with zeros before the spectral analysis
        data.set_fill_value(0.0)

        # prepare_data(data, info_coords)
        return np.ma.transpose(data, axes=coord_order)

    def _coordinate_range(self, name="level", limits=None):
        # Find vertical coordinate and sort 'coord_range' accordingly
        coord = self.find_coordinate(coord_name=name)

        c_range = [coord.values.min(), coord.values.max()]
        if limits is None:
            limits = c_range
        elif isinstance(limits, (list, tuple)):
            # Replaces None in 'coord_range' and sort the values. If None appears in the first
            # position, e.g. [None, b], the integration is done from the [coord.min, b].
            # Similarly for [a, None] , the integration is done from the [a, coord.max].
            if None not in limits:
                limits = sorted(limits)
            else:
                limits = [le if le is not None else r for le, r in zip(limits, c_range)]

            limits = np.clip(limits, *c_range)
        else:
            raise ValueError("Illegal type of parameter 'limits'. Expecting a scalar, "
                             "a list or tuple of length two defining the integration limits.")

        return sorted(limits), coord

    def difference_range(self, variable=None, dim="level", coord_range=None):
        if variable is None:
            # create a copy and integrate the entire dataset.
            data = self.copy()
        elif isinstance(variable, (list, tuple)):
            data = self[list(variable)]
        elif variable in self:
            data = self[variable].to_dataset(promote_attrs=True)
        else:
            raise ValueError(f"Variable {variable} not found in dataset!")

        # Find vertical coordinate and sort 'coord_range' accordingly
        coord_range, coord = self._coordinate_range(name=dim, limits=coord_range)
        coord_name = coord.name

        data = data.sel({coord_name: coord_range}, method='nearest').diff(dim=coord_name)

        return data.squeeze(drop=True)

    def gradient(self, variable=None, dim="level", order=6):
        """ Differentiate along a given coordinate using high-order compact
            finite difference scheme (Lele 1992).
        """
        if variable is None:
            # create a copy and integrate the entire dataset.
            data = self.copy()
        elif isinstance(variable, (list, tuple)):
            data = self[list(variable)]
        elif variable in self:
            data = self[variable].to_dataset(promote_attrs=True)
        else:
            raise ValueError(f"Variable {variable} not found in dataset!")

        # Find vertical coordinate and sort 'coord_range' accordingly
        coord = self.find_coordinate(coord_name=dim)
        axis = tuple(data.dims).index(coord.name)

        for name in data.data_vars:

            var_attrs = data[name].attrs

            data[name] = (data[name].dims, gradient_1d(data[name].values,
                                                       coord.values,
                                                       axis=axis, order=order))

            # keep attrs
            data[name].attrs.update(var_attrs)

            if "units" in coord.attrs and "units" in var_attrs:
                converted_units = _parse_units(var_attrs['units']) / _parse_units(coord.units)
                data[name].attrs['units'] = str(converted_units.units)

        return data

    def integrate_range(self, variable=None, dim="level", coord_range=None):

        if variable is None:
            # create a copy and integrate the entire dataset.
            data = self.copy()
        elif isinstance(variable, (list, tuple)):
            data = self[list(variable)]
        elif variable in self:
            data = self[variable].to_dataset(promote_attrs=True)
        else:
            raise ValueError(f"Variable {variable} not found in dataset!")

        # Find vertical coordinate and sort 'coord_range' accordingly
        coord_range, coord = self._coordinate_range(name=dim, limits=coord_range)

        # Sorting dataset in ascending order along coordinate before selection. This ensures
        # that the selection of integration limits is inclusive (behaves like method='nearest')
        if is_sorted(coord.values, ascending=False):
            data = data.sortby(coord.name)

        # save data units before integration (integrate drops units... it's getting annoying)
        var_units = {name: data[name].attrs.get("units") for name in data.data_vars}

        # Integrate over coordinate 'coord_name' using the trapezoidal rule
        data = data.sel({coord.name: slice(*coord_range)}).integrate(coord.name)

        # Convert to height coordinate if data is pressure levels (hydrostatic approximation)
        convert_units = coord.attrs.get("units") or ''

        if is_standard(coord, "pressure"):
            data /= cn.g
            convert_units = 'kg / m**2'

        # units conversion
        for name, units in var_units.items():
            if units:
                converted_units = _parse_units(units) * _parse_units(convert_units)
                data[name].attrs['units'] = str(converted_units.units)

        return data

    def cumulative_sum(self, variable=None, dim=None):
        """
        Compute the cumulative sum of Dataset variables along the given dimension,
        considering the sorting order of the dimension and starting from a given index.

        Parameters
        ----------
        variable: str, optional
            Name of the variable to accumulate. If None, operation is performed for all variable
            in dataset

        dim : str
            Dimension name along which to compute the cumulative sum.

        Returns
        -------
        xarray.Dataset
            Dataset with the cumulative sum along the given dimension. The accumulation is done by
            summing all variables in dataset along the sorted coordinate elements such that the
            result starts with the total sum at the first position along the given dimension.
        """
        if variable is None:
            # create a copy and integrate the entire dataset.
            data = self.copy()
        elif variable in self:
            data = self[variable]
        else:
            raise ValueError(f"Variable {variable} not found in dataset!")

        # Ensure the input dimension exists in the dataset
        if dim not in data.dims:
            raise ValueError(f"Dimension {dim} not found in dataset or selected DataArray.")

        coordinate = data[dim]

        # Compute the cumulative sum along the given dimension. The total sum is at the
        # first position along the given dimension.
        kwargs = dict(skipna=False, keep_attrs=True)

        data = data.sortby(dim, ascending=False).cumsum(dim=dim, **kwargs)

        # Assign dimension before sorting (cumsum drops coordinates for some reason!)
        return data.assign_coords({dim: coordinate[::-1]}).sortby(dim)

    # ----------------------------------------------------------------------------------------------
    # Functions for visualizing of energy spectra and spectral fluxes
    # ----------------------------------------------------------------------------------------------
    def visualize_fluxes(self, **kwargs):
        fluxes_spectra_by_levels(self, **kwargs)

    def visualize_energy(self, **kwargs):
        energy_spectra_by_levels(self, **kwargs)

    def visualize_slices(self, **kwargs):
        fluxes_slices_by_models(self, **kwargs)


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


def _parse_name(name):
    if isinstance(name, str):
        return name.lower()
    else:
        return None


def is_standard(data, standard_name):
    # check if a DataArray is standard CF variable
    if not isinstance(data, (DataArray, Dataset)):
        return False

    var_attrs = data.attrs

    cf_attrs = CF_variable_conventions[standard_name]

    has_name = np.any([_parse_name(var_attrs.get(name)) in cf_attrs["standard_name"]
                       for name in ["name", "standard_name", "long_name"]]) or \
               _parse_name(data.name) in cf_attrs["standard_name"]

    var_units = var_attrs.get("units")

    if var_units:
        # if variable has units enforce
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


def parse_dataset(dataset, variables=None, p_levels=None, **surface_data):
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

    # Create a SebaDataset instance for convenience (remove dask chunks)
    dataset = SebaDataset(dataset).compute()

    # rename spatial coordinates to standard names: (level, latitude, longitude)
    new_dims = ("time", "level", "longitude", "latitude")
    map_dims = {dataset.find_coordinate(name).name: name for name in new_dims}
    dataset = dataset.rename(map_dims).set_index({v: v for v in new_dims})
    # dataset.swap_dims({dataset.find_coordinate('time').name: 'time'})

    # check units for pressure vertical coordinate
    if is_standard(dataset.level, "pressure") and hasattr(dataset.level, "units"):
        # convert units
        values = (dataset.level.values * _parse_units(dataset.level.units))
        dataset = dataset.assign_coords({'level': values.to('Pa').magnitude})
        # recover attributes
        dataset.level.attrs.update(units='Pa', name='pressure', long_name='pressure_level')

    # find variables by name or candidates variables based on CF conventions
    data = {name_map[0]: dataset.find_variable(name_map, raise_notfound=True)
            for name_map in variables.items()}

    if len(data) != len(variables):
        raise ValueError("Missing variables!")

    # check dimensionality of variables
    for name, values in data.items():

        # Make sure the shapes of the two components match.
        if (name != 'pressure') and values.ndim < 3:
            raise ValueError(f'Fields must be at least 3D. '
                             f'Variable {name} has {values.ndim} dimensions')

    # Create output dataset with required fields and check units consistency.
    # Find surface data arrays and assign to dataset if given externally
    data = SebaDataset(data).add_surface_data(surface_data=surface_data).check_convert_units()

    # Check if vorticity and divergence are present in dataset
    for variable in ['vorticity', 'divergence']:

        values = dataset.find_variable(variable, raise_notfound=False)
        if values is not None:
            data[variable] = values

    # Check for pressure velocity: If not present in dataset, it is estimated from
    # height-based vertical velocity using the hydrostatic approximation.
    omega = dataset.find_variable('omega', raise_notfound=False)

    if omega is not None:
        data['omega'] = omega
    else:
        # Searching for vertical velocity. Raise error if 'w_wind' cannot be found!
        # This is neither 'omega' nor 'w_wind' are present in dataset.
        w_wind = dataset.find_variable('w_wind', raise_notfound=True)

        # Checking 'w_wind' units and converting to standard before computing omega.
        w_wind = data.check_convert_units(w_wind)

        # compute omega from pressure [Pa], temperature [K] and vertical velocity [m/s]
        data['omega'] = pressure_vertical_velocity(w_wind, data.temperature, data.pressure)

        # add standard units [Ps/s]
        data.omega.attrs["units"] = expected_units['omega']

    # Check if geopotential is present in dataset and add to data before interpolation
    phi = dataset.find_variable('geopotential', raise_notfound=False)

    if phi is not None:
        data['geopotential'] = phi

    # Perform interpolation to constant pressure levels if necessary (after all dynamic fields
    # added). Ensures latitude dimension is ordered north-to-south and vertical levels starting
    # at the surface.
    data = data.interpolate_levels(p_levels).sortby(["level", 'latitude'], ascending=False)

    # Computing geopotential if not found in dataset. This step must be done with data in
    # pressure coordinates sorted from the surface up.
    if 'geopotential' not in data:

        # reshape temperature to pass it to fortran subroutine
        target_dims = ('time', 'latitude', 'longitude', 'level')
        coord_order = [data.temperature.dims.index(dim) for dim in target_dims]

        data_shape = tuple(data.temperature.shape[dim] for dim in coord_order)
        proc_shape = (data_shape[0], -1, data_shape[-1])

        # get temperature field
        temperature = data.temperature.to_masked_array()
        temperature = np.transpose(temperature, axes=coord_order).reshape(proc_shape)
        temperature = np.ma.filled(temperature, fill_value=0.0)

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

        print("Get surface elevation ...")
        # get surface elevation for the given grid. Using interpolated data
        # from global dataset if it does not exist already.
        sfch = get_surface_elevation(data.latitude, data.longitude).values.ravel()

        print("Computing geopotential from temperature ...")
        # Compute geopotential from the temperature field: d(phi)/d(log p) = - Rd * T(p)
        phi = numeric_tools.geopotential(data.pressure.values, temperature, sfch, sfcp, sfct=sfct)
        phi = DataArray(phi.reshape(data_shape), dims=target_dims)

        data['geopotential'] = phi.transpose("time", "level", "latitude", "longitude")
        data['geopotential'].attrs['units'] = 'joule / kilogram'

    # Inspect grid type based on the latitude sampling and interpolate if required
    if data.latitude.size != data.longitude.size // 2:
        gridtype = 'gaussian'

        print(f'Interpolating to {gridtype} grid ...')
        latitude, _ = gaussian_lats_wts(data.longitude.size // 2)

        data = data.interp(latitude=latitude)
    else:
        gridtype, *_ = inspect_gridtype(data.latitude)

    data.attrs.update(gridtype=gridtype)

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
            dataset["pressure"] = levels.astype(float)

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
