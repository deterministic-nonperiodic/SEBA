import numpy as np
from xarray import apply_ufunc, Dataset, DataArray

from thermodynamics import pressure_vertical_velocity
from tools import interpolate_1d

CF_variable_conventions = {
    "temperature": {
        "standard_name": ("air_temperature", 'temperature'),
        "units": ('K', 'kelvin', 'Kelvin', 'degree_C')
    },
    "pressure": {
        "standard_name": ("air_pressure", "pressure"),
        "units": ('Pa', 'hPa', 'mb', 'millibar')
    },
    "omega": {
        "standard_name": ("lagrangian_tendency_of_air_pressure",
                          'pressure_velocity', 'pressure velocity'),
        "units": ("Pa s-1", 'Pa s**-1', 'Pa/s')
    },
    "u": {
        "standard_name": ('zonal_wind', 'zonal wind', "eastward_wind"),
        "units": ("m s-1", "m s**-1", "m/s")
    },
    "v": {
        "standard_name": ('meridional_wind', 'Meridional wind', "northward_wind"),
        "units": ("m s-1", "m s**-1", "m/s")
    },
    'w': {
        'standard_name': ('vertical_velocity', 'vertical velocity', "upward_air_velocity"),
        'units': ("m s-1", "m s**-1", "m/s")
    },
    'ts': {
        "standard_name": ('surface_temperature', 'surface_air_temperature', 'ts', 'temp_sfc'),
        "units": ('K', 'kelvin', 'Kelvin', 'degree_C')
    },
    "ps": {
        "standard_name": ('surface_pressure', 'surface_air_pressure', 'ps', 'sfcp', 'pres_sfc'),
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
        "units": ('meter', 'm', 'Pa', 'hPa', 'mb', 'millibar'),
        "axis": ('Z', 'vertical')
    }
}


def map_func(func, data, dim="plev", **kwargs):
    # map function to all variables in dataset along axis
    res = apply_ufunc(func, data, input_core_dims=[[dim]],
                      kwargs=kwargs, dask='allowed',
                      vectorize=True)

    if 'pressure_range' in kwargs.keys() and isinstance(data, Dataset):
        res = res.assign_coords({'layer': kwargs['pressure_range']})

    return res


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


def _find_variable(dataset, variable_name, raise_notfound=True):
    """
    Given a xarray dataset and a variable name, identifies whether the variable
    is in the dataset conforming to CF conventions.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset to identify the variable in.
    variable_name : str, dict
        The name of the variable to identify.
    raise_notfound: bool, optional, default True
        If true, raises error if variable is not found, or returns None otherwise.
    Returns
    -------
    str
        A string indicating the type of variable that is in the dataset. Possible
        values are "temperature", "pressure", "vertical velocity", "u", and "v".
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

    if array is None:
        candidates = []

        for var_name in dataset.variables:
            # Check if the variable is in the dataset and consistent with the CF conventions
            var_attrs = dataset[var_name].attrs

            cf_attrs = CF_variable_conventions[standard_name]

            has_name = np.any([var_attrs.get(name) in cf_attrs["standard_name"]
                               for name in ["name", "standard_name", "long_name"]])

            if has_name and (var_attrs.get("units") in cf_attrs["units"]):
                candidates.append(dataset[var_name])

        if not candidates:
            if raise_notfound:
                # variable is not in the dataset or not consistent with the CF conventions
                msg = "The variable {} is not in the dataset or is " \
                      "inconsistent with the CF conventions."
                raise ValueError(msg.format(variable_name))
            else:
                return None

        array = candidates[0]

    return array


def parse_dataset(dataset, variables=None):
    """
        Parse input xarray dataset

        Returns
        _______
        arrays: a list of requested DataArray objects

        info_coords: string containing the order of the spatial dimensions: 'z' for levels,
              'y' latitude, and 'x' for longitude are mandatory, while any other character
              may be used for other dimensions. Default is 'tzyx' or (time, levels, lat, lon)
    """
    default_variables = ['u', 'v', 'omega', 'temperature', 'pressure']

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

    # Check for pressure velocity in dataset. If not present, it is estimated from
    # height based vertical velocity. If neither is found raises ValueError.
    if arrays.get('omega') is None:
        p = arrays.get('pressure').values
        t = arrays.get('temperature').values
        w = _find_variable(dataset, 'w', raise_notfound=True)

        arrays['omega'] = pressure_vertical_velocity(p, np.array(w), t)

    # find surface data arrays
    for name in ['ts', 'ps']:

        sfc_var = _find_variable(dataset, name, raise_notfound=False)

        # surface data must have one less dimension
        if np.ndim(sfc_var) == len(dataset.dims) - 1:
            arrays[name] = sfc_var

    # create dataset and fill nans with zeros for spectral computations
    return Dataset(arrays, coords=dataset.coords).fillna(0.0)


def regrid_levels(dataset, p_levels=None):
    """Find a vertical coordinate in an `xarray.DataArray`."""

    # find vertical coordinate
    levels, nlevels = _find_coordinate(dataset, "level")

    # find pressure array
    pressure = _find_variable(dataset, 'pressure')

    # Perform interpolation to constant pressure levels if needed:
    data_shape = tuple(dataset.dims[i] for i in dataset.dims)

    if levels.ndim != np.ndim(pressure):
        # check pressure dimensions
        if np.shape(pressure) != data_shape:
            raise ValueError("Pressure field must match the dimensionality of the other "
                             "dynamic fields when using height coordinates."
                             "Expected shape {}".format(data_shape))

        if p_levels is None:
            # creating levels from pressure range and number of levels if no pressure levels
            # are given (set bottom level to 1000 hPa)
            p_levels = np.linspace(1000e2, np.min(pressure), nlevels)  # [Pa]
        else:
            p_levels = np.array(p_levels)
            assert p_levels.ndim == 1, "If given, 'p_levels' must be a 1D array."

        nlevels = p_levels.size

        print("Interpolating data to {} isobaric levels ...".format(nlevels))

        # values outside interpolation range are masked (levels below the surface p > ps)
        exclude_vars = ['pressure', 'ps', 'ts']

        dims = [dim for dim in dataset.dims]
        coords = [dataset.coords[d] for d in dataset.dims]
        z_axis = ''.join([coord.axis for coord in coords]).lower().find('z')

        dims[z_axis] = 'plev'
        coords[z_axis] = DataArray(data=p_levels, name='plev', dims=["plev"],
                                        coords=dict(plev=("plev", p_levels)),
                                        attrs=dict(standard_name="pressure",
                                                   long_name="pressure", positive="up",
                                                   units="Pa", axis="Z"))
        coords = {coord.name: coord for coord in coords}

        result_dataset = {}
        for name, data in dataset.data_vars.items():
            if name not in exclude_vars:
                result, = interpolate_1d(p_levels, pressure.values, data.values,
                                         scale='log', fill_value=np.nan, axis=z_axis)

                result_dataset[name] = (dims, result, data.attrs)

        # add variable pressure
        attrs = {'standard_name': "pressure", 'units': "Pa"}
        result_dataset['pressure'] = ('plev', p_levels, attrs)

        dataset = Dataset(data_vars=result_dataset, coords=coords)

    return dataset
