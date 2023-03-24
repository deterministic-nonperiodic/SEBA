import functools
import time

import numpy as np
import scipy.signal as sig
import scipy.special as spec
import scipy.stats as stats
from joblib import Parallel, delayed, cpu_count
from scipy.spatial import cKDTree
from xarray import apply_ufunc, Dataset

from fortran_libs import numeric_tools
from spectral_analysis import lambda_from_deg


class Timer(object):
    # Simple class to perform profiling and check code performance
    def __init__(self, title=""):
        self.title = title

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.interval = time.time() - self.start
        print("{}: time elapsed: {}".format(self.title, self.interval))


def _find_coordinates(array, predicate, name):
    """
    Find a dimension coordinate in an `xarray.DataArray` that satisfies
    a predicate function.
    """
    candidates = [coord
                  for coord in [array.coords[n] for n in array.dims]
                  if predicate(coord)]
    if not candidates:
        raise ValueError('cannot find a {!s} coordinate'.format(name))
    if len(candidates) > 1:
        msg = 'multiple {!s} coordinates are not allowed'
        raise ValueError(msg.format(name))
    coord = candidates[0]
    dim = array.dims[coord.name]
    return coord, dim


def _find_latitude(dataset):
    """Find a latitude dimension coordinate in an `xarray.Dataset`."""
    return _find_coordinates(dataset, lambda c: (c.name in ('latitude', 'lat') or
                                                 c.attrs.get('units') == 'degrees_north' or
                                                 c.attrs.get('axis') == 'Y'), 'latitude')


def _find_longitude(dataset):
    """Find a latitude dimension coordinate in an `xarray.DataArray`."""
    return _find_coordinates(dataset, lambda c: (c.name in ('longitude', 'lon') or
                                                 c.attrs.get('units') == 'degrees_east' or
                                                 c.attrs.get('axis') == 'X'), 'longitude')


def _find_variable(dataset, name, var_attrs):
    """
    Find a dimension coordinate in an `xarray.DataArray` that satisfies
    a predicate function.
    """

    # trying variable selection by name
    array = dataset.variables.get(name)

    if array is None:
        # trying flexible candidates for the variable based on attributes
        def predicate(d):
            return any([d.attrs.get(key) in values for key, values in var_attrs.items()])

        # list all candidates
        candidates = [data for name, data in dataset.variables.items() if predicate(data)]

        if not candidates:
            raise ValueError('Cannot find variable {!s} in dataset.'.format(name))

        array = candidates[0]

    return array


def inspect_leveltype(dataset):
    """Find a vertical coordinate in an `xarray.DataArray`."""

    levels, _ = _find_coordinates(dataset,
                                  lambda c: (c.name in ('p', 'plev', 'pressure') or
                                             c.attrs.get('units') in ('Pa', 'hPa',
                                                                      'mb', 'millibar') or
                                             c.attrs.get('axis') == 'Z'), 'pressure')

    p = _find_variable(dataset, 'p', {'long_name': ['pressure', 'air_pressure'],
                                      'units': ['Pa', 'hPa', 'mb', 'millibar']})

    leveltype = ['pressure', 'height'][levels.ndim != p.ndim]

    return leveltype, levels.size


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
    var_map = {
        'u': {'long_name': ['zonal_wind', 'zonal wind'],
              'units': ['m s**-1', 'm/s'], 'code': [131]},
        'v': {'long_name': ['meridional_wind', 'meridional wind'],
              'units': ['m s**-1', 'm/s'], 'code': [132]},
        # 'w': {'long_name': ['vertical_velocity', 'vertical velocity'],
        #       'units': ['m s**-1', 'm/s'], 'code': [120]},
        'omega': {'long_name': ['pressure_velocity', 'pressure velocity'],
                  'units': ['Pa s**-1', 'Pa/s'], 'code': [135]},
        't': {'long_name': ['temperature', 'air_temperature'],
              'units': ['K', 'kelvin'], 'code': [130]},
        'p': {'long_name': ['pressure', 'air_pressure'], 'units': ['Pa', 'hPa', 'mb']}
    }

    if variables is None:
        variables = list(var_map.keys())

    # find variables by name or candidates from attrs
    arrays = {name: _find_variable(dataset, name, attrs)
              for (var_key, attrs), name in zip(var_map.items(), variables)}

    if len(arrays) != len(variables):
        raise ValueError("Missing variables!")

    # check sanity of 3D fields
    for name, values in arrays.items():

        if np.isnan(values).any():
            raise ValueError('Array {} contain missing values'.format(name))

        # Make sure the shapes of the two components match.
        if (name != 'p') and values.ndim < 3:
            raise ValueError('Fields must be at least 3D.'
                             'Variable {} has {} dimensions'.format(name, values.ndim))

    # create dataset and fill nans with zeros for spectral computations
    dataset = Dataset(arrays, coords=dataset.coords).fillna(0.0)

    return dataset


def prepare_data(data, dim_order):
    """
    Prepare data for input to `EnergyBudget` method calls.

    Parameters:
    -----------
        data: `ndarray`
          Data array. The array must be at least 3D.
        dim_order: `string`,
          String specifying the order of dimensions in the data array. The
          characters 'x' and 'y' represent longitude and latitude
          respectively. Any other characters can be used to represent
          other dimensions.
    Returns:
    --------
        pdata: `ndarray`
          data reshaped/reordered to (latitude, longitude, other, levels).

        info_dict: `dict`
            A dictionary of information required to recover data.

    Examples:
    _________
    Prepare an array with dimensions (12, 17, 73, 144, 2) where the
    dimensions are (time, level, latitude, longitude, other):
      pdata, out_order = prep_data(data, 'tzyxs')

    The ordering of the output data dimensions is out_order = 'yx(ts)z',
    where the non-spatial dimensions between brackets are packed into a single axis:
    pdata.shape = (73, 144, 24, 17)
    """
    if data.ndim < 3:
        raise ValueError('Input fields must be at least 3D')

    if len(dim_order) > data.ndim:
        raise ValueError("Inconsistent number dimensions"
                         "'dim_order' must have length {}".format(data.ndim))

    if 'x' not in dim_order or 'y' not in dim_order:
        raise ValueError('A latitude-longitude grid is required')

    if 'z' not in dim_order:
        raise ValueError('A vertical grid is required')

    spatial_dims = [dim_order.lower().find(dim) for dim in 'yxz']

    data = np.moveaxis(data, spatial_dims, [0, 1, -1])

    # pack sample dimension
    inter_shape = data.shape

    data = data.reshape(inter_shape[:2] + (-1, inter_shape[-1]))  # .squeeze()

    out_order = dim_order.replace('x', '')
    out_order = out_order.replace('y', '')
    out_order = out_order.replace('z', '')
    out_order = 'yx(' + out_order + ')z'

    info_dict = {
        'interm_shape': inter_shape,
        'origin_order': dim_order,
        'output_order': out_order,
    }
    return data, info_dict


def recover_data(data, info_dict):
    """
    Recover the shape and dimension order of an array output
    after calling 'prepare_data'.
    """
    data = data.reshape(info_dict['interm_shape'])

    spatial_dims = [info_dict['origin_order'].find(dim) for dim in 'yxz']

    return np.moveaxis(data, [0, 1, -1], spatial_dims)


def recover_spectra(data, info_dict):
    """
    Recover the shape and dimension order of an array output
    after calling 'prepare_data'.
    """
    spectra_order = info_dict['origin_order'].replace('x', '')
    spatial_dims = [spectra_order.find(dim) for dim in 'yz']

    return np.moveaxis(data, [0, -1], spatial_dims)


def map_func(func, data, dim="plev", **kwargs):
    # map function to all variables in dataset along axis
    res = apply_ufunc(func, data, input_core_dims=[[dim]],
                      kwargs=kwargs, dask='allowed',
                      vectorize=True)

    if 'pressure_range' in kwargs.keys() and isinstance(data, Dataset):
        res = res.assign_coords({'layer': kwargs['pressure_range']})

    return res


def get_num_cores():
    """Returns number of physical CPU cores"""
    return np.ceil(0.5 * cpu_count()).astype(int)


def get_chunk_size(n_workers, len_iterable, factor=4):
    """Calculate chunk size argument for Pool-methods.

    Resembles source-code within `multiprocessing.pool.Pool._map_async`.
    """
    chunk_size, extra = divmod(len_iterable, n_workers * factor)
    if extra:
        chunk_size += 1
    return chunk_size


def get_number_chunks(sample_size, workers, factor=4):
    """
        Calculate number of chunks for Pool-methods.
    """
    n_chunks = sample_size / get_chunk_size(workers, sample_size, factor=factor)

    return np.ceil(n_chunks).astype(int)


def number_chunks(sample_size, workers):
    # finds the integer factor of 'sample_size' closest to 'workers'
    # for parallel computations: ensures maximum cpu usage for chunk_size = 1
    if sample_size < 2:
        return 1

    jobs = workers
    while sample_size % jobs:
        jobs -= 1
    return jobs if jobs != 1 else workers


def broadcast_1dto(arr, shape):
    """
    Broadcast a 1-dimensional array to a given shape using numpy rules
    by appending dummy dimensions. Raises error if the array cannot be
    broadcast to target shape.
    """
    a_size = arr.size
    if a_size in shape:
        # finds corresponding dimension from left to right.
        # if shape contains multiple dimensions with the same size
        # the result is broadcast to the left-most dimension
        index = shape.index(a_size)
    else:
        raise ValueError("Array of size {} cannot "
                         "be broadcast to shape: {}".format(a_size, shape))

    # create extra dimensions to be appended to the array
    lag = index == 0
    extra_dims = tuple(range(index + int(lag), len(shape) - int(not lag)))

    return np.expand_dims(arr, extra_dims)


def rotate_vector(vector, axis=0):
    if axis != 0:
        vector = np.moveaxis(vector, axis, 0)

    rotated = np.stack([-vector[1], vector[0]])

    if axis != 0:
        rotated = np.moveaxis(rotated, 0, axis)
    return rotated


def getspecindx(ntrunc):
    """
     compute indices of zonal wavenumber (index_m) and degree (index_n)
     for complex spherical harmonic coefficients.
     @param ntrunc: spherical harmonic triangular truncation limit.
     @return: C{B{index_m, index_n}} - rank 1 numpy Int32 arrays
     containing zonal wavenumber (index_m) and degree (index_n) of
     spherical harmonic coefficients.
    """
    index_m, index_n = np.indices((ntrunc + 1, ntrunc + 1))

    indices = np.nonzero(np.greater(index_n, index_m - 1).flatten())
    index_n = np.take(index_n.flatten(), indices)
    index_m = np.take(index_m.flatten(), indices)

    return np.squeeze(index_m), np.squeeze(index_n)


def transform_io(func, order='C'):
    """
    Decorator for handling arrays' IO dimensions for calling spharm's spectral functions.
    The dimensions of the input arrays with shapes (nlat, nlon, nlev, ntime, ...)
    or (ncoeffs, nlev, ntime, ...) are packed to (nlat, nlon, samples) and (ncoeffs, samples)
    respectively, where ncoeffs = (ntrunc+1)*(ntrunc+2)/2. Finally, the outputs are transformed
    back to the original shape where needed.

    Parameters:
    -----------
    func: decorated function
    order: {‘C’, ‘F’, ‘A’}, optional
        Reshape the elements of the input arrays using this index order.
        ‘C’ means to read / write the elements using C-like index order, with the last axis index
        changing fastest, back to the first axis index changing slowest. See 'numpy.reshape' for
        details.
    """

    @functools.wraps(func)
    def dimension_packer(*args, **kwargs):
        # self passed as first argument
        self, *_ = args
        transformed_args = [self, ]
        for arg in args:
            if isinstance(arg, np.ndarray):
                # fill masked arrays before spectral transforms
                arg = np.ma.fix_invalid(arg).filled(fill_value=0.0)
                transformed_args.append(self._pack_levels(arg, order=order))

        results = func(*transformed_args, **kwargs)
        # convert output back to original shape
        return self._unpack_levels(results, order=order)

    return dimension_packer


def regular_lats_wts(nlat):
    """
    Computes the latitude points and weights of a regular grid
    (equally spaced in longitude and latitude). Regular grids
    will include the poles and equator if nlat is odd. The sampling
    is a constant 180 deg/nlat. Weights are defined as the cosine of latitudes.

    Parameters:
        nlat (int): The number of latitude points.

    Returns:
        latitudes (numpy.ndarray): A 1D array containing the regular latitudes.
        weights (numpy.ndarray): A 1D array containing the corresponding weights.
    """
    ns_latitude = 90. - (nlat + 1) % 2 * (90. / nlat)

    lats = np.linspace(ns_latitude, -ns_latitude, nlat)

    return lats, np.cos(np.deg2rad(lats))


def gaussian_lats_wts(nlat):
    """
    Returns the Gaussian quadrature latitudes and weights for a given number of latitude points.

    Parameters:
        nlat (int): The number of latitude points.

    Returns:
        latitudes (numpy.ndarray): A 1D array containing the Gaussian quadrature latitudes.
        weights (numpy.ndarray): A 1D array containing the corresponding weights.
    """
    nodes, weights = spec.roots_legendre(nlat)
    # ensure north-south orientation (weights are symmetric)
    latitudes = np.arcsin(nodes[::-1]) * 180.0 / np.pi

    return latitudes, weights


def inspect_gridtype(latitudes):
    """
    Determine a grid type by examining the points of a latitude
    dimension.
    Raises a ValueError if the grid type cannot be determined.
    **Argument:**
    *latitudes*
        An iterable of latitude point values.
    **Returns:**
    *gridtype*
        Either 'gaussian' for a Gaussian grid or 'regular' for an
        equally-spaced grid.
    *reference latitudes*
    *quadrature weights*
    """
    # Define a tolerance value for differences, this value must be much
    # smaller than expected grid spacings.
    tolerance = 5e-8

    # Get the number of latitude points in the dimension.
    nlat = len(latitudes)
    diffs = np.abs(np.diff(latitudes))
    equally_spaced = (np.abs(diffs - diffs[0]) < tolerance).all()

    if equally_spaced:
        # The latitudes are equally-spaced. Construct reference global
        # equally spaced latitudes and check that the two match.
        reference, weights = regular_lats_wts(nlat)

        if not np.allclose(latitudes, reference, atol=tolerance):
            raise ValueError('Invalid equally-spaced latitudes (they may be non-global)')
        gridtype = 'regular'
    else:
        # The latitudes are not equally-spaced, which suggests they might be gaussian.
        # Construct sample gaussian latitudes and check if the two match.
        reference, weights = gaussian_lats_wts(nlat)

        if not np.allclose(latitudes, reference, atol=tolerance):
            raise ValueError('Wrong grid type: latitudes are neither equally-spaced or Gaussian')
        gridtype = 'gaussian'

    return gridtype, reference, weights


def cumulative_flux(spectra, axis=0):
    """
    Computes cumulative spectral energy transfer. The spectra are added starting
    from the largest wave number N (triangular truncation) to a given degree l.
    """
    spectra_flux = spectra.copy()
    dim_axis = spectra_flux.shape[axis]

    if axis != 0:
        spectra_flux = np.moveaxis(spectra_flux, axis, 0)

    # Set fluxes to 0 at ls=0 to avoid small truncation errors.
    for ln in range(dim_axis):
        spectra_flux[ln] = np.nansum(spectra_flux[ln:], axis=0)

    return np.moveaxis(spectra_flux, 0, axis)


def kernel_2d(fc, n):
    """ Generate a low-pass Lanczos kernel
        :param fc: float or  iterable [float, float],
            cutoff frequencies for each dimension (normalized by the sampling frequency)
        :param n: size of one quadrant of the circular kernel.
    """
    fc_sq = np.prod(fc)
    ns = 2 * n + 1

    # construct wavenumbers
    k = np.moveaxis(np.indices([ns, ns]) - n, 0, -1)

    z = np.sqrt(np.sum((fc * k) ** 2, axis=-1))
    w = fc_sq * spec.j1(2 * np.pi * z) / z.clip(1e-12)
    w *= np.prod(spec.sinc(np.pi * k / n), axis=-1)

    w[n, n] = np.pi * fc_sq

    return w / w.sum()


def lowpass_lanczos(data, window_size, cutoff_freq, axis=None, jobs=None):
    if axis is None:
        axis = -1

    arr = np.moveaxis(data, axis, 0)

    if jobs is None:
        jobs = min(cpu_count(), arr.shape[0])

    # compute lanczos 2D kernel for convolution
    kernel = kernel_2d(cutoff_freq, window_size)
    kernel = np.expand_dims(kernel, 0)

    # padding array using circular boundary conditions along the longitudinal dimension
    # and constant mirror reflection along latitudinal dimension with the kernel size.
    arr = np.pad(arr, ((0,), (window_size,), (0,)), mode='reflect')
    arr = np.pad(arr, ((0,), (0,), (window_size,)), mode='wrap')

    # wrapper of convolution function for parallel computations
    convolve_2d = functools.partial(sig.fftconvolve, in2=kernel, mode='same', axes=(1, 2))

    # Chunks of arrays along axis=0 for the mp mapping ...
    n_chunks = get_number_chunks(arr.shape[0], jobs, factor=4)

    # Create pool of workers
    pool = Parallel(n_jobs=jobs, backend="threading")

    # applying lanczos filter in parallel
    result = np.array(pool(delayed(convolve_2d)(chunk) for chunk in np.array_split(arr, n_chunks)))

    result = np.concatenate(result, axis=0)

    result[np.isnan(result)] = 1.0

    # remove added pad
    result = result[:, window_size:-window_size, window_size:-window_size]

    return np.moveaxis(result, 0, axis)


def search_closet(points, target_points):
    if target_points is None:
        return slice(None)
    else:
        points = np.atleast_2d(points).T
        target_points = np.atleast_2d(target_points).T
        # creates a search tree
        # noinspection PyArgumentList
        search_tree = cKDTree(points)
        # nearest neighbour (k=1) in levels to each point in target levels
        _, nn_idx = search_tree.query(target_points, k=1)

        return nn_idx


def indices_to_3d(mask, size):
    shape = tuple(mask.shape) + (size,)

    mask_bc, result_bc = np.broadcast_arrays(mask[..., np.newaxis], np.zeros(shape))
    result_bc[mask_bc <= np.arange(size)] = 1.0

    return result_bc


def terrain_mask(p, ps, smooth=True, jobs=None):
    """
    Creates a terrain mask based on surface pressure and pressure profile
    :param: smoothed, optional
        Apply a low-pass filter to the terrain mask
    :return: 'np.array'
        beta contains 0 for levels satisfying p > ps and 1 otherwise
    """
    if jobs is None:
        jobs = get_num_cores()
    else:
        jobs = int(jobs)

    nlevels = p.size
    nlat, nlon = ps.shape

    # Search last level pierced by terrain for each vertical column
    level_m = p.size - np.searchsorted(np.sort(p), ps)

    # create 3D mask from 2D mask
    beta = indices_to_3d(level_m, nlevels)

    if smooth:  # generate a smoothed heavy-side function
        # Calculate normalised cut-off frequencies for zonal and meridional directions:
        resolution = lambda_from_deg(nlon)  # grid spacing at the Equator
        cutoff_scale = lambda_from_deg(80)  # wavenumber 40 (scale ~500 km) from A&L (2013)

        # Normalized spatial cut-off frequency (cutoff_frequency / sampling_frequency)
        cutoff_freq = resolution / cutoff_scale
        # window size set to cutoff scale in grid points
        window_size = (2.0 / np.min(cutoff_freq)).astype(int)

        # Smoothing mask with a low-pass Lanczos filter (axis is the non-spatial dimension)
        beta = lowpass_lanczos(beta, window_size, cutoff_freq, axis=-1, jobs=jobs)

    # clipping is necessary to remove overshoots
    return beta.clip(0.0, 1.0)


def _select_by_distance(priority, distance):
    """
    Evaluate which peaks fulfill the distance condition.
    Parameters
    ----------
    priority : ndarray
        An array matching `peaks` used to determine priority of each peak. A
        peak with a higher priority value is kept over one with a lower one.
    distance :
        Minimal distance that peaks must be spaced.
    Returns
    -------
    keep : ndarray[bool]
        A boolean mask evaluating to true where `peaks` fulfill the distance
        condition.
    """

    peaks_size = priority.shape[0]
    # Round up because actual peak distance can only be natural number
    keep = np.ones(peaks_size, dtype=np.uint8)  # Prepare array of flags

    distance_ = np.ceil(distance)
    # Create map from `i` (index for `peaks` sorted by `priority`) to `j` (index
    # for `peaks` sorted by position). This allows to iterate `peaks` and `keep`
    # with `j` by order of `priority` while still maintaining the ability to
    # step to neighbouring peaks with (`j` + 1) or (`j` - 1).
    priority_to_position = np.argsort(priority)

    # Highest priority first -> iterate in reverse order (decreasing)
    for i in range(peaks_size - 1, -1, -1):
        # "Translate" `i` to `j` which points to current peak whose
        # neighbours are to be evaluated
        j = priority_to_position[i]
        if keep[j] == 0:
            # Skip evaluation for peak already marked as "don't keep"
            continue

        k = j - 1
        # Flag "earlier" peaks for removal until minimal distance is exceeded
        while 0 <= k and j - k < distance_:
            keep[k] = 0
            k -= 1

        k = j + 1
        # Flag "later" peaks for removal until minimal distance is exceeded
        while k < peaks_size and k - j < distance_:
            keep[k] = 0
            k += 1

    return keep.astype(bool)  # Return as boolean array


def intersections(coords, a, b, direction='all', return_y=False):
    # Return intersections between arrays a and b with common coordinate 'coords'
    coords = np.asarray(coords)
    a = np.asarray(a)
    b = np.asarray(b)

    if a.size == 1:
        if b.size != 1:
            a = np.repeat(a, b.size)
    else:
        if b.size == 1:
            b = np.repeat(b, a.size)
        else:
            assert a.size == b.size, "Arrays 'a' and 'b' must be the same size"

    assert coords.size == a.size, "Array 'coords' must be the same size as 'a' and 'b'"

    x, y = find_intersections(coords, a, b, direction=direction)

    if len(x) == 0:
        x = y = np.nan
    elif len(x) == 1:
        x = x[0]
        y = y[0]
    else:
        pass

    if return_y:
        return x, y
    else:
        return x


def find_intersections(x, a, b, direction='all'):
    """Calculate the best estimate of intersection.

    Calculates the best estimates of the intersection of two y-value
    data sets that share a common x-value set.

    Parameters
    ----------
    x : array-like
        1-dimensional array of numeric x-values
    a : array-like
        1-dimensional array of y-values for line 1
    b : array-like
        1-dimensional array of y-values for line 2
    direction : string
        specifies direction of crossing. 'all', 'increasing' (a becoming greater than b),
        or 'decreasing' (b becoming greater than a).

    Returns
    -------
        A tuple (x, y) of array-like with the x and y coordinates of the
        intersections of the lines.
    """
    # Find the index of the points just before the intersection(s)
    nearest_idx = nearest_intersection_idx(a, b)
    next_idx = nearest_idx + 1

    # Determine the sign of the change
    sign_change = np.sign(a[next_idx] - b[next_idx])

    # x-values around each intersection
    _, x0 = _next_non_masked_element(x, nearest_idx)
    _, x1 = _next_non_masked_element(x, next_idx)

    # y-values around each intersection for the first line
    _, a0 = _next_non_masked_element(a, nearest_idx)
    _, a1 = _next_non_masked_element(a, next_idx)

    # y-values around each intersection for the second line
    _, b0 = _next_non_masked_element(b, nearest_idx)
    _, b1 = _next_non_masked_element(b, next_idx)

    # Calculate the x-intersection.
    delta_y0 = a0 - b0
    delta_y1 = a1 - b1
    intersect_x = (delta_y1 * x0 - delta_y0 * x1) / (delta_y1 - delta_y0)

    # Calculate the y-intersection of the lines.
    intersect_y = ((intersect_x - x0) / (x1 - x0)) * (a1 - a0) + a0

    # Make a mask based on the direction of sign change desired
    if direction == 'increasing':
        mask = sign_change > 0
    elif direction == 'decreasing':
        mask = sign_change < 0
    elif direction == 'all':
        return intersect_x, intersect_y
    else:
        raise ValueError(
            'Unknown option for direction: {0}'.format(str(direction)))
    return intersect_x[mask], intersect_y[mask]


def nearest_intersection_idx(a, b):
    """Determine the index of the point just before two lines with common x values.

    Parameters
    ----------
    a : array-like
        1-dimensional array of y-values for line 1
    b : array-like
        1-dimensional array of y-values for line 2

    Returns
    -------
        An array of indexes representing the index of the values
        just before the intersection(s) of the two lines.
    """
    # Determine the points just before the intersection of the lines
    sign_change_idx, = np.nonzero(np.diff(np.sign(a - b)))

    return sign_change_idx


def _next_non_masked_element(x, idx):
    """Return the next non-masked element of a masked array.

    If an array is masked, return the next non-masked element (if the given index is masked).
    If no other unmasked points are after the given masked point, returns none.

    Parameters
    ----------
    x : array-like
        1-dimensional array of numeric values
    idx : integer
        index of requested element

    Returns
    -------
        Index of next non-masked element and next non-masked element
    """
    try:
        next_idx = idx + x[idx:].mask.argmin()
        if np.ma.is_masked(x[next_idx]):
            return None, None
        else:
            return next_idx, x[next_idx]
    except (AttributeError, TypeError, IndexError):
        return idx, x[idx]


def mean_confidence_interval(data, confidence=0.95, axis=0):
    a = np.asanyarray(data)
    n = a.shape[axis]

    m, se = np.nanmean(a, axis=axis), stats.sem(a, axis=axis)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def transform_spectra(s):
    st = np.reshape(s, (-1, s.shape[-1]))

    return mean_confidence_interval(st, confidence=0.95, axis=0)


def broadcast_indices(indices, shape, axis):
    """Calculate index values to properly broadcast index array within data array.
    The purpose of this function is work around the challenges trying to work with arrays of
    indices that need to be "broadcast" against full slices for other dimensions.
    [Taken from Metpy v.1.4]

    """
    ret = []
    ndim = len(shape)
    for dim in range(ndim):
        if dim == axis:
            ret.append(indices)
        else:
            broadcast_slice = [np.newaxis] * ndim
            broadcast_slice[dim] = slice(None)
            dim_ind = np.arange(shape[dim])
            ret.append(dim_ind[tuple(broadcast_slice)])
    return tuple(ret)


def gradient_1d(scalar, x, axis=-1, order=6):
    """
    Computes gradient of a scalar function d(scalar)/dx along a given axis.

    Parameters
    ----------
    scalar: np.ndarray,
            An N-dimensional array containing samples of a scalar function.
    x: array_like
        Coordinate of the values along dimension 'axis'

    axis: int, optional, default axis=-1
        Gradient is calculated only along the given axis.

    order: int, default order=6 (higher combined accuracy)
        Determines the order of the formal truncation error. Available options are 2-8.

    Returns
    -------
        gradient : ndarray,
            Gradient of the 'scalar' along 'axis'.
    """

    msg = "Coordinate 'x' must be the same size as 'scalar' along the specified axis."
    assert x.size == scalar.shape[axis], msg

    # determine if the grid is regular
    dx = np.diff(x)

    is_regular = np.allclose(np.max(dx), np.min(dx), atol=1e-12)

    if is_regular:
        # Using high order schemes for regular grid, otherwise
        # using second-order accurate central finite differences
        scalar = np.moveaxis(scalar, axis, -1)
        scalar_shape = scalar.shape

        scalar = scalar.reshape((-1, scalar_shape[-1]))
        # compute gradient with a 6th-order compact finite difference scheme (Lele 1992),
        # and explicit 4th-order scheme at the boundaries.
        scalar_grad = numeric_tools.gradient(scalar, dx[0], order=order)
        scalar_grad = np.moveaxis(scalar_grad.reshape(scalar_shape), -1, axis)
    else:
        # Using numpy implementation of second-order finite differences for irregular grids
        scalar_grad = np.gradient(scalar, x, axis=axis, edge_order=2)

    return scalar_grad


def interpolate_1d(x, xp, *args, axis=0, fill_value=np.nan, scale='log'):
    r"""Interpolates data with any shape over a specified axis.
    Interpolation over a specified axis for arrays of any shape.

    Modified from the nicely vectorized version in metpy v.1.4

    Parameters
    ----------
    x : array-like
        1-D array of desired interpolated values.
    xp : array-like
        The x-coordinates of the data points.
    args : array-like
        The data to be interpolated. Can be multiple arguments, all must be the same shape as
        xp.
    axis : int, optional
        The axis to interpolate over. Defaults to 0.
    fill_value: float, optional
        Specify handling of interpolation points out of data bounds. If None, will return
        ValueError if points are out of bounds. Defaults to nan.
    scale: str, optional
        Interpolate in logarithmic ('log') or linear space
    Returns
    -------
    array-like
        Interpolated values for each point with coordinates sorted in ascending order.
    """

    if scale == 'log':
        x = np.log(x)
        xp = np.log(xp)

    # Make x an array
    x = np.asanyarray(x).reshape(-1)

    # Sort input data
    sort_args = np.argsort(xp, axis=axis)
    sort_x = np.argsort(x)

    # The shape after all arrays are broadcast to each other
    # Can't use broadcast_shapes until numpy >=1.20 is our minimum
    final_shape = np.broadcast(xp, *args).shape

    # indices for sorting
    sorter = broadcast_indices(sort_args, final_shape, axis)

    # sort xp -- need to make sure it's been manually broadcast due to our use of indices
    # along all axes.
    xp = np.broadcast_to(xp, final_shape)
    xp = xp[sorter]

    # Ensure source arrays are also in sorted order
    variables = [arr[sorter] for arr in args]

    # Make x broadcast with xp
    x_array = x[sort_x]
    expand = [np.newaxis] * len(final_shape)
    expand[axis] = slice(None)
    x_array = x_array[tuple(expand)]

    # Calculate value above interpolated value
    min_value = np.apply_along_axis(np.searchsorted, axis, xp, x[sort_x])
    min_value2 = np.copy(min_value)

    # If fill_value is none and data is out of bounds, raise value error
    if ((np.max(min_value) == xp.shape[axis]) or (np.min(min_value) == 0)) and fill_value is None:
        raise ValueError('Interpolation point out of data bounds encountered')

    # Warn if interpolated values are outside data bounds, will make these the values
    # at end of data range.
    if np.max(min_value) == xp.shape[axis]:
        min_value2[min_value == xp.shape[axis]] = xp.shape[axis] - 1
    if np.min(min_value) == 0:
        min_value2[min_value == 0] = 1

    # Get indices for broadcasting arrays
    above = broadcast_indices(min_value2, final_shape, axis)
    below = broadcast_indices(min_value2 - 1, final_shape, axis)

    # Calculate interpolation for each variable
    for var in variables:

        increment = (x_array - xp[below]) / (xp[above] - xp[below])
        var_interp = var[below] + (var[above] - var[below]) * increment

        # Set points out of bounds to fill value.
        var_interp[min_value == xp.shape[axis]] = fill_value
        var_interp[x_array < xp[below]] = fill_value

        # if fill_value is nan or "masked" return masked arrays
        var_interp = np.ma.masked_invalid(var_interp, copy=True)

        # Check for input points in decreasing order and return output to match.
        if x[0] > x[-1]:
            var_interp = np.swapaxes(np.swapaxes(var_interp, 0, axis)[::-1], 0, axis)

        yield var_interp


if __name__ == '__main__':
    pass
