import functools
import multiprocessing as mp

import numpy as np
import scipy.signal as sig
import scipy.special as spec
from scipy.spatial import cKDTree

from spectral_analysis import lambda_from_deg


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

    data = data.reshape(inter_shape[:2] + (-1, inter_shape[-1])).squeeze()

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


def get_chunk_size(n_workers, len_iterable, factor=4):
    """Calculate chunk size argument for Pool-methods.

    Resembles source-code within `multiprocessing.pool.Pool._map_async`.
    """
    chunk_size, extra = divmod(len_iterable, n_workers * factor)
    if extra:
        chunk_size += 1
    return chunk_size


def number_chunks(sample_size, workers):
    # finds the integer factor of 'sample_size' closest to 'workers'
    # for parallel computations: ensures maximum cpu usage for chunk_size = 1
    jobs = workers
    while sample_size % jobs:
        jobs -= 1
    return jobs if jobs != 1 else workers


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
    The dimensions of the input arrays with shapes (nlat, nlon, nlev, ntime, ...) or (ncoeffs, nlev, ntime, ...)
    are packed to (nlat, nlon, samples) and (ncoeffs, samples) respectively, where ncoeffs = (ntrunc+1)*(ntrunc+2)/2.
    Finally, the outputs are transformed back to the original shape where needed.

    Parameters:
    -----------
    func: decorated function
    order: {‘C’, ‘F’, ‘A’}, optional
        Reshape the elements of the input arrays using this index order.
        ‘C’ means to read / write the elements using C-like index order, with the last axis index changing fastest,
        back to the first axis index changing slowest. See 'numpy.reshape' for details.
    """

    @functools.wraps(func)
    def dimension_packer(*args, **kwargs):
        # self passed as first argument
        self, *_ = args
        transformed_args = [self, ]
        for arg in args:
            if isinstance(arg, np.ndarray):
                transformed_args.append(self._pack_levels(arg, order=order))

        results = func(*transformed_args, **kwargs)
        # convert output back to original shape
        return self._unpack_levels(results, order=order)

    return dimension_packer


def cumulative_flux(spectra):
    """
    Computes cumulative spectral energy transfer. The spectra are added starting
    from the largest wave number N (triangular truncation) to a given degree l.
    """
    spectral_flux = np.empty_like(spectra)

    for ln in range(spectra.shape[0]):
        spectral_flux[ln] = spectra[ln:].sum(axis=0)

    return spectral_flux


def window_2d(fc, n):
    n_x, n_y = n
    k_x, k_y = np.meshgrid(np.arange(-n[0], n[0] + 1), np.arange(-n[1], n[1] + 1))

    fc_xy = fc[0] * fc[1]

    # normalized wavenumbers:
    kx_n = k_x / n_x
    ky_n = k_y / n_y

    # Computation of the response weight on the grid
    z = np.sqrt((fc[0] * k_x) ** 2 + (fc[1] * k_y) ** 2)
    w_rect = fc_xy * spec.j1(2.0 * np.pi * z) / z.clip(min=1e-18)
    w = w_rect * spec.sinc(np.pi * kx_n) * spec.sinc(np.pi * ky_n)

    # Particular case where z=0
    w[:, n_x] = w_rect[:, n_x] * spec.sinc(np.pi * ky_n[:, n_x])
    w[n_y, :] = w_rect[n_y, :] * spec.sinc(np.pi * kx_n[n_y, :])
    w[n_y, n_x] = np.pi * fc_xy

    # Normalization of coefficients
    return w / np.nansum(w)


def convolve_chunk(a, func):
    #
    return np.array([func(ai) for ai in a])


def lowpass_lanczos(data, window_size, cutoff_freq, axis=None, jobs=None):
    if axis is None:
        axis = -1

    arr = np.moveaxis(data, axis, 0)

    if jobs is None:
        jobs = min(mp.cpu_count(), arr.shape[0])

    # compute lanczos 2D window for convolution
    coefficients = window_2d(cutoff_freq, window_size)

    # wrapper of convolution function for parallel computations
    convolve2d = functools.partial(sig.convolve2d, in2=coefficients, boundary='wrap', mode='same')

    # Chunks of arrays along axis=0 for the mp mapping ...
    chunks = np.array_split(arr, jobs, axis=0)

    # Create pool of workers
    pool = mp.Pool(processes=jobs)

    # Applying 2D lanczos filter to data chunks
    result = pool.map(functools.partial(convolve_chunk, func=convolve2d), chunks)

    # Freeing the workers:
    pool.close()
    pool.join()

    result = np.concatenate(result, axis=0)
    result[np.isnan(result)] = 1.0

    return np.moveaxis(result, 0, axis)


def intersections(coords, a, b, direction='all'):
    #
    index_coords, _ = find_intersections(coords, a, b, direction=direction)

    if len(index_coords) == 0:
        # print('No intersections found in data')
        return np.nan
    else:
        return index_coords


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


def terrain_mask(p, ps, smoothed=True, jobs=None):
    """
    Creates a terrain mask based on surface pressure and pressure profile
    :param: smoothed, optional
        Apply a low-pass filter to the terrain mask
    :return: 'np.array'
        beta contains 0 for levels satisfying p > ps and 1 otherwise
    """

    nlevels = p.size
    nlat, nlon = ps.shape

    # Search last level pierced by terrain for each vertical column
    level_m = p.size - np.searchsorted(np.sort(p), ps)
    # level_m = search_closet(p, ps)

    # create mask
    beta = np.zeros((nlat, nlon, nlevels))

    for ij in np.ndindex(*level_m.shape):
        beta[ij][level_m[ij]:] = 1.0

    if smoothed:
        # Calculate normalised cut-off frequencies for zonal and meridional directions:
        resolution = lambda_from_deg(nlon)  # zonal grid spacing at the Equator
        cutoff_scale = lambda_from_deg(np.array([128, 128]))  # wavenumber 40 (~500 km) from A&L (2013)

        # Normalized spatial cut-off frequency (cutoff_frequency / sampling_frequency)
        nsc_freq = resolution / cutoff_scale

        # Apply low-pass Lanczos filter for smoothing:
        beta = lowpass_lanczos(beta, [9, 9], nsc_freq, axis=-1, jobs=jobs)

    return beta.clip(0.0, 1.0)
