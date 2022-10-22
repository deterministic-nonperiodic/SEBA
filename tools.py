import functools
import multiprocessing as mp

import numpy as np
import scipy.signal as sig
import scipy.special as spec
from scipy.spatial import cKDTree

import constants as cn

earth_radius_km = 1.0e-3 * cn.earth_radius  # km

DATA_PATH = '/media/yanm/Resources/DYAMOND/spectra/'


def kappa_from_deg(ls):
    """
        Returns total horizontal wavenumber [radians / meter]
        from spherical harmonics degree (ls) on the surface
        of a sphere of radius Re using the Jeans formula.
        κ = sqrt[l(l + 1)] / Re
    """
    return np.sqrt(ls * (ls + 1.0)) / cn.earth_radius


def lambda_from_deg(ls):
    """
    Returns wavelength λ [meters] from total horizontal wavenumber
    λ = 2π / κ
    """
    return 2.0 * np.pi / kappa_from_deg(ls)


def deg_from_lambda(lb):
    """
        Returns wavelength from spherical harmonics degree (ls)
    """
    return np.floor(np.sqrt(0.25 + (2.0 * np.pi * cn.earth_radius / lb) ** 2) - 0.5).astype(int)


def kappa_from_lambda(lb):
    return 1.0 / lb


def unpack_2Dto1D(data_2d, ntrunc):
    """Helper function for reshaping spherepack spectra"""
    ncs = int((ntrunc + 1) * (ntrunc + 2) / 2)
    data_1d = np.empty((ncs,) + data_2d.shape[2:])

    nmstrt = 0
    for m in range(ntrunc + 1):
        for n in range(m, ntrunc + 1):
            nm = nmstrt + n - m
            data_1d[nm] = data_2d[m, n]
        nmstrt = nmstrt + ntrunc - m + 1
    return data_1d


def unpack_1Dto2D(data_1d):
    """Helper function for reshaping spherepack spectra"""

    ntrunc = -1.5 + 0.5 * np.sqrt(9. - 8. * (1. - float(data_1d.shape[0])))

    data_2d = np.empty((2, ntrunc + 1, ntrunc + 1) + data_1d.shape[1:])
    nmstrt = 0
    for m in range(ntrunc + 1):
        for n in range(m, ntrunc + 1):
            nm = nmstrt + n - m + 1
            data_2d[m, n] = data_1d[nm]
        nmstrt = nmstrt + ntrunc - m + 1
    return data_2d


def transform_data(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        # reshape input data
        nargs = []
        for arg in args:
            ashape = arg.shape
            nargs.append(np.moveaxis(arg, 0, -1).reshape(ashape[:2] + (-1)))
        value = func(*nargs, **kwargs)
        # back to original shape
        return np.moveaxis(value.reshape(value.shape[0]), -1, 0)

    return wrapper_decorator


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


def lp_lanczos(data, nw, fc, axis=None, jobs=None):
    # Grid definition according to the number of weights

    if axis is None:
        axis = -1

    arr = np.moveaxis(data, axis, 0)

    if jobs is None:
        jobs = min(mp.cpu_count(), arr.shape[0])

    # compute lanczos 2D window for convolution
    coeffs = window_2d(fc, nw)

    # wrapper of convolution function for parallel computations
    convolve2d = functools.partial(sig.convolve2d, in2=coeffs, boundary='wrap', mode='same')

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
    icoords, _ = find_intersections(coords, a, b, direction=direction)

    if len(icoords) == 0:
        # print('No intersections found in data')
        return np.nan
    else:
        return icoords


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
        resolution = 0.5 * lambda_from_deg(nlon)  # zonal grid spacing at the Equator
        cutoff_scale = lambda_from_deg(np.array([40, 40]))

        fc = resolution / cutoff_scale  # cut-off at wavenumber 40 (~500 km) from A&L (2013)

        # Apply low-pass Lanczos filter for smoothing:
        beta = lp_lanczos(beta, [5, 5], fc, axis=-1, jobs=jobs)

    return beta.clip(0.0, 1.0)


def _getvrtdiv(args, func, ntrunc):
    """
    Compute the vertical component of vorticity and the horizontal
    divergence of a vector field with components ugrid and vgrid on the sphere.
    """
    ugrid, vgrid = args
    return func(ugrid, vgrid, ntrunc=ntrunc)
