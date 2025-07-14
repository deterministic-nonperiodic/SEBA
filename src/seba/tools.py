import functools
import time
from functools import reduce
from math import gcd

import numpy as np
import scipy.special as spec
import shtns
from _shtns import sht_idx
from joblib import Parallel, delayed, cpu_count
from scipy.linalg import inv
from scipy.signal import fftconvolve, convolve
from scipy.spatial import KDTree
import scipy.stats as stats

from . import numeric_tools
from .spectral_analysis import lambda_from_deg


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
    pdata shape is (73, 144, 24, 17)
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

    data = data.reshape(inter_shape[:2] + (-1, inter_shape[-1]))

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


def get_num_cores():
    """Returns number of physical CPU cores"""
    return np.ceil(0.5 * cpu_count()).astype(int)


def get_chunk_size(n_workers, len_iterable, factor=4):
    """Calculate chunk size argument for Pool methods.

    Resembles source-code within `multiprocessing.pool.Pool._map_async`.
    """
    chunk_size, extra = divmod(len_iterable, n_workers * factor)
    if extra:
        chunk_size += 1
    return chunk_size


def get_number_chunks(sample_size, workers, factor=4):
    """
        Calculate the number of chunks for Pool methods.
    """
    n_chunks = sample_size / get_chunk_size(workers, sample_size, factor=factor)

    return np.ceil(n_chunks).astype(int)


def broadcast_1dto(arr, shape):
    """
    Broadcast a 1-dimensional array to a given shape using numpy rules
    by appending dummy dimensions. Raises error if the array cannot be
    broadcast to the target shape.
    """
    a_size = arr.size
    if a_size in shape:
        # finds corresponding dimension from left to right.
        # if shape contains multiple dimensions with the same size,
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
    """Rotate vector components"""
    vector = np.take(vector, [0, 1], axis=axis)
    # return vector rotated 90 degrees counterclockwise
    return np.ma.stack([-vector[1], vector[0]])


def getspecindx(ntrunc):
    """
     compute indices of zonal wavenumber (index_m) and degree (index_n)
     for complex spherical harmonic coefficients.
     @param ntrunc: spherical harmonic triangular truncation limit.
     @return: C{B{index_m, index_n}} - rank 1 numpy Int32 arrays
     containing zonal wavenumber (index_m) and degree (index_n) of
     spherical harmonic coefficients with size (ntrunc+1)*(ntrunc+2)/2.
    """
    index_m, index_n = np.indices((ntrunc + 1, ntrunc + 1))

    indices = np.nonzero(np.greater(index_n, index_m - 1).flatten())
    index_n = np.take(index_n.flatten(), indices)
    index_m = np.take(index_m.flatten(), indices)

    return np.squeeze(index_m), np.squeeze(index_n)


def _pack_levels(data, nlat, nlm, order='C'):
    # pack dimensions of arrays (nlat, nlon, ...) to (nlat, nlon, samples)
    data_shape = np.shape(data)

    data_length = data_shape[0]
    expected_length = [2, nlat, nlm]

    if data_length not in expected_length:
        raise ValueError(f"Inconsistent array shape: expecting "
                         f"first dimension of size {nlat} or {nlm}.")

    data_shape = data_shape[:3 - expected_length.index(data_length)] + (-1,)

    return np.reshape(data, data_shape, order=order)  # .squeeze()


def _unpack_levels(data, samples, nlevels, order='C'):
    # unpack dimensions of arrays (nlat, nlon, samples)
    if np.shape(data)[-1] == samples * nlevels:
        data_shape = np.shape(data)[:-1] + (samples, nlevels)
        return np.reshape(data, data_shape, order=order)
    else:
        return data


def transform_io(func):
    """
    Decorator for handling arrays' IO dimensions for calling SHTns' spectral functions.
    The dimensions of the input arrays with shapes (nlat, nlon, nlev, ntime, ...)
    or (ncoeffs, nlev, ntime, ...) are packed to (nlat, nlon, samples) and (ncoeffs, samples)
    respectively, where ncoeffs = (ntrunc+1)*(ntrunc+2)/2. Finally, the outputs are transformed
    back to the original shape where needed.

    Parameters:
    -----------
    func: decorated function
    """

    @functools.wraps(func)
    def dimension_packer(*args, **kwargs):
        # self passed as the first argument
        cls, *args = args

        # Avoid class instance attributes lookup in loop
        nlat = cls.nlat
        nlm = cls.sphere.nlm

        transformed_args = [cls, ] + [_pack_levels(arg, nlat, nlm) for arg in args]

        # convert output back to the original shape
        return _unpack_levels(func(*transformed_args, **kwargs), cls.samples, cls.nlevels)

    return dimension_packer


def convert_longitude(longitude):
    """
    Converts a longitude array from the range (-180, 180) to the range (0, 360).

    Args:
    - longitude: array representing the longitude values in degrees

    Returns:
    - array representing the converted longitude values in degrees
    """
    if np.any(longitude > 180.):
        # Longitudes are within (0, 360), need converting to (-180, 180)
        return np.where(longitude > 180, longitude - 360, longitude)
    else:
        # Converting to (0, 360)
        return np.where(longitude < 0, longitude + 360, longitude)


def regular_longitudes(nlon):
    """
     Creates longitudes corresponding a regular grid with increment 2*pi / nlon radians.
    """
    assert nlon > 3, f"Wrong value for 'nlon' {nlon} - must be at least 4."

    return np.linspace(0.0, 360., nlon, endpoint=False)


def regular_lats_wts(nlat):
    """
    Computes the latitude points and weights of a regular grid
    (equally spaced in longitude and latitude). Regular grids
    will include the poles and the Equator if nlat is odd. The sampling
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
    latitudes = np.rad2deg(np.arcsin(nodes[::-1]))

    return latitudes, weights


def bound_array(a, bounds=None):
    # check the length of the input array and set value range
    _range = (np.min(a), np.max(a)) if len(a) > 0 else (0, 0)
    # handle default bounds
    bounds = bounds or _range

    # replace None in bounds: works as np.clip
    bounds = [b if b is not None else r for b, r in zip(bounds, _range)]

    # return values in array within the given bounds
    return a[(a >= bounds[0]) & (a <= bounds[1])]


def create_grid(nlat, grid_type='regular', bounds_box=None):
    """
    Creates coordinates for a regular or gaussian grid
    with 'nlat' latitudes points and '2*nlat' longitude points.
    """

    if grid_type not in ['regular', 'gaussian']:
        raise ValueError("Wrong specification of grid type."
                         "Must be 'regular' or 'gaussian'.")

    if bounds_box is None:
        bounds_box = {"lat": None, "lon": None}

    # generate array with longitudes
    lons = regular_longitudes(2 * nlat)

    # generate array with latitudes depending on grid type
    if grid_type == 'regular':
        lats, _ = regular_lats_wts(nlat)
    else:
        lats, _ = gaussian_lats_wts(nlat)

    # bound arrays to a given range
    lats = bound_array(lats, bounds=bounds_box['lat'])
    lons = bound_array(lons, bounds=bounds_box['lon'])

    return lats, lons


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
            raise ValueError('Invalid equally-spaced latitudes (may be non-global)')
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


def lanczos_kernel(fc, n):
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
    kernel = lanczos_kernel(cutoff_freq, window_size)
    kernel = np.expand_dims(kernel, 0)

    # padding array using circular boundary conditions along the longitudinal dimension
    # and constant mirror reflection along latitudinal dimension with the kernel size.
    arr = np.pad(arr, ((0,), (window_size,), (0,)), mode='reflect')
    arr = np.pad(arr, ((0,), (0,), (window_size,)), mode='wrap')

    # wrapper of convolution function for parallel computations
    convolve_2d = functools.partial(fftconvolve, in2=kernel, mode='same', axes=(1, 2))

    # Chunks of arrays along axis=0 for the mp mapping ...
    n_chunks = get_number_chunks(arr.shape[0], jobs, factor=4)

    # Create pool of workers
    pool = Parallel(n_jobs=jobs, backend="threading")

    # applying lanczos filter in parallel
    result = pool(delayed(convolve_2d)(chunk) for chunk in np.array_split(arr, n_chunks))
    result = np.stack(result, axis=0)

    result[np.isnan(result)] = 1.0

    # remove added pad
    result = result[:, window_size:-window_size, window_size:-window_size]

    return np.moveaxis(result, 0, axis)


def is_sorted(arr, ascending=True):
    if ascending:
        return np.all(arr[:-1] <= arr[1:])
    else:
        return np.all(arr[:-1] >= arr[1:])


def lanczos_lowpass_1d(data, cutoff_frequency, sampling_frequency, window_length=None):
    """
    Apply a Lanczos low-pass filter to a time series.

    Parameters:
    - data: 1D numpy array of time series data.
    - cutoff_frequency: The cutoff frequency of the filter in Hz.
    - sampling_frequency: The sampling frequency of the data in Hz.
    - window_length: The length of the Lanczos window (number of points).
                     If None, it is calculated based on the cutoff frequency.

    Returns:
    - filtered_data: The filtered time series.
    """

    def lanczos_kernel(cutoff, window_size):
        # Compute the Lanczos kernel
        n = np.arange(1, window_size + 1)
        h = spec.sinc(2 * cutoff * n) * np.sin(np.pi * n / window_size) / (np.pi * n / window_size)
        kernel = np.concatenate(([2 * cutoff], h))
        kernel = np.concatenate((kernel[::-1], kernel[1:]))
        # Normalize the kernel to ensure the sum of the kernel coefficients equals one
        kernel /= np.sum(kernel)
        return kernel

    # Normalize the cutoff frequency by the Nyquist frequency
    nyquist_frequency = 0.5 * sampling_frequency
    normalized_cutoff = cutoff_frequency / nyquist_frequency

    # Calculate window length if not provided
    if window_length is None:
        window_length = int(4 / normalized_cutoff)  # Rule of thumb for window size

    # Get the Lanczos kernel
    kernel = lanczos_kernel(normalized_cutoff, window_length)

    # Handle edges by padding the data
    pad_length = len(kernel) // 2
    padded_data = np.pad(data, (pad_length, pad_length), mode='reflect')

    # Remove mean before filtering
    data_mean = np.mean(data)
    padded_data -= data_mean

    # Convolve the padded data with the kernel
    filtered_data = convolve(padded_data, kernel, mode='same')

    # Add the mean back to the filtered data
    filtered_data += data_mean

    # Remove the padding
    filtered_data = filtered_data[pad_length:-pad_length]

    return filtered_data


def search_nn_index(points, target_points):
    """Uses KDTree algorithm to search for the nearest neighbors in points
       to each value in target_points.
    """
    if target_points is None:
        return slice(None)

    points = np.atleast_2d(points)
    target_points = np.atleast_2d(target_points)
    # creates a search tree
    # noinspection PyArgumentList
    search_tree = KDTree(points)
    # nearest neighbour (k=1) in levels to each point in target levels
    _, nn_idx = search_tree.query(target_points, k=1)

    return nn_idx


def interpolate_nn_2d(x, y, data, xi, yi, axes=None):
    """Uses KDTree to perform nearest neighbor interpolation.
       Does essentially the same as scipy's NearestNDInterpolator but limited to 2D data.
    """
    if axes is None:
        axes = (0, 1)

    assert data.ndim >= 2, "Input array 'data' must be at least 2D."

    # move spatial axes to first dimensions
    data = np.moveaxis(data, axes, (0, 1))

    assert data.shape[0] == y.size, f"Coordinate 'y' must have the same size " \
                                    f"as data along dimension {axes[0]}"

    assert data.shape[1] == x.size, f"Coordinate 'x' must have the same size " \
                                    f"as data along dimension {axes[1]}"

    # initialize the output sape from target coordinates
    int_shape = (-1,) + data.shape[2:]
    out_shape = (yi.size, xi.size) + int_shape[1:]

    nodes = np.column_stack([mg.ravel() for mg in np.meshgrid(x, y)])
    points = np.column_stack([mg.ravel() for mg in np.meshgrid(xi, yi)])

    # finding indexes closest to each pair in target_points
    nn_indexes = search_nn_index(nodes, points)

    # get nearest neighbors from data
    result = data.reshape(int_shape)[nn_indexes].reshape(out_shape)
    result = np.moveaxis(result, (0, 1), axes)

    return result


def indices_to_3d(mask, size):
    """
    Converts a 2D array mask[i, j] containing indices onto a 3D mask with a new dimension of
    length 'size' where all indices k < mask[i, j] along the new dimension are 0 and 1 otherwise.
    Used to mask terrain in 3D field from a 2D map (surface pressure) intersecting a 1D
    coordinate (pressure profile).
    """

    shape = tuple(mask.shape) + (size,)

    mask_bc, result_bc = np.broadcast_arrays(mask[..., np.newaxis], np.zeros(shape))
    result_bc[mask_bc <= np.arange(size)] = 1.0

    return result_bc


def reduce_mask(mask, axis=-1):
    """
    Converts a 3D array mask[i, j, k] containing indices onto a 2D terrain mask.
    Reverts the result of 'indices_to_3d'.
    """
    return mask.shape[axis] - np.count_nonzero(mask, axis=axis)


def surface_mask(p, ps, smooth=True, jobs=None):
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


def linear_scaler(data, feature_range=(0, 1), axis=0):
    #
    data = np.moveaxis(data, axis, 0)

    d_min, d_max = data.min(axis=0), data.max(axis=0)

    scaled_data = (data - d_min) * (feature_range[1] - feature_range[0]) / (d_max - d_min)

    return feature_range[0] + np.moveaxis(scaled_data, 0, axis)


def compute_mode_coupling(mask, grid_type='gaussian', realizations=3):
    # Initialize the mode-coupling matrix

    nlat, nlon = mask.shape

    # infer lmax from mask size
    lmax = nlat - 1

    # Create the grid
    if grid_type not in ['gaussian', 'regular']:
        raise ValueError('Invalid grid_type parameter. Choose between "gaussian" and "regular".')

    # initialize spherical harmonics
    sht = shtns.sht(lmax, lmax, 1, shtns.sht_fourpi + shtns.SHT_NO_CS_PHASE)

    # set horizontal grid
    if grid_type == 'gaussian':
        sht.set_grid(nlat, nlon, shtns.sht_quick_init | shtns.SHT_PHI_CONTIGUOUS, 1e-12)
    else:
        sht.set_grid(nlat, nlon, shtns.sht_reg_dct | shtns.SHT_PHI_CONTIGUOUS, 1e-12)

    # initialize mode-coupling matrix (m,l)
    coupling = np.zeros((lmax + 1, lmax + 1))

    for ln in range(lmax + 1):

        # Initialize spherical harmonics coefficients
        clm = np.zeros(sht.nlm, dtype=np.complex128)

        # get spherical harmonic index and assign 1/2 (power spectrum equals 1)
        lm_index = sht_idx(sht, ln, ln)
        clm[lm_index] = 0.5

        # generate a masked map with pure tone at degree ln
        masked_map = mask * sht.synth(clm)

        clm_sum = 0.0
        for i in range(realizations):
            # create white noise within range (0.65, 1.2)
            noise = np.random.normal(0, 1, size=mask.size)
            noise = linear_scaler(noise, feature_range=(0.75, 1.0))

            # Compute observed spectra of masked noisy data
            clm = sht.analys(masked_map * noise.reshape(mask.shape))
            clm_sum += numeric_tools.cross_spectrum(clm, clm, lmax + 1)

        # Compute the average power spectrum and store the M(m, l) values
        coupling[ln] = np.squeeze(clm_sum) / realizations

    # Transpose to (l, m) and compute the inverse of the mode-coupling matrix.
    # This step may fail if M is singular!
    coupling = inv(coupling)

    return coupling


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


def broadcast_indices(indices, shape, axis):
    """Calculate index values to properly broadcast index array within data array.
    The purpose of this function is work around the challenges trying to work with arrays of
    indices that need to be "broadcast" against full slices for other dimensions.
    [From Metpy v.1.4]

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


def gradient_1d(scalar, x=None, axis=-1, order=6):
    """
    Computes the gradient of a scalar function d(scalar)/dx along a given axis. Uses high-order
    compact finite differences schemes if the input grid is regularly spaced, otherwise uses the
    second-order accurate scheme implemented in 'numpy.gradient'.

    Parameters
    ----------
    scalar: np.ndarray,
            An N-dimensional array containing samples of a scalar function.
    x: array_like
        Coordinate of the values along dimension 'axis'

    axis: int, optional, default axis=-1
        Gradient is calculated only along the given axis.

    order: int, default order=6 (recommended higher combined accuracy)
        Determines the order of the formal truncation error.
        Available options are 2 <= order <= 8.

    Returns
    -------
        gradient : ndarray,
            Gradient of the 'scalar' along 'axis'.
    """
    if x is None:
        x = np.arange(scalar.shape[axis])
    else:
        msg = "Coordinate 'x' must be the same size as 'scalar' along the specified axis."
        assert x.size == scalar.shape[axis], msg

    # determine if the grid is regular
    dx = np.diff(x)
    equally_spaced = np.allclose(np.max(dx), np.min(dx), atol=1e-12)

    if equally_spaced:
        # Using high-order schemes for regular grid, otherwise
        # using second-order accurate central finite differences
        scalar = np.moveaxis(scalar, axis, 0)
        scalar_shape = scalar.shape

        scalar = scalar.reshape((scalar_shape[0], -1))

        # check if array is masked and compute index above which to calculate gradients
        if np.ma.is_masked(scalar):
            scalar_mask = scalar.mask
            mask_index = reduce_mask((~scalar_mask).astype(int), axis=0) + 1
        else:
            scalar_mask = False
            mask_index = np.ones(scalar.shape[1:], dtype=int)

        # compute gradient with a 6th-order compact finite difference scheme (Lele 1992),
        # and explicit 4th-order scheme at the boundaries.
        scalar_grad = numeric_tools.gradient(scalar, dx[0], order=order, mask=mask_index)

        # apply mask back to array
        scalar_grad = np.ma.masked_array(scalar_grad, mask=scalar_mask, fill_value=0.0)
        scalar_grad = np.moveaxis(scalar_grad.reshape(scalar_shape), 0, axis)
    else:
        # Using numpy implementation of second-order finite differences for irregular grids
        scalar_grad = np.gradient(scalar, x, axis=axis, edge_order=2)

    return scalar_grad


def interpolate_1d(x, xp, *args, axis=0, fill_value=None, scale='log'):
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
    fill_value: float, str, optional
        Specify handling of interpolation points out of data bounds. If None, will return
        ValueError if points are out of bounds. Can be set to "extrapolate" with similar
        behaviour as scipy.interp1d.
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

    extrapolate = np.isscalar(fill_value) and fill_value == "extrapolate"

    # Calculate interpolation for each variable
    for var in variables:

        increment = (x_array - xp[below]) / (xp[above] - xp[below])
        var_interp = var[below] + (var[above] - var[below]) * increment

        # Set points out of bounds to fill value.
        if not extrapolate:
            var_interp[min_value == xp.shape[axis]] = fill_value
            var_interp[x_array < xp[below]] = fill_value

        # if fill_value is nan or "masked" return masked arrays
        var_interp = np.ma.masked_invalid(var_interp, copy=True)

        # Check for input points in decreasing order and return output to match.
        if x[0] > x[-1]:
            var_interp = np.swapaxes(np.swapaxes(var_interp, 0, axis)[::-1], 0, axis)

        yield var_interp


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
    if np.isscalar(a):
        a = np.full_like(b, a)
    elif np.isscalar(b):
        b = np.full_like(a, b)
    else:
        assert a.size == b.size, "Arrays 'a' and 'b' must be the same size"

    assert x.size == a.size, "Array 'coords' must be the same size as 'a' and 'b'"

    # Find the index of the points just before the intersection(s)
    nearest_idx = np.argwhere(np.diff(np.sign(a - b))).ravel()
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
        raise ValueError('Unknown option for direction: {0}'.format(str(direction)))

    # get intersections
    intersect_x, intersect_y = intersect_x[mask], intersect_y[mask]

    if len(intersect_x) == 0:
        intersect_x = intersect_y = np.array([np.nan, ])

    return intersect_x, intersect_y


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


def nyquist_frequency(time_points):
    """
    Compute the Nyquist frequency given a sequence of time points.

    Parameters:
    - time_points (array-like): Array of time points.

    Returns:
    - FNy (float): Nyquist frequency.
    """

    # remove duplicates and sort time steps
    time_points = np.unique(np.sort(time_points))

    # Calculate the differences between successive time points
    time_steps = np.diff(time_points)

    # Find the greatest common divisor (GCD) of the time differences
    # Convert diffs to integers by finding the smallest difference
    min_step = np.min(time_steps)
    scaled_steps = np.round(time_steps / min_step).astype(int)

    # Compute p from the scaled p
    p = min_step * reduce(gcd, scaled_steps)

    # Compute the Nyquist frequency
    nyquist_freq = 0.5 / min(p, min_step)

    return nyquist_freq


def mean_confidence_interval(data, confidence=0.95, axis=0):
    a = np.asanyarray(data)

    m, se = np.nanmean(a, axis=axis), stats.sem(a, axis=axis, nan_policy='omit')
    h = se * stats.t.ppf((1 + confidence) / 2., a.shape[axis] - 1)
    return m, h


def ewm_confidence_interval(values, errors=None, confidence=0.95, axis=0):
    """
    Calculate the error-weighted mean and its confidence interval.

    Parameters:
        values (array-like): Measured values.
        errors (array-like, optional): Associated errors for each value.
        confidence (float): Confidence level (default is 0.95 for 95% confidence).
        axis (int): Axis along which to compute the statistics (default is 0).

    Returns:
        mean (np.ndarray): Error-weighted mean.
        confidence_interval (np.ndarray): Confidence interval of the mean.
    """
    # Ensure inputs are numpy arrays
    values = np.array(values)

    if errors is not None:
        errors = np.array(errors)
        valid = (errors > 0) & ~np.isnan(errors)

        # Compute weights (zero for invalid entries)
        weights = np.zeros_like(values)
        weights[valid] = 1 / (errors[valid] ** 2)

        # Weighted mean (replaces np.average)
        weighted_mean = np.nansum(weights * values, axis=axis) / np.nansum(weights, axis=axis)

        # Weighted standard error
        weighted_std_error = np.sqrt(1 / np.nansum(weights, axis=axis))
    else:
        # Equal weights fallback
        weights = np.ones_like(values)

        weighted_mean = np.nanmean(values, axis=axis)
        weighted_std_error = stats.sem(values, axis=axis, nan_policy='omit')

    # Handle degrees of freedom
    dof = np.nansum(weights, axis=axis) ** 2 / np.nansum(weights ** 2, axis=axis)
    dof = np.where(dof > 1, dof - 1, np.nan)  # Avoid zero or negative dof

    # Confidence interval scaling factor (z-score or t-score)
    z_score = stats.t.ppf(0.5 + confidence / 2, dof)
    confidence_interval = z_score * weighted_std_error

    # Handle edge cases
    if np.isnan(weighted_mean).all():
        return np.nan, np.nan

    return weighted_mean, confidence_interval
