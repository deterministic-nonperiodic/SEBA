# from functools import partial
from multiprocessing import cpu_count

import numpy as np
import shtns
from joblib import Parallel, delayed, cpu_count
from tools import broadcast_1dto

# private variables for class Spharmt
_private_vars = ['nlon', 'nlat', 'gridtype', 'rsphere']


# def delayed(func):
#     def wrapper(*args, **kwargs):
#         return partial(func, *args, **kwargs)
#
#     return wrapper
#
#
# def call(f):
#     return f()
#
#
# class Parallel(object):
#     def __init__(self, n_jobs=1, chunksize=1, ordered=True, **kwargs):
#         self.n_jobs = n_jobs
#         self.chunksize = chunksize
#         self.ordered = ordered
#         self.kwargs = kwargs
#
#     def __call__(self, args):
#         n_jobs = get_num_cores() if self.n_jobs == -1 else self.n_jobs
#         if n_jobs == 1:
#             # sequential mode (useful for debugging)
#             yield from map(call, args)
#         else:
#             # spawn workers
#             pool = Pool(n_jobs, **self.kwargs)
#             try:
#                 if self.ordered:
#                     yield from pool.imap(call, args, chunksize=self.chunksize)
#                 else:
#                     yield from pool.imap_unordered(call, args, chunksize=self.chunksize)
#             except KeyboardInterrupt:
#                 pool.terminate()
#                 raise
#             except Exception:
#                 pool.terminate()
#                 raise
#             else:
#                 pool.close()
#             finally:
#                 pool.join()


class Spharmt(object):
    """
         wrapper class for commonly used spectral transform operations in
         atmospheric models.  Provides an interface to shtns compatible
         with pyspharm.
    """

    def __setattr__(self, key, val):
        """
        prevent modification of read-only instance variables.
        """
        if key in self.__dict__ and key in _private_vars:
            raise AttributeError('Attempt to rebind read-only instance variable ' + key)
        else:
            self.__dict__[key] = val

    def __delattr__(self, key):
        """
        prevent deletion of read-only instance variables.
        """
        if key in self.__dict__ and key in _private_vars:
            raise AttributeError('Attempt to unbind read-only instance variable ' + key)
        else:
            del self.__dict__[key]

    def __init__(self, nlon, nlat, rsphere=6.3712e6, gridtype='gaussian', ntrunc=None, jobs=None):
        """initialize
           nlon:  number of longitudes
           nlat:  number of latitudes
           ntrunc: triangular truncation
           gridtype: grid type
        """
        if jobs is None:
            self.jobs = 1
        elif jobs == -1:
            self.jobs = cpu_count()
        else:
            self.jobs = int(jobs)

        if ntrunc is None:
            ntrunc = nlat - 1

        # checking parameters
        if rsphere > 0.0:
            self.rsphere = rsphere
        else:
            raise ValueError('Illegal value of rsphere {} - must be positive'.format(rsphere))

        if nlon > 3:
            self.nlon = nlon
        else:
            raise ValueError('Illegal value of nlon {} - must be at least 4'.format(nlon))

        if nlat > 2:
            self.nlat = nlat
        else:
            raise ValueError('Illegal value of nlat {} - must be at least 3'.format(nlat))

        if gridtype not in ('regular', 'gaussian'):
            raise ValueError('Illegal value of gridtype {} - must be'
                             'either "gaussian" or "regular"'.format(gridtype))
        else:
            self.gridtype = gridtype

        # Initialize 4pi-normalized harmonics with no CS phase shift (consistent with SPHEREPACK)
        # Alternatives are: sht_orthonormal or sht_schmidt.
        self._shtns = shtns.sht(ntrunc, ntrunc, 1, shtns.sht_fourpi + shtns.SHT_NO_CS_PHASE)

        if self.gridtype == 'gaussian':
            self._shtns.set_grid(nlat, nlon, shtns.sht_quick_init | shtns.SHT_PHI_CONTIGUOUS, 1e-12)
        else:
            self._shtns.set_grid(nlat, nlon, shtns.sht_reg_dct | shtns.SHT_PHI_CONTIGUOUS, 1e-12)

        self.ntrunc = ntrunc
        self.nlm = self._shtns.nlm
        self.degree = self._shtns.l
        self.order = self._shtns.m

        self.kappa_sq = - self.degree * (self.degree + 1.0).astype(complex) / self.rsphere

    def _map(self, func, *args):
        """Wrapper function for running _shtns functions in parallel"""

        # Compact args to a single array (input arrays must be broadcastable)
        data = np.asarray(args)

        # check data type (real objects must be of type float64)
        if np.isrealobj(data):
            data = data.astype(np.float64)

        # add dummy sample dimension if necessary
        n_samples = data.shape[-1]

        if n_samples in (self.nlon, self.nlm):
            data = np.expand_dims(data, -1)
            n_samples = 1

        # loop along the sample dimension and apply func in parallel
        pool = Parallel(n_jobs=self.jobs, backend="threading")

        results = np.array(pool(delayed(func)(*data[..., i]) for i in range(n_samples)))

        if results.shape[0] == n_samples:
            results = np.moveaxis(results, 0, -1)

        return results

    def grdtospec(self, scalar):
        """compute spectral coefficients from gridded data"""
        return self._map(self._shtns.analys, scalar)

    def spectogrd(self, scalar_spec):
        """compute gridded data from spectral coefficients"""
        return self._map(self._shtns.synth, scalar_spec)

    def laplacian(self, spec):
        """compute Laplacian in spectral space"""
        kappa_sq = broadcast_1dto(self.kappa_sq, spec.shape)

        return kappa_sq * spec

    def invert_laplacian(self, spec):
        """Invert Laplacian in spectral space"""

        inv_kappa_sq = np.insert(1.0 / self.kappa_sq[1:], 0, 1.0)
        inv_kappa_sq = broadcast_1dto(inv_kappa_sq, spec.shape)

        return inv_kappa_sq * spec

    def getpsichi(self, u, v):
        """compute streamfunction and velocity potential from horizontal wind"""
        psi_spec, chi_spec = self._map(self._shtns.analys, u, v)

        psi_grid = self.rsphere * self.spectogrd(psi_spec)
        chi_grid = self.rsphere * self.spectogrd(chi_spec)

        return psi_grid, chi_grid

    def getvrtdivspec(self, u, v):
        """compute spectral coeffs of vorticity and divergence from wind vector"""
        psi_spec, chi_spec = self._map(self._shtns.analys, u, v)

        return self.laplacian(psi_spec), self.laplacian(chi_spec)

    def getuv(self, vrt_spec, div_spec):
        """compute wind vector from spectral coeffs of vorticity and divergence"""

        psi_spec = self.invert_laplacian(vrt_spec)
        chi_spec = self.invert_laplacian(div_spec)

        return self._map(self._shtns.synth, psi_spec, chi_spec)

    def getgrad(self, scalar):
        """Compute gradient vector of scalar function on the sphere"""

        # compute spectral coefficients
        scalar_spec = self.grdtospec(scalar)

        # compute horizontal gradient of a scalar from spectral coefficients
        v, u = self._map(self._shtns.synth_grad, scalar_spec)

        return u / self.rsphere, - v / self.rsphere

    def getgrad_2(self, scalar):
        """Compute gradient vector of scalar function on the sphere.
           slightly slower than 'getgrad' probably due to the initialization with 'np.zeros_like'
        """

        # compute spectral coefficients
        scalar_spec = self.grdtospec(scalar)

        # compute horizontal gradient of a scalar from spectral coefficients
        u, v = self._map(self._shtns.synth, np.zeros_like(scalar_spec), scalar_spec)

        return u / self.rsphere, v / self.rsphere
