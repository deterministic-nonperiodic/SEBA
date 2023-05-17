import functools

import numpy as np
import shtns

from constants import earth_radius
from tools import broadcast_1dto

# private variables for class Spharmt
_private_vars = ['nlon', 'nlat', 'gridtype', 'rsphere']


def iterate_shts(func, axis=-1):
    """
    Decorator for handling arrays' IO dimensions for calling SHTns' spectral functions.
    This wrapper function for running _shtns functions along sample dimension. Input arguments
    can be in both spectral and grid-point space.

    Parameters:
    -----------
    func: decorated function
    axis: axis
    """

    @functools.wraps(func)
    def iterated_sht(*args, **kwargs):
        # self passed as first argument
        cls, *data = args

        # Compact args to a single array (input arrays must be broadcastable)
        # Infer mask when passing masked arrays and fill with zeros preserving dtype.
        data = np.ma.fix_invalid(data).filled(fill_value=0.0)

        # check data type (real objects must be of type float64)
        if np.isrealobj(data):
            data = data.astype(np.float64)

        # expand sample dimension if needed
        if data.shape[axis] in (cls.nlon, cls.nlm):
            data = np.expand_dims(data, axis)

        # Apply SHTns' functions along the sample dimension (test parallel loop)
        slices = np.split(data, data.shape[axis], axis=axis)
        data = np.stack([func(cls, *slice_[..., 0], **kwargs) for slice_ in slices], axis=axis)

        return data.squeeze() if data.shape[axis] == 1 else data

    return iterated_sht


class Spharmt(object):
    """
         wrapper class for commonly used spectral transform operators in
         atmospheric models. Provides an interface to shtns compatible
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

    def __init__(self, nlat, nlon, rsphere=earth_radius, gridtype='gaussian', ntrunc=None):
        """initialize
        :param nlon:  number of longitude points
        :param nlat:  number of latitude points

        :param ntrunc: int, optional, default None
            Triangular truncation for the spherical harmonic transforms. If truncation is not
            specified then 'truncation=nlat-1' is used.
        :param gridtype: grid type
        """
        # checking parameters
        if rsphere is None:
            self.rsphere = earth_radius
        elif type(rsphere) == int or type(rsphere) == float:
            self.rsphere = abs(rsphere)
        else:
            raise ValueError(f'Illegal value of rsphere {rsphere} - must be positive')

        if nlon > 3:
            self.nlon = nlon
        else:
            raise ValueError(f'Illegal value of nlon {nlon} - must be at least 4')

        if nlat > 2:
            self.nlat = nlat
        else:
            raise ValueError(f'Illegal value of nlat {nlat} - must be at least 3')

        if ntrunc is None:
            ntrunc = nlat - 1
        else:
            ntrunc = int(ntrunc)
            assert 0 < ntrunc <= nlat - 1, ValueError(
                f'Truncation must be between 0 and {nlat - 1}')

        if gridtype.lower() not in ('regular', 'gaussian'):
            raise ValueError('Illegal value of gridtype {} - must be'
                             'either "gaussian" or "regular"'.format(gridtype))
        else:
            self.gridtype = gridtype.lower()

        # Initialize 4pi-normalized harmonics with no CS phase shift (consistent with SPHEREPACK)
        # No normalization needed to recover the global mean/covariance from the power spectrum.
        # Alternatives are: sht_orthonormal or sht_schmidt.
        self.sht = shtns.sht(ntrunc, ntrunc, 1, shtns.sht_fourpi + shtns.SHT_NO_CS_PHASE)

        if self.gridtype == 'regular':
            self.sht.set_grid(nlat, nlon, shtns.sht_reg_dct | shtns.SHT_PHI_CONTIGUOUS, 0)
            self.weights = self.sht.cos_theta
        else:
            # default to gaussian grid
            self.sht.set_grid(nlat, nlon, shtns.sht_quick_init | shtns.SHT_PHI_CONTIGUOUS, 0)

            # create symmetrical weights from Legendre roots
            weights = self.sht.gauss_wts()
            self.weights = np.concatenate([weights, weights[::-1]])

        self.truncation = ntrunc
        self.nlm = self.sht.nlm
        self.degree = self.sht.l
        self.order = self.sht.m

        self.lap = - self.degree * (self.degree + 1.0) / self.rsphere
        self.inv_lap = np.insert(1.0 / self.lap[1:], 0, 0.0)

    # ----------------------------------------------------------------------------------------------
    # Wrappers for running SHTns functions along sample dimension (nlat, nlon, samples)
    # ----------------------------------------------------------------------------------------------
    @iterate_shts
    def synthesis(self, *args):
        return self.sht.synth(*args)

    @iterate_shts
    def analysis(self, *args):
        return self.sht.analys(*args)

    @iterate_shts
    def synth_grad(self, spec):
        return self.sht.synth_grad(spec)

    # ----------------------------------------------------------------------------------------------
    # Functions consistent with Spharm using SHTns backend for spherical harmonic transforms
    # ----------------------------------------------------------------------------------------------
    def laplacian(self, spec):
        """compute Laplacian in spectral space"""
        return broadcast_1dto(self.lap, spec.shape) * spec

    def inverse_laplacian(self, spec):
        """Invert Laplacian in spectral space"""
        return broadcast_1dto(self.inv_lap, spec.shape) * spec

    def grdtospec(self, scalar):
        """compute spectral coefficients from gridded data"""
        return self.analysis(scalar)

    def spectogrd(self, scalar_spec):
        """compute gridded data from spectral coefficients"""
        return self.synthesis(scalar_spec)

    def getpsichi(self, u, v):
        """compute streamfunction and velocity potential from horizontal wind"""
        psi_spec, chi_spec = self.analysis(u, v)

        psi_grid = self.rsphere * self.synthesis(psi_spec)
        chi_grid = self.rsphere * self.synthesis(chi_spec)

        return psi_grid, chi_grid

    def getvrtdivspec(self, u, v):
        """compute spectral coeffs of vorticity and divergence from wind vector"""
        psi_spec, chi_spec = self.analysis(u, v)

        return self.laplacian(psi_spec), self.laplacian(chi_spec)

    def getuv(self, vrt_spec, div_spec):
        """compute wind vector from spectral coeffs of vorticity and divergence"""

        psi_spec = self.inverse_laplacian(vrt_spec)
        chi_spec = self.inverse_laplacian(div_spec)

        return self.synthesis(psi_spec, chi_spec)

    def getgrad(self, scalar):
        """ Compute the zonal and meridional components of the gradient
            of a scalar function on the sphere.
        """

        # compute spectral coefficients
        scalar_spec = self.analysis(scalar)

        # compute horizontal gradient of a scalar from spectral coefficients
        v, u = self.synth_grad(scalar_spec)

        return u / self.rsphere, - v / self.rsphere

    def getuv_from_stream(self, psi_spec):
        """Compute wind vector from spectral coefficients of stream function"""
        u, v = self.synth_grad(psi_spec)

        return u / self.rsphere, v / self.rsphere
