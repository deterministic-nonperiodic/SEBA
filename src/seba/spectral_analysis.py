"""
    **Description**
    A collection of functions to compute either the cross-power spectrum, cross-energy
    spectrum, or l2-cross-norm spectrum. The total cross-power is defined as the
    integral of the clm1 times the conjugate of clm2 over all space, divided
    by the area the functions span. If the mean of the functions is zero,
    this is equivalent to the covariance of the two functions. The total
    cross-energy is the integral of clm1 times the conjugate of clm2 over all
    space and is 4pi times the total power. The l2-cross-norm is the
    sum of clm1 times the conjugate of clm2 over all angular orders as a
    function of spherical harmonic degree.

    The output spectrum can be expressed using one of three units. 'per_l'
    returns the contribution to the total spectrum from all angular orders
    at degree l. 'per_lm' returns the average contribution to the total
    spectrum from a single coefficient at degree l, and is equal to the
    'per_l' spectrum divided by (2l+1).
"""
import numpy as np
from . import numeric_tools
from . import constants as cn


def kappa_from_deg(ls, linear=False):
    """
        This function returns total horizontal wavenumber [radians / meter]
        from spherical harmonics degree (ls) on the surface
        of a sphere of radius Re using the Jean's formula.
        κ = sqrt[l(l + 1)] / Re ~ l / Re, for l>>1
    """
    ls = np.asarray(ls)

    num = ls if linear else np.sqrt(ls * (ls + 1.0))
    return num / cn.earth_radius


def lambda_from_deg(ls, linear=False):
    """
    Returns wavelength λ [meters] from total horizontal wavenumber
    λ = 2π / κ
    """
    return 2.0 * np.pi / kappa_from_deg(ls, linear=linear)


def deg_from_lambda(lb):
    """
        Returns wavelength from spherical harmonics degree (ls)
    """
    lb = np.asarray(lb)

    deg = np.sqrt(0.25 + (2.0 * np.pi * cn.earth_radius / lb) ** 2)
    return np.floor(deg - 0.5).astype(int)


def kappa_from_lambda(lb):
    return 2.0 * np.pi / np.asarray(lb)


def triangular_truncation(nspc):
    # Computes the triangular truncation from the number of spectral coefficients 'nspc'.
    # Solves (ntrunc + 1)(ntrunc + 2)/2 - nspc = 0, to get the original grid dimensions.
    # If no truncation was applied to compute the spectral coefficients, then ntrunc
    # corresponds to the number of latitude points in the original grid nlat - 1.
    return int(-1.5 + 0.5 * np.sqrt(9. - 8. * (1. - float(nspc))))


def cross_spectrum(clm1, clm2=None, lmax=None, convention='power', axis=0):
    """Returns the cross-spectrum of the spherical harmonic coefficients as a
    function of spherical harmonic degree.

    Signature
    ---------
    array = cross_spectrum(clm1, clm2, [degrees, lmax, convention, axis])

    Parameters
    ----------
    clm1 : ndarray, shape ((ntrunc+1)*(ntrunc+2)/2, ...)
        Contains the first set of spherical harmonic coefficients.
    clm2 : ndarray, shape ((ntrunc+1)*(ntrunc+2)/2, ...), optional.
        Contains the second set of spherical harmonic coefficients.
    convention : str, optional, default = 'power'
        The type of spectrum to return: 'power' for power spectrum, 'energy'
        for energy spectrum, and 'l2norm' for the l2-norm spectrum.
    lmax : int, optional,
        triangular truncation
    axis : int, optional
        axis of the spectral coefficients

    Returns
    -------
    array : ndarray, shape (lmax + 1, ...)
        contains the cross-spectrum as a function of spherical harmonic degree.
    """
    if convention not in {'energy', 'power'}:
        raise ValueError(f"Parameter 'convention' must be 'energy' or 'power'. Got '{convention}'.")

    if clm2 is not None and clm1.shape != clm2.shape:
        raise ValueError(
            f"'clm1' and 'clm2' must have the same shape. Got {clm1.shape} and {clm2.shape}.")

    # spectral coefficients moved to axis 0 for clean vectorized operations
    clm_shape = list(clm1.shape)
    nlm = clm_shape.pop(axis)

    # flatten sample dimensions
    clm1 = np.moveaxis(clm1, axis, 0).reshape((nlm, -1))

    # Get indexes of the triangular matrix with spectral coefficients
    truncation = numeric_tools.truncation(nlm)

    if lmax is not None:
        # If lmax is given, make sure it is consistent with the number of clm coefficients
        truncation = min(lmax + 1, truncation)

    # Compute cross-spectrum from spherical harmonic expansion coefficients as
    # a function of spherical harmonic degree (total wavenumber)
    if clm2 is None:
        spectrum = numeric_tools.cross_spectrum(clm1, clm1, truncation)
    else:
        clm2 = np.moveaxis(clm2, axis, 0).reshape((nlm, -1))
        spectrum = numeric_tools.cross_spectrum(clm1, clm2, truncation)

    if convention.lower() == 'energy':
        spectrum *= 4.0 * np.pi

    return np.moveaxis(spectrum.reshape(tuple([truncation] + clm_shape)), 0, axis)
