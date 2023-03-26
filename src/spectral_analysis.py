"""
    **Description**

    These functions return either the cross-power spectrum, cross-energy
    spectrum, or l2-cross-norm spectrum. Total cross-power is defined as the
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

import constants as cn


def kappa_from_deg(ls, linear=False):
    """
        Returns total horizontal wavenumber [radians / meter]
        from spherical harmonics degree (ls) on the surface
        of a sphere of radius Re using the Jeans formula.
        κ = sqrt[l(l + 1)] / Re ~ l / Re  for l>>1
    """
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
    deg = np.sqrt(0.25 + (2.0 * np.pi * cn.earth_radius / lb) ** 2)
    return np.floor(deg - 0.5).astype(int)


def kappa_from_lambda(lb):
    return 2.0 * np.pi / lb


def triangular_truncation(nspc):
    # Computes the triangular truncation from the number of spectral coefficients 'nspc'.
    # Solves (ntrunc + 1)(ntrunc + 2)/2 - nspc = 0, to obtain original grid dimensions.
    # If no truncation was applied to compute the spectral coefficients, then ntrunc
    # corresponds the number of latitude points in the original grid nlat - 1.
    return int(-1.5 + 0.5 * np.sqrt(9. - 8. * (1. - float(nspc))))


def cross_spectrum(clm1, clm2=None, normalization='4pi', degrees=None,
                   lmax=None, convention='power', unit='per_l'):
    """
    Returns the cross-spectrum of the spherical harmonic coefficients as a
    function of spherical harmonic degree.

    Signature
    ---------
    array = cross_spectrum(clm1, clm2, [normalization, degrees, lmax,
                                        convention, unit, base])

    Returns
    -------
    array : ndarray, shape (len(degrees))

    Parameters
    ----------
    clm1 : ndarray, shape (2, lmax + 1, lmax + 1, ...) ordered (l, m)
        ndarray containing the first set of spherical harmonic coefficients.
    clm2 : ndarray, shape (2, lmax + 1, lmax + 1, ...), optional
        ndarray containing the second set of spherical harmonic coefficients.
    normalization : str, optional, default = '4pi'
        '4pi', 'ortho', 'schmidt', or 'unnorm' for geodesy 4pi normalized,
        orthonormalized, Schmidt semi-normalized, or un-normalized coefficients,
        respectively.
    lmax : int, optional, default = len(clm[0,:,0]) - 1.
        Maximum spherical harmonic degree to output.
    degrees : ndarray, optional, default = numpy.arange(lmax+1)
        Array containing the spherical harmonic degrees where the spectrum
        is computed.
    convention : str, optional, default = 'power'
        The type of spectrum to return: 'power' for power spectrum, 'energy'
        for energy spectrum, and 'l2norm' for the l2-norm spectrum.
    unit : str, optional, default = 'per_l'
        If 'per_l', return the total contribution to the spectrum for each
        spherical harmonic degree l. If 'per_lm', return the average
        contribution to the spectrum for each coefficient at spherical
        harmonic degree l.
    """
    if normalization.lower() not in ('4pi', 'ortho', 'schmidt', 'unnorm'):
        raise ValueError("The normalization must be '4pi', 'ortho', " +
                         "'schmidt', or 'unnorm'. Input value was {:s}."
                         .format(repr(normalization)))

    if convention.lower() not in ('power', 'energy', 'l2norm'):
        raise ValueError("convention must be 'power', 'energy', or " +
                         "'l2norm'. Input value was {:s}"
                         .format(repr(convention)))

    if unit.lower() not in ('per_l', 'per_lm'):
        raise ValueError("unit must be 'per_l' or 'per_lm'." +
                         "Input value was {:s}".format(repr(unit)))

    if lmax is None:
        lmax = len(clm1[0, :, 0]) - 1

    if degrees is None:
        degrees = np.arange(lmax + 1).astype(int)
    else:
        degrees = degrees.astype(int)

    if clm1.ndim < 3:
        raise ValueError('clm1 and clm2 must be at least 3D')
    else:
        ashape = clm1.shape[3:]

    # Initialize array for the 1D energy/power spectrum shaped (truncation, ...)
    spectrum = np.empty((len(degrees),) + ashape)

    if clm2 is None:
        clm_sqd = (clm1 * clm1.conjugate()).real
    else:
        assert clm2.shape == clm1.shape, \
            "Arrays 'clm1' and 'clm2' of spectral coefficients must have the same shape. " \
            "Expected 'clm2' shape: {} got: {}".format(clm1.shape, clm2.shape)

        clm_sqd = (clm1 * clm2.conjugate()).real

    for degree in degrees:
        spectrum[degree] = clm_sqd[0, degree, 0:degree + 1].sum(axis=0)
        spectrum[degree] += clm_sqd[1, degree, 1:degree + 1].sum(axis=0)

    if convention.lower() == 'l2norm':
        return spectrum
    else:
        if normalization.lower() == '4pi':
            pass
        elif normalization.lower() == 'schmidt':
            spectrum /= (2. * degrees + 1.)
        elif normalization.lower() == 'ortho':
            spectrum /= (4. * np.pi)

    if convention.lower() == 'energy':
        spectrum *= 4. * np.pi

    if unit.lower() == 'per_l':
        pass
    elif unit.lower() == 'per_lm':
        spectrum /= (2. * degrees + 1.)

    return spectrum
