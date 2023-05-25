import numpy as np
from scipy.integrate import cumulative_trapezoid

import constants as cn
from tools import broadcast_1dto, gradient_1d


def hydrostatic_thickness(pressure, temperature, initial=0.0, axis=-1):
    r"""Calculate the cumulative thickness of layers from temperature profiles
        via the hypsometric equation. Returns the thickness of layers between
        every level and the first level.

    .. math:: Z_2 - Z_1 = -\frac{R_d}{g} \int_{p_1}^{p_2} T(p) d\ln p,

    Parameters
    ----------
    pressure : `np.ndarray`
        Atmospheric pressure profile
    temperature : `np.ndarray`
        Atmospheric temperature profile
    initial: object, optional
    axis: int
        Axis along which the integration is carried out
    Returns
    -------
    `np.ndarray`
        The thickness of the layer in meters.
    """
    mean_temp = cumulative_trapezoid(temperature, x=np.log(pressure), axis=axis, initial=initial)
    return - (cn.Rd / cn.g) * mean_temp


def height_to_geopotential(height):
    r"""Compute geopotential for a given height above sea level.

    Calculates the geopotential from height above mean sea level using the following formula,
    which is derived from the definition of geopotential as given in [Hobbs2006]_ Pg. 69 Eq
    3.21, along with an approximation for variation of gravity with altitude:

    .. math:: \Phi = \frac{g R_e z}{R_e + z}

    (where :math:`\Phi` is geopotential, :math:`z` is height, :math:`R_e` is average Earth
    radius, and :math:`g` is standard gravity.)

    Parameters
    ----------
    height : `np.ndarray`
        Height above sea level

    Returns
    -------
    `np.ndarray`
        The corresponding geopotential value(s)

    Notes
    -----
    This calculation approximates :math:`g(z)` as

    .. math:: g(z) = g_0 \left( \frac{R_e}{R_e + z} \right)^2

    where :math:`g_0` is standard gravity. It thereby accounts for the average effects of
    centrifugal force on apparent gravity, but neglects latitudinal variations due to
    centrifugal force and Earth's eccentricity.
    """
    return (cn.g * cn.earth_radius * height) / (cn.earth_radius + height)


def geopotential_to_height(geopotential):
    r"""Compute height above sea level from a given geopotential.

    Calculates the height above mean sea level from geopotential using the following formula,
    which is derived from the definition of geopotential as given in [Hobbs2006]_ Pg. 69 Eq
    3.21, along with an approximation for variation of gravity with altitude:

    .. math:: z = \frac{\Phi R_e}{gR_e - \Phi}

    (where :math:`\Phi` is geopotential, :math:`z` is height, :math:`R_e` is average Earth
    radius, and :math:`g` is standard gravity.)

    Parameters
    ----------
    geopotential : `np.ndarray`
        Geopotential

    Returns
    -------
    `np.array`
        The corresponding value(s) of height above sea level

    Notes
    -----
    This calculation approximates :math:`g(z)` as

    .. math:: g(z) = g_0 \left( \frac{R_e}{R_e + z} \right)^2

    where :math:`g_0` is standard gravity. It thereby accounts for the average effects of
    centrifugal force on apparent gravity, but neglects latitudinal variations due to
    centrifugal force and Earth's eccentricity.
    """
    return (geopotential * cn.earth_radius) / (cn.g * cn.earth_radius - geopotential)


def exner_function(pressure, reference_pressure=cn.ps):
    r"""Calculate the Exner function.

    .. math:: \Pi = \left( \frac{p}{p_0} \right)^\kappa

    This can be used to calculate potential temperature from temperature (and visa-versa),
    since

    .. math:: \Pi = \frac{T}{\theta}

    Parameters
    ----------
    pressure : total atmospheric pressure
    reference_pressure : optional
        The reference pressure against which to calculate the Exner function, defaults to
        metpy.constants.P0

    Returns
    -------
        The value of the Exner function at the given pressure
    """
    return (pressure / reference_pressure) ** cn.chi


def potential_temperature(pressure, temperature):
    r"""Calculate the potential temperature.

    Uses the Poisson equation to calculation the potential temperature
    given `pressure` and `temperature`.

    Parameters
    ----------
    pressure : total atmospheric pressure
    temperature : air temperature

    Returns
    -------
        The potential temperature corresponding to the temperature and pressure.
    """
    return temperature / exner_function(pressure)


def static_stability(pressure, temperature, vertical_axis=0):
    r"""Calculate the static stability within a vertical profile.

    .. math:: \sigma = -\frac{R_d T}{p} \frac{\partial \ln \theta}{\partial p}

    This formula is based on equation 4.3.6 in [Bluestein1992]_.

    Parameters
    ----------
    pressure : `np.ndarray`
        Profile of atmospheric pressure
    temperature : `np.ndarray`
        Profile of temperature
    vertical_axis : int, optional, defaults to 0.
        The axis corresponding to vertical in the pressure and temperature arrays.
    Returns
    -------
        The profile of static stability.
    """
    theta = potential_temperature(pressure, temperature)
    ddp_theta = gradient_1d(np.log(theta), pressure, axis=vertical_axis)

    return - cn.Rd * (temperature / pressure) * ddp_theta


def lorenz_parameter(pressure, theta, vertical_axis=0):
    # Static stability parameter ganma to convert from temperature variance to APE
    # using d(theta)/d(ln p) gives smoother gradients at the top/bottom boundaries.
    ddp_theta = gradient_1d(theta, pressure, axis=vertical_axis)

    ganma = - cn.Rd * exner_function(pressure) / (pressure * ddp_theta)
    # # remove unstable and neutral profiles
    # ganma[ddp_theta <= 0] = 0.0
    return ganma


def density(pressure, temperature):
    r"""Calculate density.

    This calculation must be given an air parcel's pressure, temperature, and mixing ratio.
    The implementation uses the formula outlined in [Hobbs2006]_ pg.67.

    Parameters
    ----------
    pressure: `np.ndarray`
        Total atmospheric pressure
    temperature: `np.ndarray`
        air temperature

    Returns
    -------
    `np.ndarray`
        The corresponding density of the parcel
    """
    return pressure / (cn.Rd * temperature)  # (kg/m**3)


def specific_volume(pressure, temperature):
    r"""Calculate specific volume.

    .. math:: \alpha = -\frac{1}{\rho}

    Parameters
    ----------
    pressure: `np.ndarray`
        Total atmospheric pressure
    temperature: `np.ndarray`
        air temperature

    Returns
    -------
    `np.ndarray`
        The corresponding density of the parcel
    """
    return 1.0 / density(pressure, temperature)


def vertical_velocity(omega, temperature, pressure):
    r"""Calculate omega from w assuming hydrostatic conditions.

    This function converts from vertical velocity in pressure coordinates
    to height-based vertical velocity assuming a hydrostatic atmosphere.
    By Equation 7.33 in [Hobbs2006]_,

    .. math:: \omega \simeq - w / \rho g

    Density (:math:`\rho`) is calculated using the :func:`density` function,
    from the given pressure and temperature.

    Parameters
    ----------
    pressure: `np.ndarray`
        1D profile of total atmospheric pressure
    omega: `np.ndarray`
        Vertical velocity in pressure coordinates
    temperature: `np.array`
        Air temperature
    Returns
    -------
    `np.ndarray`
        Vertical velocity in terms of height (in meters / second)
    """
    if pressure.ndim == 1:
        assert omega.shape == temperature.shape, "All variables should have the same shape"
        pressure = broadcast_1dto(pressure, omega.shape)

    return - omega / (cn.g * density(pressure, temperature))  # (m/s)


def pressure_vertical_velocity(w, temperature, pressure):
    r"""Calculate omega from w assuming hydrostatic conditions.

    This function converts vertical velocity with respect to height
    :math:`\left(w = \frac{Dz}{Dt}\right)` to that
    with respect to pressure :math:`\left(\omega = \frac{Dp}{Dt}\right)`
    assuming hydrostatic conditions on the synoptic scale.
    By Equation 7.33 in [Hobbs2006]_,

    .. math:: \omega \simeq -\rho g w

    Density (:math:`\rho`) is calculated using the :func:`density` function,
    from the given pressure and temperature.

    Parameters
    ----------
    w: `np.ndarray`, `xr.DataArray`
        Vertical velocity in terms of height
    pressure: `np.ndarray`, `xr.DataArray`
        Total atmospheric pressure
    temperature: `np.array`, `xr.DataArray`
        Air temperature
    Returns
    -------
    `np.ndarray`
        Vertical velocity in terms of pressure (in Pascals / second)
    """
    if pressure.ndim == 1:
        assert w.shape == temperature.shape, "All variables should have the same shape"
        pressure = broadcast_1dto(pressure, w.shape)

    return - cn.g * density(pressure, temperature) * w  # (Pa/s)


def brunt_vaisala_squared(pressure, temperature, vertical_axis=0):
    # compute potential temperature
    theta = potential_temperature(pressure, temperature)

    # compute the stability parameter
    gamma = lorenz_parameter(pressure, theta, vertical_axis=vertical_axis)

    return (cn.g / theta) ** 2 / gamma
