import numpy as np
from scipy.integrate import cumulative_trapezoid

import constants as cn


def height_to_pressure_std(height):
    r"""Convert height data to pressures using the U.S. standard atmosphere.

    The implementation inverts the formula outlined in [Hobbs1977]_ pg.60-61.

    Parameters
    ----------
    height : `pint.Quantity`
        Atmospheric height

    Returns
    -------
    `pint.Quantity`
        The corresponding pressure value(s)

    Notes
    -----
    .. math:: p = p_0 e^{\frac{g}{R \Gamma} \text{ln}(1-\frac{Z \Gamma}{T_0})}

    """
    p0 = 101325  # * units.Pa
    return p0 * (1.0 - (cn.gamma / cn.t0) * height) ** (cn.g / (cn.Rd * cn.gamma))


def hydrostatic_thickness(pressure, temperature, initial=0.0, axis=-1):
    """Calculate the thickness of a layer via the hypsometric equation.

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


def geopotential_height(temperature, psfc, pressure, axis=0):
    """
    Computes geopotential height from pressure and temperature profiles

    :param temperature: temperature profile
    :param pressure: pressure profile
    :param psfc: surface pressure
    :param axis: specifies axis of vertical dimension
    :return: geopotential height
    """
    nlevels = pressure.size

    temp = np.moveaxis(temperature, axis, 1)

    ashape = temperature.shape[:1]

    # Compute height via the hypsometric equation (hydrostatic layer thickness).
    height = np.zeros_like(temp)

    # Search last level pierced by terrain for each vertical column
    level_m = nlevels - np.searchsorted(np.sort(pressure), psfc)

    for ij in np.ndindex(ashape):
        # mask data above the surface
        ind_atm = level_m[ij]

        # approximate surface temperature with first level above the ground
        ts = temp[ij][ind_atm:ind_atm + 1]

        pres_m = np.append(psfc[ij], pressure[ind_atm:])
        temp_m = np.append(ts, temp[ij][ind_atm:], axis=0)

        # Integrating the hypsometric equation
        height[ij][ind_atm:] = hydrostatic_thickness(pres_m, temp_m, initial=None, axis=0)

    # return geopotential height (m)
    return np.moveaxis(height, 1, axis)


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

    (Prior to MetPy v0.11, this formula instead calculated :math:`g(z)` from Newton's Law of
    Gravitation assuming a spherical Earth and no centrifugal force effects.)
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


def static_stability(pressure, temperature, axis=0):
    r"""Calculate the static stability within a vertical profile.

    .. math:: \sigma = -\frac{RT}{p} \frac{\partial \ln \theta}{\partial p}

    This formula is based on equation 4.3.6 in [Bluestein1992]_.

    Parameters
    ----------
    pressure : `np.ndarray`
        Profile of atmospheric pressure
    temperature : `np.ndarray`
        Profile of temperature
    axis : int, optional, defaults to 0.
        The axis corresponding to vertical in the pressure and temperature arrays.
    Returns
    -------
        The profile of static stability.
    """
    theta = potential_temperature(pressure, temperature)

    return - cn.Rd * temperature / pressure * np.gradient(np.log(theta), pressure, axis=axis)


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


def vertical_velocity(pressure, omega, temperature, axis=None):
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
    axis:
        axis corresponding to the vertical dimension
    Returns
    -------
    `np.ndarray`
        Vertical velocity in terms of height (in meters / second)
    """
    if axis is not None:
        assert pressure.size == omega.shape[axis], "Variable 'omega' should have " \
                                                   "the same shape as 'pressure' along axis"
        assert pressure.size == temperature.shape[axis], "Variable 'temperature' should have " \
                                                         "the same shape as 'pressure' along axis"
    else:
        axis = -1

    omega = np.moveaxis(omega, axis, -1)
    temperature = np.moveaxis(temperature, axis, -1)

    w = - omega / (cn.g * density(pressure, temperature))  # (m/s)

    return np.moveaxis(w, -1, axis)


def pressure_vertical_velocity(pressure, w, temperature):
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
    w: `np.ndarray`
        Vertical velocity in terms of height
    pressure: `np.ndarray`
        Total atmospheric pressure
    temperature: `np.array`
        Air temperature

    Returns
    -------
    `np.ndarray`
        Vertical velocity in terms of pressure (in Pascals / second)
    """
    return - cn.g * density(pressure, temperature) * w  # (Pa/s)
