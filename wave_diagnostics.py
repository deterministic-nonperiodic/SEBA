import warnings

import numpy as np
from scipy import stats
from scipy.integrate import cumulative_trapezoid

from kinematics import coriolis_parameter
from spectral_analysis import kappa_from_deg

warnings.filterwarnings('ignore')


def brunt_vaisala_frequency(z, squared=False):
    """
        Piece-wise linear function for the brunt vaisala frequency

        :param z: height in meter
        :param squared: bool, optional
            returns N**2
    """
    # Piecewise linear regression Brunt-Vaisala frequency
    breaks = [1.07286139e+04, 1.15887021e+04]
    slopes = [2.47934135e-07, 1.03969079e-05, 4.67723271e-08]
    intersect = [0.00938348, -0.09950095, 0.02044369]

    z = np.atleast_1d(z)

    ind = np.searchsorted(breaks, z)

    result = np.array([intersect[ind[i]] + slopes[ind[i]]
                       * zi for i, zi in enumerate(z)])

    if squared:
        return result * result
    else:
        return result


def compute_vwn(omega, kappa, height=20., lat=40.0, scale=8., squared=False):
    # ----------------------------------------------------------------------
    # Compute vertical wavenumber for linear non-hydrostatic gravity waves
    # ----------------------------------------------------------------------
    omega_sqd = omega * omega
    kappa_sqd = kappa * kappa

    n_sqd = brunt_vaisala_frequency(1e3 * height, squared=True)
    f_sqd = coriolis_parameter(lat) ** 2

    nh_term = 1.0 / (2.0 * scale) ** 2

    m_sqd = kappa_sqd * (n_sqd - omega_sqd) / (omega_sqd - f_sqd) - nh_term

    if squared:
        return m_sqd
    else:
        return np.sqrt(m_sqd)


def compute_he(mz, omega, height, height_trop=16.0, lat=40.0):
    coriolis_sqd = coriolis_parameter(lat) ** 2
    omega_sqd = omega * omega

    n2_trop = brunt_vaisala_frequency(2.0e3, squared=True)
    n2_stra = brunt_vaisala_frequency(2.4e4, squared=True)

    a1 = 1.0 + (n2_stra - omega_sqd) / (n2_trop - omega_sqd)
    a2 = (a1 - 2.0) * np.cos(2.0 * mz * height)

    if height < height_trop:
        arg = (a1 - a2) / (a1 + a2)
    else:
        arg = np.ones_like(a1)

    return np.sqrt(arg / (1.0 + coriolis_sqd / omega_sqd)) / mz


def compute_expectation(x, stat_func=None):
    if stat_func is None:
        stat_func = lambda a, n: np.nanmean(a)

    expected_value, ci = stats.bootstrap(x, statistic=stat_func)

    return expected_value, ci


def diagnose_anisotropy(u, v):
    # compute expectations
    euu, _ = compute_expectation(u ** 2)
    evv, _ = compute_expectation(v ** 2)
    euv, _ = compute_expectation(u * v)

    return (euu - evv) / euv


def decompose_spectra(k, cu, cv, cuv, anisotropy=0.0, axis=-1):
    """Apply anisotropic Helmholtz decomposition to 1D Energy spectra
    :param k: horizontal wavenumber
    :param cu: Power spectra of u component
    :param cv: Power spectra of v component
    :param cuv: Cross-spectra of u and v components
    :param anisotropy: anisotropy factor
    :param axis:
    :return: rotational and divergent spectra
    """
    arg_ans = cv - cu + anisotropy * cuv

    # reverse integration
    arg_ans = np.moveaxis(arg_ans, axis, -1)[..., ::-1]
    k_int = k[::-1]

    # compute semi indefinite integral
    anisotropy_integrand = cumulative_trapezoid(
        arg_ans, x=k_int, axis=axis,
        initial=k_int[0])

    # correct for sign and sort data
    anisotropy_integrand = - anisotropy_integrand[..., ::-1] / k

    # back to original axes
    anisotropy_integrand = np.moveaxis(anisotropy_integrand, -1, axis)

    # perform decomposition
    rot_spectra = (cv + anisotropy_integrand) / 2.0
    div_spectra = (cu - anisotropy_integrand) / 2.0

    rot_spectra[rot_spectra < 0.0] = np.nan
    div_spectra[div_spectra < 0.0] = np.nan

    div_spectra[..., -1] = div_spectra[..., -2]

    return rot_spectra, div_spectra


def compute_wave_energy(div_spectra, cw):
    return 2.0 * div_spectra + cw


def masscont_model(k, divergent_spectra, alpha=0.5, beta=0.11, height=20.,
                   anisotropy=True):
    # compute depth of layers with effectively uniform divergent flow
    # beta2 = np.sqrt(2.0) * (1.0 + beta / height - np.log(height / beta) )

    arg = (beta * height * k) ** 2

    # filter function controls the relationship between Ew and ED
    if anisotropy:
        filter_arg = alpha * arg / (alpha + arg)
    else:
        filter_arg = arg

    return filter_arg * divergent_spectra


def largescale_model(divergent_spectra, rotational_spectra,
                     ganma=0.5, height=30.0, lat=40.0, mid_freq=False):
    # filter function controls the relationship between Ew and ED
    r = divergent_spectra / rotational_spectra
    total_spectra = divergent_spectra + rotational_spectra

    # coriolis parameter at 45 degrees
    n2 = brunt_vaisala_frequency(1e3 * height, squared=True)
    f2_n2 = coriolis_parameter(lat) ** 2 / n2

    d = ganma * r

    if not mid_freq:
        d = np.where(r <= 1.0, ganma * r, d * (d - 1.0) / (d + 1.0))

    return f2_n2 * d * total_spectra


def mesoscale_model(k, divergent_spectra, rotational_spectra, ganma=0.5, height=30.0, lat=40.0):
    # filter function controls the relationship between Ew and ED
    r = divergent_spectra / rotational_spectra
    total_spectra = divergent_spectra + rotational_spectra

    # coriolis parameter at mid latitudes (latitude in degrees)
    f_sqd = coriolis_parameter(lat) ** 2

    # Compute intrinsic frequency
    omega_sqd = r * f_sqd

    omega = ganma * np.sqrt(omega_sqd)

    # compute vertical wavenumber
    mz = compute_vwn(omega, k, height=height, lat=lat, scale=7.0, squared=False)

    # compute effective height from wave properties
    he = np.sqrt(omega_sqd / (omega_sqd + f_sqd)) / mz

    # compute averaged effective height in the mesoscale
    mscale_mask = k > kappa_from_deg(100.0)

    he = np.nanmean(he[mscale_mask])

    return total_spectra * (he * k) ** 2


def gardner_model(k, divergent_spectra, rotational_spectra, ganma=0.5, height=30.0, ):
    # filter function controls the relationship between Ew and ED
    r = ganma * divergent_spectra / rotational_spectra

    # coriolis parameter at 45 degrees
    f2 = coriolis_parameter(45.0) ** 2
    n2 = brunt_vaisala_frequency(1e3 * height, squared=True)

    # Bw = (np.sqrt(f2) / n2) * np.sqrt(R - 1.0) * ( (n2/f2 - R)**1.5 ) / R
    # Bw[np.isnan(Bw)] = 356512.93476716

    bw = (np.sqrt(f2) / n2) * (n2 / f2 - r) / r
    return abs(bw) * k ** 2


def w_spectral_model(k, divergent_spectra, rotational_spectra, beta=0.11, ganma=0.5, height=30):
    # Compute f^2 / N^2
    # coriolis parameter at 45 degrees
    f2 = coriolis_parameter(45.0) ** 2
    n2 = brunt_vaisala_frequency(1e3 * height, squared=True)

    # wave and continuity components of the model
    r = divergent_spectra / rotational_spectra

    # large scales: omega << N
    wave_component = ganma * (f2 / n2) * (r + 1.0)

    # Compute depth of layers with effectively uniform divergent flow:
    # filter function controlling the ratio between Ew and ED
    mass_component = (beta * height * k) ** 2

    return divergent_spectra * (wave_component + mass_component)


def rotdiv_ratio_model(k, alpha_0=0.5, beta_0=0.11, height=30.):
    # compute depth of layers with effectively uniform divergent flow
    hk2 = (height * k) ** 2
    arg = hk2 * beta_0 * beta_0

    # filter function controls the relationship between Ew and ED
    filter_arg = alpha_0 * arg / (alpha_0 + arg)

    # Compute f^2 / N^2
    # coriolis parameter at 45 degrees
    f2 = coriolis_parameter(45.0) ** 2
    n2 = brunt_vaisala_frequency(1e3 * height, squared=True)

    return hk2 * (n2 / f2) * filter_arg - 1.0
