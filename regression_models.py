import warnings

warnings.filterwarnings('ignore')

import numpy as np
from scipy import stats
from scipy.special import expit
from scipy.integrate import cumulative_trapezoid

earth_rad = 6371.2  # km
two_pi = 2.0 * np.pi
four_pi = 2.0 * two_pi
earth_frq = 7.2921e-5  # rad/s
omega = 2.0 * earth_frq
g = 9.8  # m/s
ln_ps = np.log(1000.0)
RT_0 = 287.05 * 300.0


def zpm_to_zm(zpm):
    return zpm - RT_0 * ln_ps / g


def kappa_from_deg(ls):
    """
        Returns wavenumber [km^-1] from spherical harmonics degree (ls)
        λ = 2π Re / sqrt[l(l + 1)]
        kappa = 1 / λ
    """
    return np.sqrt(ls * (ls + 1.0)) / (two_pi * earth_rad)


def lambda_from_deg(ls):
    """
    Returns wavelength from spherical harmonics degree (ls)
    """
    return 1.0 / kappa_from_deg(ls)


def deg_from_lambda(lb):
    """
        Returns wavelength from spherical harmonics degree (ls)
    """
    return np.floor(np.sqrt(0.25 + (two_pi * earth_rad / lb) ** 2) - 0.5).astype(int)


def coriolis_parameter(latitude, squared=False):
    r"""Calculate the coriolis parameter at each point.
    The implementation uses the formula outlined in [Hobbs1977]_ pg.370-371.
    Parameters
    ----------
    latitude : array_like
        Latitude at each point
    """
    fc = omega * np.sin(np.deg2rad(latitude))

    if squared:
        return fc * fc
    else:
        return fc


def brunt_vaisala_frequency(z, squared=False):
    # Piecewise linear regression Brunt-Vaisala frequency
    breaks = [1.07286139e+04, 1.15887021e+04]
    slopes = [2.47934135e-07, 1.03969079e-05, 4.67723271e-08]
    interc = [0.00938348, -0.09950095, 0.02044369]

    z = np.atleast_1d(z)

    ind = np.searchsorted(breaks, z)

    result = np.array([interc[ind[i]] + slopes[ind[i]]
                       * zi for i, zi in enumerate(z)])

    if squared:
        return result * result
    else:
        return result


def compute_vwn(omega, kappa, height=20, lat=40.0, scale=8., squared=False):
    # ----------------------------------------------------------------------
    # Compute vertical wavenumber for linear non-hydrostatic gravity waves
    # ----------------------------------------------------------------------
    omega_sqd = omega * omega
    kappa_sqd = kappa * kappa

    n_sqd = brunt_vaisala_frequency(1e3 * height, squared=True)
    f_sqd = coriolis_parameter(lat, squared=True)

    nhterm = 1.0 / (2.0 * scale) ** 2

    m_sqd = kappa_sqd * (n_sqd - omega_sqd) / (omega_sqd - f_sqd) - nhterm

    if squared:
        return m_sqd
    else:
        return np.sqrt(m_sqd)


def compute_he(mz, omega, height, height_trop=16.0, lat=40.0):
    corio_sqd = coriolis_parameter(lat, squared=True)
    omega_sqd = omega * omega

    n2_trop = brunt_vaisala_frequency(2.0e3, squared=True)
    n2_stra = brunt_vaisala_frequency(2.4e4, squared=True)

    a1 = 1.0 + (n2_stra - omega_sqd) / (n2_trop - omega_sqd)
    a2 = (a1 - 2.0) * np.cos(2.0 * mz * height)

    if height < height_trop:
        arg = (a1 - a2) / (a1 + a2)
    else:
        arg = np.ones_like(a1)

    return np.sqrt(arg / (1.0 + corio_sqd / omega_sqd)) / mz


def compute_expectation(x, stat_func=None):
    if stat_func is None:
        stat_func = lambda x, n: np.nanmean(x)

    expected_value, ci = stats.bootstrap(x, statistic=stat_func)

    return expected_value, ci


def diagnose_anisotropy(u, v):
    # compute expectations
    Euu, _ = compute_expectation(u ** 2)
    Evv, _ = compute_expectation(v ** 2)
    Euv, _ = compute_expectation(u * v)

    return (Euu - Evv) / Euv


def decompose_spectra(k, cu, cv, cuv, anisotropy=0.0, axis=-1):
    """Apply anisotropic Helmholtz decomposition to 1D Energy spectra
    :param cu: Power spectra of u component
    :param cv: Power spectra of v component
    :param cuv: Cross-spectra of u and v components
    :return: rotational and divergent spectra
    """
    arg_ans = cv - cu + anisotropy * cuv

    # reverse integration
    arg_ans = np.moveaxis(arg_ans, axis, -1)[..., ::-1]
    k_integ = k[::-1]

    # k_integ = np.insert(k_integ, 0, 0.0)
    # arg_ans = np.insert(arg_ans, 0, np.zeros(arg_ans.shape[:-1]), axis=-1)

    # compute semi indefinite integral
    aniso_integrand = cumulative_trapezoid(
        arg_ans, x=k_integ, axis=axis,
        initial=k_integ[0])

    # correct for sign and sort data
    aniso_integrand = - aniso_integrand[..., ::-1] / k

    # back to original axes
    aniso_integrand = np.moveaxis(aniso_integrand, -1, axis)

    # perform decomposition
    rot_spectra = (cv + aniso_integrand) / 2.0
    div_spectra = (cu - aniso_integrand) / 2.0

    rot_spectra[rot_spectra < 0.0] = np.nan
    div_spectra[div_spectra < 0.0] = np.nan

    div_spectra[..., -1] = div_spectra[..., -2]

    return rot_spectra, div_spectra


def compute_wave_energy(div_spectra, cw):
    return 2.0 * div_spectra + cw


def masscont_model(k, divergent_spectra, alpha=0.5, beta=0.11, height=20., anisotropy=True):
    # compute depth of layers with effectively uniform divergent flow
    # beta2 = np.sqrt(2.0) * (1.0 + beta / height - np.log(height / beta) )

    arg = (beta * height * k) ** 2

    # filter function controls the relationship between Ew and ED
    if anisotropy:
        filter_arg = alpha * arg / (alpha + arg)
    else:
        filter_arg = arg

    return filter_arg * divergent_spectra


def largescaleIG_model(k, divergent_spectra, rotational_spectra, ganma=0.5, height=30.0, lat=40.0, mid_freq=False):
    # filter function controls the relationship between Ew and ED
    r = divergent_spectra / rotational_spectra
    total_spectra = divergent_spectra + rotational_spectra

    # coriolis parameter at 45 degrees
    n2 = brunt_vaisala_frequency(1e3 * height, squared=True)
    f2_n2 = coriolis_parameter(lat, squared=True) / n2

    d = ganma * r

    if not mid_freq:
        d = np.where(r <= 1.0, ganma * r, d * (d - 1.0) / (d + 1.0))

    return f2_n2 * d * total_spectra


def mesoscaleIG_model(k, divergent_spectra, rotational_spectra, ganma=0.5, height=30.0, lat=40.0, mid_freq=False):
    # filter function controls the relationship between Ew and ED
    r = divergent_spectra / rotational_spectra
    total_spectra = divergent_spectra + rotational_spectra

    # coriolis parameter at mid latitudes (latitude in degrees)
    f_sqd = coriolis_parameter(lat, squared=True)

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
    f2 = coriolis_parameter(45.0, squared=True)
    n2 = brunt_vaisala_frequency(1e3 * height, squared=True)

    # Bw = (np.sqrt(f2) / n2) * np.sqrt(R - 1.0) * ( (n2/f2 - R)**1.5 ) / R
    # Bw[np.isnan(Bw)] = 356512.93476716

    Bw = (np.sqrt(f2) / n2) * (n2 / f2 - r) / r

    # result = np.where(R < 2.1, (f2/n2) * (R + 1.0) * divergent_spectra, Bw * k ** 3 )

    return abs(Bw) * k ** 2


def nonh_waves_energy_ratios(R, height=20e3, height_scale=8e3, lat_0=45.0):
    """
        Gravity-wave kinetic energy ratios
        input: R (ratio of divergent to rotational kinetic energies)
        height: height above the ground in m
        height_scale: density height scale in m
        lat_0: central latitude of the model
    """

    f2 = coriolis_parameter(lat_0, squared=True)
    n2 = brunt_vaisala_frequency(height, squared=True)

    h2 = height_scale * height_scale
    omg2 = R * f2

    # Assuming wave saturation-cascade conditions (Dewan 1997)
    eps = 1e-5  # m^2 / s^3
    c = 0.4e3  # a1 / a2

    m2 = (c / eps) * n2 * np.sqrt(f2 * R)

    # Non-hydrostatic term
    mnh = n2 * (m2 * h2 + 0.25)

    denm_ke_pe = (n2 * mnh - omg2) * (R - 1.0)

    # H Kinetic to potential energy ratio:
    ke_pe = mnh * (R + 1.0) / denm_ke_pe

    # V Kinetic to potential energy ratio:
    denm_ve_pe = omg2 * (n2 / omg2 - 1.0) ** 2
    print(denm_ve_pe)

    denm_ve_pe -= n2 * (n2 - omg2 + mnh) / omg2
    print(denm_ve_pe)

    ve_pe = mnh / denm_ve_pe

    ve_ke = ve_pe / ke_pe

    # ratio of vertical to horizontal wave kinetic energy
    return ke_pe, ve_pe, ve_ke


def logistic_func(x, x0=0.0, a=0.0, b=1.0, c0=1.0):
    # bounded logistic function in [a, b] granted that a < b
    return a + abs(b - a) * expit(c0 * (x - x0))


def w_spectral_model(k, divergent_spectra, rotational_spectra, beta=0.11, ganma=0.5, height=30):
    # Compute f^2 / N^2
    # coriolis parameter at 45 degrees
    f2 = coriolis_parameter(45.0, squared=True)
    n2 = brunt_vaisala_frequency(1e3 * height, squared=True)

    # wave and continuity components of the model
    # R = simps(divergent_spectra / rotational_spectra, x=k, axis=-1,
    # even='avg') #/ k
    R = divergent_spectra / rotational_spectra

    # large scales: omega << N
    wave_component = ganma * (f2 / n2) * (R + 1.0)

    # Compute depth of layers with effectively uniform divergent flow:
    # filter function controlling the ratio between Ew and ED
    mass_component = (beta * height * k) ** 2

    return divergent_spectra * (wave_component + mass_component)


def w_simple_model(k, alpha=1.0, beta=0.11, ganma=0.5, height=30):
    # Compute f^2 / N^2
    # coriolis parameter at 45 degrees
    f2 = coriolis_parameter(45.0, squared=True)
    n2 = brunt_vaisala_frequency(1e3 * height, squared=True)

    # large scales: omega << N

    wave_component = (f2 / n2) * (alpha * k ** (-1. / 3.) +
                                  ganma * (f2 / n2) * k ** (-5. / 3.))

    # Compute depth of layers with effectively uniform divergent flow:
    # filter function controlling the ratio between Ew and ED
    mass_component = beta * height ** 2

    return wave_component + mass_component * k ** (1. / 3.)


def w_simple_model_residuals(parameters, kappa, w_spectra, height):
    wm_spectra = w_simple_model(kappa,
                                alpha=parameters[0],
                                beta=parameters[1],
                                ganma=parameters[2],
                                height=height)
    # return residuals
    return wm_spectra - w_spectra


def rotdiv_ratio_model(k, alpha_0=0.5, beta_0=0.11, height=30.):
    # compute depth of layers with effectively uniform divergent flow
    hk2 = (height * k) ** 2
    arg = hk2 * beta_0 * beta_0

    # filter function controls the relationship between Ew and ED
    filter_arg = alpha_0 * arg / (alpha_0 + arg)

    # Compute f^2 / N^2
    # coriolis parameter at 45 degrees
    f2 = coriolis_parameter(45.0, squared=True)
    n2 = brunt_vaisala_frequency(1e3 * height, squared=True)

    return hk2 * (n2 / f2) * filter_arg - 1.0


def rotdiv_ratio_residuals(parameters, kappa, r_spectra, height):
    rm_spectra = rotdiv_ratio_model(kappa, alpha_0=parameters[
        0], beta_0=parameters[1], height=height)

    return rm_spectra - r_spectra


def w_model_residuals(parameters, kappa, div_spectra, rot_spectra, w_spectra, height):
    wm_spectra = w_spectral_model(kappa, div_spectra, rot_spectra,
                                  beta=parameters[0],
                                  ganma=parameters[1],
                                  height=height)
    # return residuals
    return wm_spectra - w_spectra


def masscont_model_residuals(parameters, kappa, div_spectra, w_spectra, height):
    wm_spectra = masscont_model(kappa, div_spectra,
                                alpha=parameters[0],
                                beta=parameters[1],
                                height=height)
    # return residuals
    return wm_spectra - w_spectra


def gardner_model_residuals(parameters, kappa, div_spectra, rot_spectra, w_spectra, height):
    wm_spectra = gardner_model(
        kappa, div_spectra, rot_spectra,
        ganma=parameters[0], height=height)

    # return residuals
    return wm_spectra - w_spectra
