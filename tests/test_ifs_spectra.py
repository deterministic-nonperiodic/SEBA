import warnings

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from numpy.core.numeric import normalize_axis_index

from src.seba import EnergyBudget
from src.spectral_analysis import kappa_from_lambda, kappa_from_deg
from src.spectral_analysis import triangular_truncation
from src.tools import search_closet, getspecindx
from src.visualization import spectra_base_figure, reference_slopes

params = {'xtick.labelsize': 'medium',
          'ytick.labelsize': 'medium',
          'text.usetex': True, 'font.size': 14,
          'font.family': 'serif', 'font.weight': 'normal'}
plt.rcParams.update(params)
plt.rcParams['legend.title_fontsize'] = 12

warnings.filterwarnings('ignore')


def cast_to_complex(rl, im):
    return np.vectorize(complex)(rl, im)


def compute_spectra(clm1, clm2=None, degrees=None, convention='power', axis=None):
    """Returns the cross-spectrum of the spherical harmonic coefficients as a
    function of spherical harmonic degree.

    Signature
    ---------
    array = cross_spectrum(clm1, [clm2, normalization, convention, unit])

    Parameters
    ----------
    clm1 : ndarray, shape ((ntrunc+1)*(ntrunc+2)/2, ...)
        contains the first set of spherical harmonic coefficients.
    clm2 : ndarray, shape ((ntrunc+1)*(ntrunc+2)/2, ...), optional
        contains the second set of spherical harmonic coefficients.
    degrees: 1D array, optional, default = None
        Spherical harmonics degree. If not given, degrees are inferred from
        the class definition or calculated from the number of latitude points.
    convention : str, optional, default = 'power'
        The type of spectrum to return: 'power' for power spectrum, 'energy'
        for energy spectrum, and 'l2norm' for the l2-norm spectrum.
    axis: int,
        axis of the spectral coefficients
    Returns
    -------
    array : ndarray, shape (len(degrees), ...)
        contains the 1D spectrum as a function of spherical harmonic degree.
    """

    # Get indexes of the triangular matrix with spectral coefficients
    # (move this to class init?)
    if axis is None:
        axis = 0
    else:
        axis = normalize_axis_index(axis, clm1.ndim)

    clm1 = np.moveaxis(clm1, axis, 0)

    sample_shape = clm1.shape[1:]

    if degrees is None:
        ntrunc = triangular_truncation(clm1.shape[0])
        degrees = np.arange(ntrunc + 1, dtype=int)

    if convention not in ['energy', 'power']:
        raise ValueError("Parameter 'convention' must be one of"
                         " ['energy', 'power']. Given {}".format(convention))

    if clm2 is None:
        clm_sqd = (clm1 * clm1.conjugate()).real
    else:
        clm2 = np.moveaxis(clm2, axis, 0)

        assert clm2.shape == clm1.shape, \
            "Arrays 'clm1' and 'clm2' of spectral coefficients must have the same shape. " \
            "Expected 'clm2' shape: {} got: {}".format(clm1.shape, clm2.shape)

        clm_sqd = (clm1 * clm2.conjugate()).real

    # define wavenumbers locally
    ms, ls = getspecindx(degrees.size - 1)

    # Multiplying by 2 to account for symmetric coefficients (ms != 0)
    clm_sqd = (np.where(ms == 0, 1.0, 2.0) * clm_sqd.T).T

    # Initialize array for the 1D energy/power spectrum shaped (truncation, ...)
    spectrum = np.zeros((degrees.size,) + sample_shape)

    # Compute spectrum as a function of total wavenumber by adding up the zonal wavenumbers.
    for ln, degree in enumerate(degrees):
        # Sum over all zonal wavenumbers <= total wavenumber
        degree_range = (ms <= degree) & (ls == degree)
        spectrum[ln] = np.nansum(clm_sqd[degree_range], axis=0)

    # Using the normalization in equation (7) of Lambert [1984].
    spectrum /= 2.0

    if convention.lower() == 'energy':
        spectrum *= 4.0 * np.pi

    return np.moveaxis(spectrum, 0, axis)


def horizontal_kinetic_energy(vrt, div, axis=0):
    """
    Horizontal kinetic energy after Augier and Lindborg (2013), Eq.13
    :return:
    """
    coeffs_size = vrt.shape[axis]

    ntrunc = triangular_truncation(coeffs_size)
    ls = np.arange(ntrunc + 1, dtype=int)

    vector_norm = kappa_from_deg(ls) ** 2
    vector_norm[ls == 0] = 1.0

    vrt_sqd = compute_spectra(vrt, convention='energy', axis=axis)
    div_sqd = compute_spectra(div, convention='energy', axis=axis)

    return (vrt_sqd + div_sqd) / (2.0 * vector_norm)


if __name__ == "__main__":
    # -------------------------------------------------------------------------------
    #    Test IFS kinetic energy spectra
    # -------------------------------------------------------------------------------

    p_levels = [850e2, 100e2]

    # compute spectra from coefficients
    dataset_sps = xr.open_mfdataset("../data/IFS_atm_3d_inst_n256_sps_200.nc")

    p_index = search_closet(dataset_sps.plev.values, p_levels)

    # from IFS original spectral coefficients
    vrt_spc = dataset_sps.vor.values.mean(axis=0)
    div_spc = dataset_sps.div.values.mean(axis=0)
    w_spc = dataset_sps.w.values.mean(axis=0)

    # cast to complex arrays
    vrt_spc = cast_to_complex(vrt_spc[..., 0], vrt_spc[..., 1])
    div_spc = cast_to_complex(div_spc[..., 0], div_spc[..., 1])
    w_spc = cast_to_complex(w_spc[..., 0], w_spc[..., 1])

    hke_sps = horizontal_kinetic_energy(vrt_spc, div_spc, axis=1)
    vke_sps = compute_spectra(w_spc, convention='energy', axis=1) / 2.0

    # select levels
    hke_sps = hke_sps[p_index]
    vke_sps = vke_sps[p_index]

    # from grid-point fields
    budget = EnergyBudget("data/IFS_atm_3d_inst_n256_200.nc")

    kappa = 1e3 * budget.kappa_h

    hke_gps = 4 * np.pi * budget.horizontal_kinetic_energy().values.squeeze()

    # select levels
    p_index = search_closet(budget.pressure, p_levels)
    hke_gps = hke_gps[p_index]

    # load ICON as control
    budget = EnergyBudget("data/ICON_atm_3d_inst_n256_gps_200.nc")
    hke_icon = 4 * np.pi * budget.horizontal_kinetic_energy().values.squeeze()
    vke_icon = 4 * np.pi * budget.vertical_kinetic_energy().values.squeeze()

    p_index = search_closet(budget.pressure, p_levels)
    hke_icon = hke_icon.mean(0)[p_index]
    vke_icon = vke_icon.mean(0)[p_index]

    # DYAMOND simulations
    dataset_test = xr.open_mfdataset("../data/IFS_dyamond_test.nc")

    p_index = search_closet(dataset_test.plev.values, p_levels)

    # from IFS original spectral coefficients
    vrt_spc = dataset_test.vo.values.mean(axis=0)
    div_spc = dataset_test.d.values.mean(axis=0)

    # cast to complex arrays
    vrt_spc = cast_to_complex(vrt_spc[..., 0], vrt_spc[..., 1])
    div_spc = cast_to_complex(div_spc[..., 0], div_spc[..., 1])

    hke_ifs = horizontal_kinetic_energy(vrt_spc, div_spc, axis=1)

    kappa_ifs = 1e3 * kappa_from_deg(np.arange(triangular_truncation(vrt_spc.shape[1]) + 1,
                                               dtype=int))

    # select levels
    hke_ifs = hke_ifs[p_index]

    # Create figure for horizontal kinetic energy
    x_scales = [kappa_from_lambda(2 * np.linspace(3000, 600, 2)),
                kappa_from_lambda(2 * np.linspace(300, 50, 2))]
    scale_st = ['-3', '-5/3']
    scale_mg = [1.0e-6, 0.1e-2]

    y_label = r'Horizontal kinetic energy $[m^2/s^2]$'
    y_limits = [8e-6, 8e4]

    layers = ['Troposphere', 'Stratosphere']
    ax_titles = [layers[lv < 500e2] + ' ({:>2d} hPa)'.format(int(1e-2 * lv)) for lv in p_levels]

    fig, axes = spectra_base_figure(n_rows=1, n_cols=len(p_levels), y_limits=y_limits,
                                    y_label=y_label, y_scale='log', ax_titles=ax_titles,
                                    lambda_lines=[40, ], frame=True, truncation='n2048')

    for i, ax in enumerate(axes.ravel()):
        ax.plot(kappa[:-1], hke_icon[i, :-1], lw=1.5, color='k',
                label="ICON", linestyle='solid', alpha=1.0)

        ax.plot(kappa_ifs, hke_ifs[i], lw=1.5, color='magenta',
                label="IFS DYAMOND", linestyle='solid', alpha=1.0)

        ax.plot(kappa, 2 * hke_sps[i], lw=1.5, color='blue',
                label="IFS scaled by 2",
                linestyle='dashed', alpha=1.0)

        ax.plot(kappa, hke_sps[i], lw=1.5, color='red',
                label=r"IFS from $\zeta$ and $\delta$ coefficients",
                linestyle='solid', alpha=1.0)

        ax.plot(kappa[:-1], hke_gps[i, :-1], lw=1.5, color='green',
                label=r"IFS from grid-point $\zeta$ and $\delta$",
                linestyle='solid', alpha=1.0)

        reference_slopes(ax, x_scales, scale_mg, scale_st)

    axes[0].legend(loc='lower left')

    plt.show()

    # Create figure for vertical kinetic energy
    x_scales = [kappa_from_lambda(2 * np.linspace(2800, 600, 2)),
                kappa_from_lambda(2 * np.linspace(300, 60, 2))]
    scale_st = [['-1', '1/3'], ['2/3', '1/3']]
    scale_mg = [[6.5e-7, 3.e-4], [1.6e-3, 0.36e-3]]

    y_label = r'Vertical kinetic energy $[m^2/s^2]$'
    y_limits = [1e-6, 1e-3]

    fig, axes = spectra_base_figure(n_rows=1, n_cols=len(p_levels), y_limits=y_limits,
                                    y_label=y_label, y_scale='log', ax_titles=ax_titles,
                                    lambda_lines=[60, ], frame=True, truncation='n1024')

    for i, ax in enumerate(axes.ravel()):
        ax.plot(kappa, vke_icon[i], lw=1.5, color='k',
                label="ICON", linestyle='solid', alpha=1.0)

        ax.plot(kappa, 2 * vke_sps[i], lw=1.5, color='blue',
                label="IFS scaled by 2",
                linestyle='dashed', alpha=1.0)

        ax.plot(kappa, vke_sps[i], lw=1.5, color='red',
                label=r"IFS from $w$ coefficients",
                linestyle='solid', alpha=1.0)

        # plot reference slopes
        level_id = p_levels[i] < 500e2
        reference_slopes(ax, x_scales, scale_mg[level_id], scale_st[level_id])

    axes[0].legend(loc='lower left')

    plt.show()
