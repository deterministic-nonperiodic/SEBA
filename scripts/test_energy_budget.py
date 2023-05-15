import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from AtmosphericEnergyBudget import EnergyBudget
from _spherepack import onedtotwod
from sklearn import preprocessing

from src.spectral_analysis import spectrum
from src.tools import kappa_from_deg


def create_fractal_noise(samples, ny, nx, slopes=None, amplitude=1.0):
    """Generate fractal noise with prescribed spectral slopes
    """

    # np.random.seed(117)  # fixed seed for reproduction

    if slopes is None:
        slopes = [-1, ]

    rfield = np.random.rand(samples, ny, nx)
    spec2_d = np.fft.fft2(rfield, axes=(-1, -2))

    slopes = np.array(slopes) + np.sign(slopes)

    kx = np.fft.fftfreq(nx, 1.0)
    ky = np.fft.fftfreq(ny, 1.0)

    k = ky if ky.size < kx.size else kx
    k = k[k > 0]
    k_rad = np.sqrt(ky[:, np.newaxis] ** 2 + kx[np.newaxis, :] ** 2)

    regimes = np.logspace(-2, np.log10(0.35 * k.max()), len(slopes))

    kbins = [0]
    for regime in regimes:
        kbins.append(regime * k.max())

    sbins = []
    for klb, kub in zip(kbins[:-1], kbins[1:]):
        sbins.append(np.logical_and(k_rad > klb, k_rad <= kub))
    sbins.append(k_rad > kbins[-1])

    # Generate 2D field from power spectrum
    amp = np.ones_like(rfield)

    for i in range(1, len(sbins)):
        amp[:, sbins[i]] = k_rad[sbins[i]] ** slopes[i - 1]
        scale = amp[:, sbins[i]].max() / amp[:, sbins[i - 1]].min()
        amp[:, sbins[i]] /= scale

    amp[:, k_rad == 0.] = amp.max()
    spec2_d *= amplitude * np.sqrt(amp)

    return np.abs(np.fft.ifft2(spec2_d, axes=(-1, -2)))


def generate_atmospheric_state(samples=5, levels=20, spatial_dims=(256, 512)):
    dataset = {}
    variable_names = ['u', 'v', 'w', 't']
    spectral_slopes = [(-5 / 3,), (-5 / 3,), (-1 / 3,), (-5 / 3,)]

    dataset['p'] = np.linspace(1000e2, 100e2, levels)  # Pa

    # Create synthetic atmospheric data: fractal noise with predefined slopes.
    for ivar, slopes in zip(variable_names, spectral_slopes):
        dataset[ivar] = create_fractal_noise(
            samples * levels, *spatial_dims, slopes=slopes).reshape(
            (samples, levels) + spatial_dims
        )

    scaler = preprocessing.MinMaxScaler(feature_range=(-100.0, 100.0))
    for ivar in ['u', 'v']:
        dataset[ivar] = scaler.fit_transform(dataset[ivar].reshape((-1, 1)))
        dataset[ivar] = np.reshape(dataset[ivar], (samples, levels) + spatial_dims)
        dataset[ivar] = np.moveaxis(np.linspace(0.0, 1.0, levels) * np.moveaxis(dataset[ivar], 1, -1), -1, 1)

    # scale variables to realistic values
    scaler = preprocessing.MinMaxScaler(feature_range=(-2.0e-2, 2.0e-2))
    dataset['w'] = scaler.fit_transform(dataset['w'].reshape((-1, 1)))
    dataset['w'] = np.reshape(dataset['w'], (samples, levels) + spatial_dims)

    scaler = preprocessing.MinMaxScaler(feature_range=(210.0, 288.0))
    dataset['t'] = np.moveaxis(np.linspace(1.0, 0.0, levels) * np.moveaxis(dataset['t'], 1, -1), -1, 1)
    dataset['t'] = scaler.fit_transform(dataset['t'].reshape((-1, 1)))
    dataset['t'] = np.reshape(dataset['t'], (samples, levels) + spatial_dims)

    return dataset


if __name__ == '__main__':
    # number of realizations
    nsample = 5
    nlevels = 19

    # spatial shape of the generated data
    data_shape = (256, 512)

    # Generate synthetic atmospheric data
    dset = generate_atmospheric_state(samples=nsample, levels=nlevels, spatial_dims=data_shape)

    # load earth topography and surface pressure
    ghsl = xr.open_dataset('../data/DYAMOND2_topography_n128.nc').topography_c.values
    sfcp = xr.open_dataset('data/DYAMOND2_sfcp_n128.nc').pres_sfc.values.squeeze()

    # Create energy budget object
    AEB = EnergyBudget(dset['u'], dset['v'], dset['w'], dset['t'], dset['p'],
                       ps=sfcp, ghsl=ghsl, leveltype='pressure', gridtype='gaussian',
                       truncation=None, legfunc='stored', axes=(1, 2, 3), sample_axis=0)

    # Testing AEB
    Tk = AEB.hke_nonlinear_transfer()
    Ta = AEB.ape_nonlinear_transfer()
    Cka = AEB.energy_conversion()

    Ek = AEB.horizontal_kinetic_energy()
    Ea = AEB.available_potential_energy()

    nlat = AEB.nlat

    # Kinetic energy in vector form accumulate and integrate vertically
    Ek_p = AEB.vertical_integration(Ek, pressure_range=None)
    Ek_l = Ek_p.mean(-1)  # average over samples

    Ea_p = AEB.vertical_integration(Ea, pressure_range=None)
    Ea_l = Ea_p.mean(-1)  # average over samples

    # Kinetic energy in scalar form for reference
    u_ml = np.asarray(onedtotwod(AEB._spectral_transform(AEB.u), nlat))
    v_ml = np.asarray(onedtotwod(AEB._spectral_transform(AEB.v), nlat))

    Ek_s = spectrum(u_ml, convention='power') + spectrum(v_ml, convention='power')
    Ek_s *= 0.5  # * np.atleast_2d(AEB.degrees).T  #  where is this l factor coming from? No clue!

    Ek_s = AEB.vertical_integration(Ek_s, pressure_range=None).mean(-1)  # average over samples

    kappa = 1e3 * kappa_from_deg(np.arange(nlat))  # km^-1

    plt.loglog(kappa[1:-1], Ek_s[1:-1], label='Ek scalar [m,l]', linewidth=1.5, linestyle='-', color='g')
    plt.loglog(kappa[1:-1], Ek_l[1:-1], label='Ek vector form', linewidth=1.5, linestyle='-', color='b')
    plt.loglog(kappa[1:-1], Ea_l[1:-1], label='Ea', linewidth=1.5, linestyle='-', color='red')

    slope_53 = kappa[1:] ** (-5 / 3)
    slope_53 *= Ek_l.max() / np.nanmax(slope_53)
    slope_31 = kappa[1:] ** (-3)
    slope_31 *= Ek_l.max() / np.nanmax(slope_31)

    plt.loglog(kappa[1:], 1.e-1 * slope_53, linewidth=1., linestyle='--', color='gray', alpha=0.8, label=r'$-5/3$')
    plt.loglog(kappa[1:], 0.1e2 * slope_31, linewidth=1., linestyle='--', color='black', alpha=0.8, label=r'$-3$')
    plt.legend()
    plt.show()

    # Nonlinear transfer of Kinetic energy and Available potential energy
    prange = [100e2, 950e2]

    Tk_p = AEB.vertical_integration(Tk, pressure_range=prange).mean(-1)
    Ta_p = AEB.vertical_integration(Ta, pressure_range=prange).mean(-1)
    Cka_l = AEB.vertical_integration(Cka, pressure_range=prange).mean(-1)

    # Accumulate
    Tk_l = np.sum(Tk_p) - np.cumsum(Tk_p)
    Ta_l = np.sum(Ta_p) - np.cumsum(Ta_p)

    plt.semilogx(kappa[1:-1], Tk_l[1:-1], label='KE nonlinear transfer', linewidth=1.5, linestyle='-', color='k')
    plt.semilogx(kappa[1:-1], Ta_l[1:-1], label='APE nonlinear transfer', linewidth=1.5, linestyle='--', color='g')
    plt.semilogx(kappa[1:-1], Cka_l[1:-1], label='Energy conversion', linewidth=1.5, linestyle='--', color='r')
    plt.legend()
    plt.show()
