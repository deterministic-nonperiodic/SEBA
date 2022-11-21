import warnings

import matplotlib.pyplot as plt
import numpy as np
import pyshtools as pysh
import spharm
import xarray as xr

from AtmosphericEnergyBudget import EnergyBudget

params = {'xtick.labelsize': 'medium',
          'ytick.labelsize': 'medium',
          'text.usetex': False, 'font.size': 12,
          'font.family': 'serif', 'font.weight': 'normal'}
plt.rcParams['legend.title_fontsize'] = 10
plt.rcParams.update(params)

warnings.filterwarnings('ignore')


def sh_cross_spectrum(grid1, grid2):
    nlat, nlon, *extra_dims = grid1.shape

    grid1 = grid1.reshape((nlat, nlon, -1))
    grid2 = grid2.reshape((nlat, nlon, -1))

    clm_sqd = np.empty((nlat, grid1.shape[-1]))

    for ix in range(clm_sqd.shape[-1]):
        clm_1 = pysh.SHGrid.from_array(grid1[..., ix], grid='GLQ').expand()
        clm_2 = pysh.SHGrid.from_array(grid2[..., ix], grid='GLQ').expand()

        clm_sqd[:, ix] = clm_1.cross_spectrum(clm_2, convention='power')

    return clm_sqd.reshape((nlat,) + tuple(extra_dims))


if __name__ == '__main__':
    # Load dyamond dataset
    resolution = 'n128'
    data_path = 'data/'
    date_time = '20200127'
    file_names = data_path + 'ICON_atm_3d_inst_{}_PL_{}_{}.nc'

    dset_dyn = xr.merge([
        xr.open_mfdataset(file_names.format(idv, resolution, date_time))
        for idv in ['uvt', 'pwe']])

    # load earth topography and surface pressure
    dset_sfc = xr.merge([
        xr.open_dataset(data_path + 'ICON_sfcp_{}_{}.nc'.format(date_time, resolution)),
        xr.open_dataset(data_path + 'DYAMOND2_topography_{}.nc'.format(resolution))])

    sfc_hgt = dset_sfc.topography_c.values
    sfc_pres = dset_sfc.pres_sfc.values

    # Create energy budget object
    AEB = EnergyBudget(
        dset_dyn['u'].values, dset_dyn['v'].values, dset_dyn['omega'].values,
        dset_dyn['temp'].values, dset_dyn['plev'].values, ps=sfc_pres, ghsl=sfc_hgt,
        level_type='pressure', grid_type='gaussian', truncation=None, legfunc='stored',
        axes='tzyx', filter_terrain=False, jobs=None)

    # visualize profiles
    variables = ['w', 'omega', 'wind', 'theta_pbn']
    vars_info = {
        'w': ('scalar', r'Vertical kinetic energy $(m^{2}~s^{-2})$'),
        'omega': ('scalar', r'Pressure velocity $(Pa^{2}~s^{-2})$'),
        'theta_pbn': ('scalar', r'${\theta^{\prime}}^{2}~(K^{2})$'),
        'wind': ('vector', r'Horizontal kinetic energy  $(m^{2}~s^{-2})$')
    }
    pressure = 1e-2 * AEB.p
    lats, weights_gs = spharm.gaussian_lats_wts(AEB.nlat)

    weights_ln = np.cos(np.deg2rad(lats))  # np.ones_like(lats) / lats.size

    n_cols = len(variables)
    fig, axes = plt.subplots(nrows=1, ncols=n_cols, figsize=(n_cols * 5, 10.0), constrained_layout=True)

    results = {}
    # Compute spectrum of scalar variables
    for i, variable in enumerate(variables):
        ax = axes[i]
        data = AEB.__dict__[variable]

        if vars_info[variable][0] == 'vector':
            # The global average of the dot product of two vectors must equal the sum
            # of the vectors' cross-spectrum along all spherical harmonic degrees.
            data_sqd = np.sum(data ** 2, axis=0)
            data_ln = AEB.global_average(data_sqd, weights=weights_ln).mean(0)
            data_gs = AEB.global_average(data_sqd, weights=weights_gs).mean(0)
            data_sp = AEB._vector_spectra(data).sum(0).mean(0)
        else:
            data_sqd = data ** 2
            data_ln = AEB.global_average(data_sqd, weights=weights_ln).mean(0)
            data_gs = AEB.global_average(data_sqd, weights=weights_gs).mean(0)
            data_sp = AEB._scalar_spectra(data).sum(0).mean(0)

        lines = ax.plot(data_ln.T, pressure, '-r', data_gs.T, pressure, '-b', data_sp.T, pressure, '--k', lw=1.5)

        ax.legend(lines, ['global mean', 'global average', 'recovered'],
                  title=vars_info[variable][1], loc='best')
        ax.set_ylim(1015, 80)

    axes[0].set_ylabel('Pressure (hPa)')

    plt.show()
    plt.close(fig)
