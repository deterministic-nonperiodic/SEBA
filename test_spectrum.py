import warnings

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pyshtools as pysh

from AtmosphericEnergyBudget import EnergyBudget

params = {'xtick.labelsize': 'medium',
          'ytick.labelsize': 'medium',
          'text.usetex': False, 'font.size': 12,
          'font.family': 'serif', 'font.weight': 'normal'}
plt.rcParams.update(params)
plt.rcParams['legend.title_fontsize'] = 10

warnings.filterwarnings('ignore')


def sh_cross_spectrum(grid1, grid2=None):
    nlat, nlon, *extra_dims = grid1.shape

    grid1 = grid1.reshape((nlat, nlon, -1))
    grid2 = grid2.reshape((nlat, nlon, -1))

    clm_sqd = np.empty((nlat, grid1.shape[-1]))

    for i in range(clm_sqd.shape[-1]):
        clm_1 = pysh.SHGrid.from_array(grid1[..., i], grid='GLQ').expand()
        clm_2 = pysh.SHGrid.from_array(grid2[..., i], grid='GLQ').expand()

        clm_sqd[:, i] = clm_1.cross_spectrum(clm_2, convention='power')

    return clm_sqd.reshape((nlat,) + tuple(extra_dims))


if __name__ == '__main__':
    # Load dyamond dataset
    resolution = 'n128'
    data_path = 'data/'
    date_time = '20200126'

    dset_uvt = xr.open_dataset(data_path + 'ICON_atm_3d_inst_uvt_PL_{}_{}.nc'.format(resolution, date_time))
    dset_pwe = xr.open_dataset(data_path + 'ICON_atm_3d_inst_pwe_PL_{}_{}.nc'.format(resolution, date_time))

    # load earth topography and surface pressure
    sfcp = xr.open_dataset(data_path + 'ICON_sfcp_{}.nc'.format(resolution)).pres_sfc.values
    ghsl = xr.open_dataset(data_path + 'DYAMOND2_topography_{}.nc'.format(resolution)).topography_c.values

    # Create energy budget object
    AEB = EnergyBudget(
        dset_uvt['u'].values, dset_uvt['v'].values,
        dset_pwe['omega'].values, dset_uvt['temp'].values, dset_uvt['plev'].values,
        ps=sfcp, ghsl=ghsl, leveltype='pressure', gridtype='gaussian', truncation=None,
        legfunc='stored', axes=(1, 2, 3), sample_axis=0, filter_terrain=False,
        standard_average=True)

    # visualize profiles
    variables = ['w', 'wind', 'theta_p']
    vars_info = {
        'w': ('scalar', r'Vertical kinetic energy $(m^{2}~s^{-2})$'),
        'theta_p': ('scalar', r'${\theta^{\prime}}^{2}~(K^{2})$'),
        'wind': ('vector', r'Horizontal kinetic energy  $(m^{2}~s^{-2})$')
    }
    pressure = 1e-2 * AEB.p

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
            data_gp = AEB._global_average(data[0] ** 2 + data[1] ** 2).mean(0)
            data_sc = AEB._vector_spectra(data).sum(0).mean(0)
        else:
            data_gp = AEB._global_average(data ** 2).mean(0)
            data_sc = AEB._scalar_spectra(data).sum(0).mean(0)

        lines = ax.plot(data_gp.T, pressure, '-b', data_sc.T, pressure, '--k')
        ax.legend(lines, ['global mean', 'recovered mean'], title=vars_info[variable][1], loc='lower right')
        ax.set_ylim(1015, 60)

    plt.show()
    plt.close(fig)
