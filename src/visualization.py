import os
import warnings
from functools import reduce

import cartopy.crs as ccrs
import colorcet
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import SymLogNorm
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import ScalarFormatter, FixedLocator
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import MinMaxScaler

from spectral_analysis import kappa_from_deg, kappa_from_lambda
from tools import find_intersections

# from scipy.signal import find_peaks

warnings.filterwarnings('ignore')
plt.style.use('default')

params = {'xtick.labelsize': 'medium',
          'ytick.labelsize': 'medium',
          'text.usetex': True, 'font.size': 14,
          'legend.title_fontsize': 15,
          'font.family': 'serif', 'font.weight': 'normal'}

plt.rcParams.update(params)
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab10.colors)
plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

# global variables
color_sequences = {
    "models": {
        4: ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', ],
        8: ['#377eb8', '#e41a1c', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf'],
    },
    "mint": {
        2: ["#e4f1e1", "#0d585f"],
        3: ["#e4f1e1", "#63a6a0", "#0d585f"],
        4: ["#e4f1e1", "#89c0b6", "#448c8a", "#0d585f"],
        5: ["#E4F1E1", "#9CCDC1", "#63A6A0", "#337F7F", "#0D585F"],
        6: ["#E4F1E1", "#abd4c7", "#7ab5ad", "#509693", "#2c7778", "#0d585f"],
        7: ["#e4f1e1", "#b4d9cc", "#89c0b6", "#63a6a0", "#448c8a", "#287274", "#0d585f"]
    },
    "DarkMint": {
        2: ("#d2fbd4", "#123f5a"),
        3: ("#d2fbd4", "#559c9e", "#123f5a"),
        4: ("#d2fbd4", "#7bbcb0", "#3a7c89", "#123f5a"),
        5: ("#d2fbd4", "#8eccb9", "#559c9e", "#2b6c7f", "#123f5a"),
        6: ("#d2fbd4", "#9cd5be", "#6cafa9", "#458892", "#266377", "#123f5a"),
        7: ("#d2fbd4", "#a5dbc2", "#7bbcb0", "#559c9e", "#3a7c89", "#235d72", "#123f5a")
    },
}

cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905],
           [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143],
           [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952,
                                                  0.779247619],
           [0.1252714286, 0.3242428571, 0.8302714286],
           [0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238,
                                                        0.8819571429],
           [0.0059571429, 0.4086142857, 0.8828428571],
           [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571,
                                                  0.8719571429],
           [0.0498142857, 0.4585714286, 0.8640571429],
           [0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667,
                                                        0.8467],
           [0.0779428571, 0.5039857143, 0.8383714286],
           [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571,
                                                       0.8262714286],
           [0.0640571429, 0.5569857143, 0.8239571429],
           [0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524,
                                                        0.819852381], [0.0265, 0.6137, 0.8135],
           [0.0238904762, 0.6286619048,
            0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667],
           [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381,
                                                        0.7607190476],
           [0.0383714286, 0.6742714286, 0.743552381],
           [0.0589714286, 0.6837571429, 0.7253857143],
           [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429],
           [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429,
                                                        0.6424333333],
           [0.2178285714, 0.7250428571, 0.6192619048],
           [0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619,
                                                        0.5711857143],
           [0.3481666667, 0.7424333333, 0.5472666667],
           [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524,
                                                  0.5033142857],
           [0.4871238095, 0.7490619048, 0.4839761905],
           [0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476,
                                                        0.4493904762],
           [0.609852381, 0.7473142857, 0.4336857143],
           [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333],
           [0.7184095238, 0.7411333333, 0.3904761905],
           [0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667,
                                                  0.3632714286],
           [0.8185047619, 0.7327333333, 0.3497904762],
           [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217],
           [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857,
                                                        0.2886428571],
           [0.9738952381, 0.7313952381, 0.266647619],
           [0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857,
                                                       0.2164142857],
           [0.9955333333, 0.7860571429, 0.196652381],
           [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857],
           [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309],
           [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333,
                                                  0.0948380952],
           [0.9661, 0.9514428571, 0.0755333333],
           [0.9763, 0.9831, 0.0538]]

parula = LinearSegmentedColormap.from_list('parula', cm_data)

BWG = LinearSegmentedColormap.from_list('BWG', colorcet.CET_D13)

VARIABLE_KEYMAP = {
    'hke': r'$E_K$',
    'rke': r'$E_R$',
    'dke': r'$E_D$',
    'vke': r'$E_w$',
    'ape': r'$E_A$',
    'pi_hke': r'$\Pi_K$',
    'pi_dke': r'$\Pi_D$',
    'pi_rke': r'$\Pi_R$',
    'pi_ape': r'$\Pi_A$',
    'cdr': r'$\mathcal{C}_{D \rightarrow R}$',
    'cdr_w': r'$\mathcal{C}_{D \rightarrow R}^{\omega}$',
    'cdr_v': r'$\mathcal{C}_{D \rightarrow R}^{\zeta}$',
    'cdr_c': r'$\mathcal{C}_{D \rightarrow R}^{f}$',
    'cad': r'$\mathcal{C}_{A \rightarrow D}$',
    'vf_dke': r'$\mathcal{F}_{D\uparrow}$',
    'vfd_dke': r'$\partial_{p}\mathcal{F}_{D\uparrow}$',
    'vfd_tot': r'$\mathcal{F}_{\uparrow}(p_b) - \mathcal{F}_{\uparrow}(p_t)$',
    'vf_ape': r'$\mathcal{F}_{A\uparrow}$',
    'vfd_ape': r'$\partial_{p}\mathcal{F}_{A\uparrow}$',
    'uw_vf': r'$\rho\overline{u^{\prime}w^{\prime}}$',
    'vw_vf': r'$\rho\overline{v^{\prime}w^{\prime}}$',
    'gw_vf': r'$\overline{u^{\prime}w^{\prime}}$ + $\overline{v^{\prime}w^{\prime}}$',
    'dke_dl_vf': r'$\partial_{\kappa}(\partial_{p}\mathcal{F}_{D\uparrow})$',
    'dcdr_dl': r'$\partial_{\kappa} \mathcal{C}_{D \rightarrow R}$',
    'dis_rke': r'$\mathcal{D}_R$',
    'dis_dke': r'$\mathcal{D}_D$',
    'dis_hke': r'$\mathcal{D}_K$'
}

LINES_KEYMAP = {
    'hke': ('black', 'solid', 1.5),
    'rke': ('red', 'dashed', 1.5),
    'dke': ('green', 'dashed', 1.5),
    'vke': ('black', '-.', 1.5),
    'ape': ('navy', 'solid', 1.5),
    'pi_hke': ('red', 'solid', 2.0),
    'pi_dke': ('green', 'solid', 1.6),
    'pi_rke': ('red', 'solid', 1.6),
    'pi_ape': ('navy', 'solid', 2.0),
    'cdr': ('cyan', 'dashed', 2.0),
    'cdr_w': ('black', '-.', 1.6),
    'cdr_v': ('red', '-.', 1.6),
    'cdr_c': ('green', '-.', 1.6),
    'cad': ('green', 'dashed', 2.0),
    'vfd_dke': ('magenta', '-.', 1.6),
    'vfd_tot': ('magenta', '-.', 1.6),
    'vfd_ape': ('blue', '-.', 1.6),
    'dis_rke': ('blue', '-.', 1.6),
    'dis_dke': ('blue', '-.', 1.6),
    'dis_hke': ('blue', '-.', 1.6),
}

crs = ccrs.PlateCarree()

locations = {
    'southpole': (-90, 140.7),
    'northpole': (90, -57),
    'northincl': (65, -75.),
    'goes': (10, -65.),
}

name_conv_models = {
    'nwp2.5': 'icon 2.5 km',
    'ifs4': 'ifs 4 km',
    'geos3': 'geos 3 km',
    'nicam3': 'nicam 3 km',
}

comp_names = {
    'RO': 'ROS',
    'IG': 'IGW'
}


def wind_speed(u, v):
    return np.sqrt(u * u + v * v)


def normalize(x, min_x=-1, max_x=1):
    return min_x + (max_x - min_x) * (x - np.nanmin(x)) / x.ptp()


def global_map(lon, lat, data, wind=None, titles=None, data_name='', time_str='', units='',
               data_range=None,
               cmap='viridis', central_proj=None, gridtype='regular'):
    shape = np.shape(data)

    if titles is None:
        titles = shape[0] * ['']

    if central_proj is None:
        central_proj = (90, 10.0)

    # Geostationary extent
    central_lat, central_lon = central_proj  # 45

    semi_major = 6378137.0
    semi_minor = 6356752.31414

    globe = ccrs.Globe(semimajor_axis=semi_major,
                       semiminor_axis=semi_minor,
                       flattening=1.0e-30)

    projection = ccrs.Orthographic(central_longitude=central_lon,
                                   central_latitude=central_lat,
                                   globe=globe)

    crs_map = ccrs.PlateCarree(central_longitude=180, globe=globe)
    crs_wind = ccrs.RotatedPole(pole_latitude=90, pole_longitude=180)

    # 0.6 * 15. / 0.75
    fig = plt.figure(figsize=(10.2, 6.), constrained_layout=True)

    if data_range is None:
        data_range = {key: np.percentile(value, [1., 99.])
                      for key, value in data.items()}

    # -------------------------------------------------------------------------
    # Using the Orthographic projection centered at the poles
    # -------------------------------------------------------------------------
    indicator = 'abcdefghijklmn'
    for i, (component, var) in enumerate(data.items()):

        ax_id = int('1' + str(len(data)) + str(i + 1))

        ax = fig.add_subplot(ax_id, projection=projection)

        at = AnchoredText(comp_names[component], prop=dict(size=15),
                          frameon=False, loc='upper left', borderpad=1.)
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at)

        at = AnchoredText("({})".format(indicator[i]), prop=dict(size=16),
                          frameon=False, loc='upper right', borderpad=1.)
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at)

        gl = ax.gridlines(ccrs.PlateCarree(),
                          draw_labels=False,
                          linestyle="--", linewidth=0.2,
                          color='gray', zorder=2)
        # Manipulate latitude and longitude grid-line numbers and spacing
        gl.ylocator = FixedLocator(np.arange(-90, 90, 30))
        gl.xlocator = FixedLocator(np.arange(-180, 180, 30))

        # Plot the image
        if gridtype == 'irregular':

            img = ax.tricontourf(lon, lat, var, levels=14, alpha=0.9,
                                 vmin=data_range[component][0], vmax=data_range[component][1],
                                 transform=crs_map, interpolation='bilinear', cmap=cmap)

        else:

            img = ax.imshow(var, vmin=data_range[component][0], vmax=data_range[component][1],
                            transform=crs_map, interpolation='bilinear',
                            origin='upper', cmap=cmap, alpha=0.9)

        cb = plt.colorbar(img, ax=ax, orientation='horizontal',
                          aspect=10, pad=0.015, fraction=0.15, shrink=0.75)
        cb.ax.tick_params(labelsize=9)
        cb.ax.set_title(data_name + ' ' + units, fontsize=12)

        # add streamplot of wind vector
        if component == 'RO' and (wind is not None):
            u, v = wind[component]
            ax.streamplot(lon, lat, u, v, linewidth=0.28, arrowsize=0.6,
                          density=2.25, color='w', transform=crs_wind)

        # Add coastlines, borders and gridlines
        ax.coastlines(resolution='50m', color='k', linewidth=0.25, alpha=0.6)

        ax_title = ' '.join([titles, 4 * ' ', time_str, 'UTC'])
        at = AnchoredText(ax_title, prop=dict(size=14),
                          frameon=False, loc='upper center', borderpad=-2.)
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at)

    plt.show()
    return fig


def generate_global_maps(model,
                         variable=None, components=None, level=None, skip=1,
                         location='southpole', time_range=None):
    if time_range is None:
        time_range = [None, ]

    if variable is None:
        variable = 'wind'

    if components is None:
        components = ['RO', 'IG']

    variable_names = {
        'wind': 'Horizontal wind speed',
        'ua': 'Zonal wind',
        'va': 'Meridional wind',
        'wa': 'Vertical velocity',
        'ta': 'Temperature'
    }
    variable_units = {'wind': r'(m/s)', 'ua': r'(m/s)', 'va': r'(m/s)',
                      'wa': r'($\times 10^{-2}$ m/s)', 'ta': r'(K)'}

    fig_title = '{}  {} km'.format(model, int(level))

    time_id = slice(None)

    model_data = {}
    model_wind = {}
    timestamps = {}
    coordinates = {}
    for component in components:

        print('Retrieving data from ', component)

        base_path = '/media/yanm/Data/DYAMOND/data/'
        file_names = os.path.join(base_path, '{}_{}_wind_202002*_n256.nc'.format(model, component))

        model_dataset = xr.open_mfdataset(file_names)
        model_dataset = model_dataset.sel(time=slice(*time_range))

        # Create a list with timestamps
        timestamps[component] = model_dataset.time.dt.strftime('%Y-%m-%d %H:%M')

        lat = model_dataset['lat'].values
        lon = model_dataset['lon'].values
        height = 1.0e-3 * model_dataset['height'].values

        coordinates[component] = (lat, lon, height)

        level_id = np.argmin(abs(height - level))
        dims = (time_id, slice(level_id, level_id + 1)) + 2 * (slice(None, None, skip),)

        u = model_dataset['ua'].values[dims]
        v = model_dataset['va'].values[dims]

        model_wind[component] = np.stack([u, v])

        # create dataset
        if variable == 'wind':
            model_data[component] = wind_speed(u, v)
        elif variable in ['ua', 'va', 'wa']:
            model_data[component] = model_dataset[variable].values[dims]
            if variable == 'wa':
                model_data[component] *= 1e2
        else:
            raise ValueError('Wrong variable name')

    if variable == 'wind':
        colormap = BWG  # 'rainbow' # CET_L20
    elif variable in ['ua', 'va', 'wa']:
        colormap = BWG
    else:
        colormap = parula

    data_range = {key: np.percentile(value, [1., 99.]) for key, value in model_data.items()}

    # Create maps
    for i, timestamp in enumerate(timestamps['IG']):
        wind = {'RO': model_wind['RO'][:, i].squeeze()}
        data = {c: model_data[c][i].squeeze() for c in components}

        lat, lon, height = coordinates['IG']

        fig = global_map(lon, lat, data, wind=wind, data_range=data_range,
                         titles=fig_title, data_name=variable_names[variable],
                         units=variable_units[variable],
                         time_str=timestamp, cmap=colormap,
                         central_proj=locations[location])

        fig.savefig(
            'figures/animation/{}_{}_{}km_{}_comparison_{:04d}.pdf'.format(
                model, variable, int(level), timestamp.split(' ')[0], i + 45), dpi=300)
        plt.close(fig)


def find_symlog_params(data):
    """
    Determines appropriate values of linscale and linthresh for use with a SymLogNorm color map.

    Parameters:
        data (ndarray): The data to be plotted.

    Returns:
        tuple: A tuple of two floats representing the values of linscale and linthresh.
    """

    # Compute the range of the data
    data_range = np.ptp(data)

    # Scale the data to the range [0, 1] using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))

    # Find the value of linthresh that separates the linear and logarithmic parts
    # We want the linear part to be as large as possible while still excluding the
    # top and bottom 5% of the data, which could skew the normalization
    linthresh = np.nanpercentile(scaled_data, [5, 95])
    linthresh = np.interp(0.5, [0, 1], linthresh)  # use the median value

    # Compute the scale factor for the logarithmic part of the normalization
    # We want the logarithmic part to span a decade or so
    if 0.0 <= data_range <= 1.0:
        linscale = 0.9 * data_range
    else:
        # Find the value of linscale that stretches the linear part to cover most of the range
        # We want the linear part to cover about half of the range
        qr_5 = np.nanpercentile(data, [5, 95]).ptp()  # / 2.0
        linscale = qr_5 / (linthresh * np.log10(data_range))

    abs_max = 0.65 * np.nanmax(abs(data))

    v_min = 0.0 if np.nanmin(data) > 0 else -abs_max
    v_max = 0.0 if np.nanmax(data) < 0 else abs_max

    return dict(linthresh=linthresh, linscale=linscale, vmin=v_min, vmax=v_max)


def mean_confidence_interval(data, confidence=0.95, axis=0):
    a = np.asanyarray(data)
    n = a.shape[axis]

    m, se = np.nanmean(a, axis=axis), stats.sem(a, axis=axis)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def transform_spectra(s):
    return mean_confidence_interval(np.reshape(s, (-1, s.shape[-1])), confidence=0.95, axis=0)


def spectra_base_figure(n_rows=1, n_cols=1, x_limits=None, y_limits=None, y_label=None,
                        lambda_lines=None, y_scale='log', base=10, ax_titles=None, aligned=True,
                        frame=True, truncation='n1024', figure_size=None, shared_ticks=True,
                        **figure_kwargs):
    """Creates a figure template for spectral line plots

    :param n_rows: number of rows in figure
    :param n_cols: number of columns in figure
    :param x_limits: limits of the horizontal axis
    :param y_limits: limits of the vertical axis
    :param y_label: label of the vertical axis
    :param y_scale: 'linear', 'log', 'symlog'
    :param base: base of the logarithmic axis
    :param frame: draw frame on ax titles
    :param lambda_lines: positions to draw vertical lines
    :param ax_titles: title to display on the left corner of each axis
    :param aligned:
    :param truncation: spectral truncation
    :param figure_size:
    :param shared_ticks: always share y/x ticks for multiple subplots.
    :param figure_kwargs: additional keyword arguments to pass to plt.figure()
    :return: fig, axes
    """
    if y_limits is None:
        y_limits = [1e-10, 1e2]

    if 'm' in truncation:
        prefix = 'Vertical'
        if x_limits is None:
            x_limits = kappa_from_lambda(np.array([40, 1.5]))
        xticks = np.array([1, 10, 100, 1000])

        scale_str = '{:.1f}km'
    else:
        scale_str = '{:d}km'
        if truncation == 'n1024':
            prefix = ''
            if x_limits is None:
                x_limits = kappa_from_lambda(np.array([40e3, 40]))
            xticks = np.array([1, 10, 100, 1000])
        else:
            prefix = ''
            if x_limits is None:
                x_limits = kappa_from_lambda(np.array([40e3, 20]))
            xticks = np.array([2, 20, 200, 2000])

    show_limit = True
    if lambda_lines is None:
        lambda_lines = [18., ]
        show_limit = False

    kappa_lines = kappa_from_lambda(np.asarray(lambda_lines))

    if ax_titles is None:
        pass
    elif isinstance(ax_titles, str):
        ax_titles = n_rows * n_cols * [ax_titles, ]
    else:
        ax_titles = list(ax_titles)

    if figure_size is None:
        figure_size = (5.5, 5.5)
    elif isinstance(figure_size, (tuple, list)):
        pass
    else:
        reduced_size = 0.92 * float(figure_size)
        aspect = reduced_size if n_cols > 2 else figure_size
        figure_size = (aspect * n_cols, figure_size * n_rows)

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
                             figsize=figure_size,
                             constrained_layout=True, **figure_kwargs)

    if n_rows * n_cols == 1:
        axes = np.array([axes, ])

    for m, ax in enumerate(axes.flatten()):

        # axes frame limits
        ax.set_xscale('log', base=base)
        ax.set_yscale(y_scale)

        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)

        # axes title as annotation
        if ax_titles is not None:
            at = AnchoredText(ax_titles[m], prop=dict(size=15), frameon=frame, loc='upper right')
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax.add_artist(at)

        # draw vertical reference lines
        if show_limit:

            for lambda_line in lambda_lines:
                kappa_line = kappa_from_lambda(lambda_line)
                ax.axvline(x=kappa_line, ymin=0, ymax=1,
                           color='gray', linewidth=0.8,
                           alpha=0.6, linestyle='solid')

                ax.annotate(scale_str.format(int(lambda_line)),
                            xy=(0.90, 0.98), xycoords='axes fraction',
                            color='black', fontsize=10, horizontalalignment='left',
                            verticalalignment='top')

            ax.axvspan(np.max(kappa_lines), x_limits[1], alpha=0.15, color='gray')

        # Align left ytick labels:
        if aligned:
            for label in ax.yaxis.get_ticklabels():
                label.set_horizontalalignment('left')
            ax.yaxis.set_tick_params(pad=30)

        # set y label only for left most panels
        if not (m % n_cols):
            ax.set_ylabel(y_label, fontsize=15)
        else:
            if shared_ticks:
                ax.axes.get_yaxis().set_visible(False)

        if m >= n_cols * (n_rows - 1):  # lower boundary only
            ax.set_xlabel('{} wavenumber'.format(prefix), fontsize=14, labelpad=4)
        else:
            if shared_ticks:
                ax.axes.get_xaxis().set_visible(False)

        # set lower x ticks
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xticks(1e3 * kappa_from_deg(xticks))
        ax.set_xticklabels(xticks)

        if m < n_cols:  # upper boundary only
            secax = ax.secondary_xaxis('top', functions=(kappa_from_lambda, kappa_from_lambda))
            secax.xaxis.set_major_formatter(ScalarFormatter())

            secax.set_xlabel('{} wavelength '.format(prefix) + r'/ $km$', fontsize=14, labelpad=5)

    return fig, axes


def reference_slopes(ax, k_scales, magnitude, slopes, name='horizontal'):
    # Function for adding reference slopes to spectral plot

    if name == 'horizontal':
        s_scales = [r"$l^{{{0}}}$".format(s) for s in slopes]
    else:
        s_scales = [r"$m^{{{0}}}$".format(s) for s in slopes]

    n_slopes = [float(reduce(lambda x, y: float(x) / float(y), s.split('/')))
                for s in slopes]
    # Plot reference slopes
    for k, m, sn, ss in zip(k_scales, magnitude, n_slopes, s_scales):
        y_scale = m * k ** sn
        ax.plot(k, y_scale, lw=1.2, ls='dashed', color='gray')

        scale_pos = np.argmax(y_scale)

        x_scale_pos = k[scale_pos]
        y_scale_pos = y_scale[scale_pos]
        x_text_pos = -10. if scale_pos == y_scale.size else -4

        ax.annotate(ss, xy=(x_scale_pos, y_scale_pos), xycoords='data',
                    xytext=(x_text_pos, 16.), textcoords='offset points',
                    color='k', horizontalalignment='left',
                    verticalalignment='top', fontsize=13)


def _parse_variable(varname):
    # parser variable names and return the corresponding line properties

    if varname in VARIABLE_KEYMAP:
        color, style, width = LINES_KEYMAP[varname]
        label = VARIABLE_KEYMAP[varname]
    elif "+" in varname:
        color, style, width = 'black', 'solid', 2
        label = VARIABLE_KEYMAP[varname.split('+')[0]].split('_')[0] + r'=$ '
        label += r' $+$ '.join([VARIABLE_KEYMAP[name] for name in varname.split('+')])
    else:
        raise ValueError(f'Unknown variable name: {varname}.')

    return dict(label=label, lw=width, color=color, linestyle=style)


def fluxes_slices_by_models(dataset, model=None, variables=None, compensate=False,
                            resolution='n1024', x_limits=None, y_limits=None,
                            cmap=None, fig_name='test.png'):
    if variables is None:
        variables = list(dataset.data_vars)

    ax_titles = [VARIABLE_KEYMAP[name] for name in variables]

    if y_limits is None:
        y_limits = [1000., 100.]

    if cmap is None:
        cmap = BWG

    if compensate:
        y_label = r'Compensated energy ($\times\kappa^{5/3}$)'
    else:
        y_label = r'Kinetic energy $[m^2/s^2]$'

    # get coordinates
    level = 1e-2 * dataset['level']
    kappa = 1e3 * dataset['kappa']

    if 'time' in dataset.dims:
        dataset = dataset.mean(dim='time')

    # -----------------------------------------------------------------------------------
    # Visualization of Kinetic energy and Available potential energy
    # -----------------------------------------------------------------------------------
    n = len(variables)
    cols = 2 if not n % 2 else n
    rows = max(1, n // cols)

    fig, axes = spectra_base_figure(n_rows=rows, n_cols=cols, x_limits=x_limits,
                                    y_limits=y_limits, figure_size=4.5,
                                    y_label=y_label, y_scale='linear', ax_titles=ax_titles,
                                    frame=True, truncation=resolution)
    axes = axes.ravel()

    cs_levels = 100

    for m, (ax, varname) in enumerate(zip(axes, variables)):
        spectra = 1e3 * dataset[varname].values

        smoothed_data = gaussian_filter(spectra, 0.25)

        # Create plots:
        cs = ax.contourf(kappa, level, smoothed_data,
                         cmap=cmap, levels=cs_levels,
                         norm=SymLogNorm(**find_symlog_params(spectra)))

        ax.contour(kappa, level, smoothed_data,
                   color='black', linewidths=0.8, linestyles='solid',
                   levels=[0, ], alpha=0.6)

        ax.set_ylim(y_limits)
        ax.set_ylabel(r'Pressure (hPa)')

        # add a colorbar to all axes
        cb = plt.colorbar(cs, ax=ax, orientation='vertical', pad=0.001, format="%.2f")
        cb.ax.set_title(r"($\times 10^{3}~W m^{-2}$)", fontsize=11, loc='center', pad=10)
        cb.ax.tick_params(labelsize=12)

    if model is not None:
        at = AnchoredText(model.upper(), prop=dict(size=15), frameon=True, loc='upper left')
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        axes[0].add_artist(at)

    plt.show()
    fig.savefig(fig_name, dpi=300)
    plt.close(fig)


def fluxes_spectra_by_levels(dataset, model=None, variables=None, layers=None,
                             resolution='n1024', x_limits=None, y_limits=None,
                             fig_name=None):
    if model is None:
        model = ''

    if variables is None:
        variables = list(dataset.data_vars)

    if layers is None:
        layers = {'': [20e2, 950e2]}

    if y_limits is None:
        y_limits = {name: [-1.5, 1.5] for name in layers.keys()}

    # get coordinates
    kappa = 1e3 * dataset['kappa']

    # -----------------------------------------------------------------------------------
    # Visualization of Kinetic energy and Available potential energy
    # -----------------------------------------------------------------------------------
    n = len(layers)
    cols = 2 if not n % 2 else n
    rows = max(1, n // cols)

    legend_cols = 1 + int(len(variables) > 6)

    fig, axes = spectra_base_figure(n_rows=rows, n_cols=cols,
                                    figure_size=(cols * 7.0, rows * 5.8),
                                    x_limits=x_limits, y_limits=None, aligned=False,
                                    y_label=r'Cumulative energy flux / $W ~ m^{-2}$',
                                    y_scale='linear', ax_titles=None, frame=False,
                                    truncation=resolution, shared_ticks=False)
    axes = axes.ravel()
    indicator = 'abcdefghijklmn'

    for m, (level, prange) in enumerate(layers.items()):

        data = dataset.integrate_range(coord_range=prange).mean(dim='time')

        for varname in variables:
            spectra = np.sum([data[name] for name in varname.split('+')], axis=0)

            axes[m].plot(kappa, spectra, **_parse_variable(varname))

        axes[m].axhline(y=0.0, xmin=0, xmax=1, color='gray',
                        linewidth=1.2, linestyle='dashed', alpha=0.5)

        axes[m].set_ylim(*y_limits[level])

        prange_str = [int(1e-2 * p) for p in sorted(prange)[::-1]]
        axes[m].legend(title=r"{} ({:4d} - {:4d} hPa)".format(level, *prange_str),
                       loc='upper right', fontsize=14, ncol=legend_cols)

        art = AnchoredText(f"({indicator[m]})",
                           loc='lower left', prop=dict(size=20), frameon=False,
                           bbox_to_anchor=(-0.1, 1.0), bbox_transform=axes[m].transAxes)
        art.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        axes[m].add_artist(art)

    art = AnchoredText(model.upper(), prop=dict(size=20), frameon=False, loc='lower left',
                       bbox_to_anchor=(-0.02, 0.865), bbox_transform=axes[0].transAxes)
    art.patch.set_boxstyle("round,pad=-0.3, rounding_size=0.2")
    axes[0].add_artist(art)

    plt.show()

    if fig_name is not None:
        fig.savefig(fig_name, dpi=300)

    plt.close(fig)


def energy_spectra_by_levels(dataset, model=None, variables=None, layers=None,
                             resolution='n1024', x_limits=None, y_limits=None,
                             fig_name=None):
    if model is None:
        model = ''

    if variables is None:
        variables = ['hke', 'rke', 'dke', 'vke', 'ape']

    if layers is None:
        layers = {'': [20e2, 950e2]}

    if y_limits is None:
        y_limits = {name: [1e-5, 5e7] for name in layers.keys()}

    if isinstance(resolution, str):
        mesoscale_limits = 1e3 * kappa_from_deg([100, float(resolution.split('n')[-1]) / 2.0])
    else:
        mesoscale_limits = kappa_from_lambda(np.linspace(480, 100, 2))

    x_scales = [kappa_from_lambda(np.linspace(3500, 600, 2)), mesoscale_limits]
    scale_st = ['-3', '-5/3']
    scale_mg = [2.5e-4, 0.1]

    # get coordinates
    kappa = 1e3 * dataset['kappa']

    # -----------------------------------------------------------------------------------
    # Visualization of Kinetic energy and Available potential energy
    # -----------------------------------------------------------------------------------
    n = len(layers)
    cols = 2 if not n % 2 else n
    rows = max(1, n // cols)

    legend_cols = 1 + int(len(variables) > 3)

    fig, axes = spectra_base_figure(n_rows=rows, n_cols=cols, shared_ticks=False,
                                    figure_size=(cols * 6.25, rows * 5.8),
                                    x_limits=x_limits, y_limits=None, aligned=True,
                                    y_label=r'Energy / $J ~ m^{-2}$',
                                    y_scale='log', ax_titles=None,
                                    frame=False, truncation=resolution)

    axes = axes.ravel()
    indicator = 'abcdefghijklmn'

    for m, (level, prange) in enumerate(layers.items()):

        data = dataset.integrate_range(coord_range=prange)

        for varname in variables:
            # compute mean and standard deviation
            spectra, spectra_sel, spectra_seu = mean_confidence_interval(data[varname].values,
                                                                         confidence=0.95, axis=0)

            # plot confidence interval
            axes[m].fill_between(kappa, spectra_seu, spectra_sel,
                                 color='gray', interpolate=False, alpha=0.1)

            axes[m].plot(kappa, spectra, **_parse_variable(varname))

        # crossing scales: intersections between rotational and divergent kinetic energies
        kappa_c, spectra_c = find_intersections(kappa.values,
                                                data['rke'].mean('time').values,
                                                data['dke'].mean('time').values,
                                                direction='decreasing')

        # take median of multiple crossings
        kappa_c = np.median(kappa_c)
        spectra_c = np.median(spectra_c)

        # vertical lines denoting crossing scales
        if not np.isnan(kappa_c):
            axes[m].vlines(x=kappa_c, ymin=0., ymax=spectra_c,
                           color='black', linewidth=0.8,
                           linestyle='dashed', alpha=0.6)

            if kappa_c > kappa_from_lambda(40.):
                kappa_c_pos = -60
            else:
                kappa_c_pos = 4

            # Scale is defined as half-wavelength
            axes[m].annotate(r'$L_{c}$' + '~ {:d} km'.format(int(kappa_from_lambda(kappa_c))),
                             xy=(kappa_c, y_limits[level][0]), xycoords='data',
                             xytext=(kappa_c_pos, 20.), textcoords='offset points',
                             color='black', fontsize=9, horizontalalignment='left',
                             verticalalignment='top')

        # plot reference slopes
        reference_slopes(axes[m], x_scales, scale_mg, scale_st)

        axes[m].set_ylim(*y_limits[level])

        prange_str = [int(1e-2 * p) for p in sorted(prange)[::-1]]
        axes[m].legend(title=r"{} ({:4d} - {:4d} hPa)".format(level, *prange_str),
                       loc='upper right', fontsize=14, ncol=legend_cols)

        art = AnchoredText(f"({indicator[m]})",
                           loc='lower left', prop=dict(size=20), frameon=False,
                           bbox_to_anchor=(-0.14, 1.0), bbox_transform=axes[m].transAxes)
        art.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        axes[m].add_artist(art)

    art = AnchoredText(model.upper(), prop=dict(size=20), frameon=False, loc='lower left',
                       bbox_to_anchor=(-0.02, 0.865), bbox_transform=axes[0].transAxes)
    art.patch.set_boxstyle("round,pad=-0.3, rounding_size=0.2")
    axes[0].add_artist(art)

    plt.show()

    if fig_name is not None:
        fig.savefig(fig_name, dpi=300)

    plt.close(fig)
