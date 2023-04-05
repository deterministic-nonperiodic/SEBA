import numpy as np
import xarray as xr
from numpy.core.numeric import normalize_axis_index
from scipy.integrate import simpson

import constants as cn
from fortran_libs import numeric_tools
from io_tools import get_coordinate_names, reindex_coordinate, get_surface_elevation
from io_tools import parse_dataset, _find_coordinate, interpolate_pressure_levels
from kinematics import coriolis_parameter
from spectral_analysis import triangular_truncation, kappa_from_deg
from spherical_harmonics import Spharmt
from thermodynamics import exner_function, potential_temperature, vertical_velocity
from thermodynamics import geopotential_to_height, stability_parameter
from tools import inspect_gridtype, cumulative_flux
from tools import prepare_data, recover_data, recover_spectra
from tools import rotate_vector, broadcast_1dto, gradient_1d
from tools import terrain_mask, transform_io, get_num_cores

# declare global read-only variables
_private_vars = ['nlon', 'nlat', 'nlevels', 'gridtype']


class EnergyBudget:
    """
        Description:
        ------------
        A collection of tools to compute the Spectral Energy Budget of a dry hydrostatic
        Atmosphere (SEBA). This package is developed for application to global numerical
        simulations of General Circulation Models (GCMs). SEBA is implemented based on the
        formalism developed by Augier and Lindborg (2013) and includes the Helmholtz decomposition
        into the rotational and divergent kinetic energy contributions to the nonlinear energy
        fluxes introduced by Li et al. (2023). The Spherical Harmonic Transforms are carried out
        with the high-performance SHTns C library. The analysis supports data sampled on a
        regular (equally spaced in longitude and latitude) or gaussian (equally spaced in
        longitude, latitudes located at roots of ordinary Legendre polynomial of degree nlat)
        horizontal grids. The vertical grid can be arbitrary; if data is not sampled on
        pressure levels it is interpolated to isobaric levels before the analysis.

        References:
        -----------
        Augier, P., and E. Lindborg (2013), A new formulation of the spectral energy budget
        of the atmosphere, with application to two high-resolution general circulation models,
        J. Atmos. Sci., 70, 2293–2308, https://doi.org/10.1175/JAS-D-12-0281.1.

        Li, Z., J. Peng, and L. Zhang, 2023: Spectral Budget of Rotational and Divergent Kinetic
        Energy in Global Analyses.  J. Atmos. Sci., 80, 813–831,
        https://doi.org/10.1175/JAS-D-21-0332.1.

        Schaeffer, N. (2013). Efficient spherical harmonic transforms aimed at pseudospectral
        numerical simulations, Geochem. Geophys. Geosyst., 14, 751– 758,
        https://doi.org/10.1002/ggge.20071.
    """

    def __setattr__(self, key, val):
        """
        Prevent modification of read-only instance variables.
        """
        if key in self.__dict__ and key in _private_vars:
            raise AttributeError('Attempt to rebind read-only instance variable ' + key)
        else:
            self.__dict__[key] = val

    def __delattr__(self, key):
        """
        Prevent deletion of read-only instance variables.
        """
        if key in self.__dict__ and key in _private_vars:
            raise AttributeError('Attempt to unbind read-only instance variable ' + key)
        else:
            del self.__dict__[key]

    def __init__(self, dataset, variables=None, ps=None, p_levels=None,
                 truncation=None, rsphere=None, jobs=None):
        """
        Initializing class EnergyBudget.

        Signature
        ---------
        energy_budget =  EnergyBudget(dataset, [ps, ghsl, truncation, rsphere, jobs])

        Parameters
        ----------
        :param dataset: xarray.Dataset or str indicating the path to a dataset.

            The dataset must contain the following analysis fields:
            u: Horizontal wind component in the zonal direction
            v: Horizontal wind component in the meridional direction
            w: Height/pressure vertical velocity depending on leveltype (inferred from dataset)
            t: air temperature
            p: Atmospheric pressure. A 1D array for isobaric levels or a 3-4D array for arbitrary
               vertical coordinate. Data is interpolated to pressure levels before the analysis.

        :param variables: dict, optional,
            A dictionary mapping of the field names in the dataset to the internal variable names.
            The default names are: ['u_wind', 'v_wind', 'omega', 'temperature', 'pressure'].
            Ensures all variables needed for the analysis are found. If not given, variables are
            looked up based on standard CF conventions of variable names, units and typical value
            ranges. Example: variables = {'u_wind': 'U', 'temperature': 'temp'}. Note that often
            used names 'U' and 'temp' are not conventional names.

        :param truncation: int, optional, default None
            Triangular truncation for the spherical harmonic transforms. If truncation is not
            specified then 'truncation=nlat-1' is used, where 'nlat' is the number of
            latitude points.

        :param rsphere: float, optional,
            Averaged earth radius (meters), default 'rsphere = 6371200'.

        :param p_levels: iterable, optional
            Contains the pressure levels in (Pa) for vertical interpolation.
            Ignored if the data is already in pressure coordinates.

        :param jobs: integer, optional, default None
            Number of processors to operate along non-spatial dimensions in parallel.
            Recommended jobs=1 since spectral transforms are already efficiently parallelized.
        """

        # check number of workers
        if jobs is None:
            self.jobs = get_num_cores()
        else:
            self.jobs = int(jobs)

        # check if input dataset is a path to file
        if isinstance(dataset, str):
            dataset = xr.open_mfdataset(dataset, combine='by_coords', parallel=False)

        if not isinstance(dataset, xr.Dataset):
            raise TypeError("Input parameter 'dataset' must be xarray.Dataset instance"
                            "or a string containing the path to a netcdf file.")

        # Initialize analysis variables
        dataset = parse_dataset(dataset, variables=variables)

        # Perform interpolation to constant pressure levels if needed
        dataset = interpolate_pressure_levels(dataset, p_levels=p_levels)

        # Create dictionary with axis/coordinate pairs (ensure dimension order is preserved)
        # These coordinates are used to export data as xarray objects
        self.coords = {dataset.coords[d].axis.lower(): dataset.coords[d]
                       for d in get_coordinate_names(dataset)}
        self.info_coords = ''.join(self.coords)  # string used for data reshaping and handling.

        # find dataset coordinates
        self.latitude, self.nlat = _find_coordinate(dataset, "latitude")
        self.longitude, self.nlon = _find_coordinate(dataset, "longitude")

        # Get dynamic fields as masked arrays... loading the entire data on memory at this step!
        omega = dataset['omega'].to_masked_array()
        u_wind = dataset['u_wind'].to_masked_array()
        v_wind = dataset['v_wind'].to_masked_array()
        pressure = dataset['pressure'].to_masked_array()
        temperature = dataset['temperature'].to_masked_array()

        # Get surface fields if given in dataset
        self.ps = dataset.get('ps')
        self.ts = dataset.get('ts')

        dataset.close()

        # Get the dimensions for levels, latitude and longitudes in the input arrays.
        self.grid_shape = (self.nlat, self.nlon)

        # Get the size of the non-spatial dimensions (time, others ...). The analysis if performed
        # over 3D slices of data (lat, lon, pressure) by simply iterating over the sample axis.
        self.nlevels = pressure.size
        self.samples = np.prod(u_wind.shape) // (self.nlat * self.nlon * self.nlevels)

        # resulting shape after collapsing all non-spatial dimension into samples.
        self.data_shape = self.grid_shape + (self.samples, self.nlevels)

        # Inferring the direction of the vertical axis and reordering
        # from the surface to the model top if needed.
        if not (np.all(pressure[1:] >= pressure[:-1]) or np.all(pressure[1:] <= pressure[:-1])):
            raise ValueError("The vertical coordinate is not monotonic.")

        self.reverse_levels = pressure[0] < pressure[-1]
        self.pressure = np.asarray(sorted(pressure, reverse=True))

        if self.reverse_levels:
            self.coords['z'] = reindex_coordinate(self.coords['z'], self.pressure)
            self.coords['z'].attrs['positive'] = 'up'

        if self.ps is None:
            if ps is None:
                # If surface pressure not found set first pressure level as surface pressure
                self.ps = np.broadcast_to(np.max(pressure), self.grid_shape)
            else:
                if isinstance(ps, xr.DataArray):
                    self.ps = np.squeeze(ps.values)
                else:
                    self.ps = ps.squeeze()

            if np.shape(self.ps) != self.grid_shape:
                raise ValueError('If given, the surface pressure must be a scalar or a '
                                 '2D array with shape (nlat, nlon). Expected shape {}, '
                                 'but got {}!'.format(self.grid_shape, np.shape(ps)))
        else:
            self.ps = np.nanmean(self.ps, axis=self.info_coords.find('t'))

        # -----------------------------------------------------------------------------
        # Create sphere object to perform the spectral transformations.
        # -----------------------------------------------------------------------------
        # Inspect grid type based on the latitude sampling
        self.gridtype, self.latitude, self.weights = inspect_gridtype(self.latitude)

        if rsphere is None:
            self.rsphere = cn.earth_radius
        elif type(rsphere) == int or type(rsphere) == float:
            self.rsphere = abs(rsphere)
        else:
            raise ValueError("Incorrect value for 'rsphere'.")

        # define the triangular truncation
        if truncation is None:
            self.truncation = self.nlat - 1
        else:
            self.truncation = int(truncation)

            if self.truncation < 0 or self.truncation > self.nlat - 1:
                raise ValueError('Truncation must be between 0 and {:d}'.format(self.nlat - 1, ))

        # Create sphere object for spectral transforms
        self.sphere = Spharmt(self.nlon, self.nlat,
                              gridtype=self.gridtype, rsphere=self.rsphere,
                              ntrunc=self.truncation, jobs=self.jobs)

        # reverse latitude array (weights are symmetric around the equator)
        self.reverse_latitude = self.latitude[0] < self.latitude[-1]

        if self.reverse_latitude:
            self.latitude = self.latitude[::-1]

            # reorder latitude coordinates from north to south.
            # self.coords['y'] = self.coords['y'].reindex({self.coords['y'].name: self.latitude})
            self.coords['y'] = reindex_coordinate(self.coords['y'], self.latitude)

        # number of spectral coefficients: (truncation + 1) * (truncation + 2) / 2
        self.nlm = self.sphere.nlm

        # get spherical harmonic degree and horizontal wavenumber (rad / meter)
        self.degrees = np.arange(self.truncation + 1, dtype=int)
        self.kappa_h = kappa_from_deg(self.degrees)

        # create horizontal wavenumber coordinates for spectral quantities.
        self.sp_coords = [c for ic, c in self.coords.items() if ic not in 'xy']

        self.sp_coords.append(xr.Coordinate('kappa', self.kappa_h,
                                            attrs={'standard_name': 'wavenumber',
                                                   'long_name': 'horizontal wavenumber',
                                                   'axis': 'X', 'units': 'm**-1'}))

        # normalization factor for vector analysis:
        # vector_norm = n * (n + 1) / re ** 2 =  kh ** -2
        spectrum_shape = (self.truncation + 1, self.samples, self.nlevels)

        self.vector_norm = broadcast_1dto(self.kappa_h ** 2, spectrum_shape)
        self.vector_norm[0] = 1.0

        # --------------------------------------------------------------------------------
        # Preprocessing data:
        #  - Exclude extrapolated data below the surface (p >= ps).
        #  - Ensure latitude axis is oriented north-to-south.
        #  - Reverse vertical axis from the surface to the model top.
        # --------------------------------------------------------------------------------
        self.beta = terrain_mask(self.pressure, self.ps, smooth=False)  # mask is calculated once

        # Reshape and reorder data dimensions for computations
        # self.data_info stores information to recover the original shape.
        self.omega, self.data_info = self._transform_data(omega)
        del omega

        # create wind array from unfiltered wind components
        self.wind = np.stack((self._transform_data(u_wind)[0],
                              self._transform_data(v_wind)[0]))
        del u_wind, v_wind

        # compute thermodynamic quantities with unfiltered temperature
        self.temperature = self._transform_data(temperature)[0]
        del temperature

        # Compute vorticity and divergence of the wind field
        self.vrt_spc, self.div_spc = self.vorticity_divergence()

        # Transform of divergence and vorticity to the gaussian/regular grid
        self.div = self._inverse_transform(self.div_spc)
        self.vrt = self._inverse_transform(self.vrt_spc)

        # compute the vertical wind shear before filtering to avoid sharp gradients.
        self.wind_shear = self.vertical_gradient(self.wind)

        # Perform Helmholtz decomposition
        self.wind_div, self.wind_rot = self.helmholtz()

        # Coriolis parameter
        self.fc = coriolis_parameter(self.latitude)

        # -----------------------------------------------------------------------------
        # Thermodynamic diagnostics:
        # -----------------------------------------------------------------------------
        # Compute potential temperature
        self.exner = exner_function(self.pressure)
        self.theta = potential_temperature(self.pressure, self.temperature)

        # Compute geopotential (Compute before applying mask!)
        self.phi = self.geopotential()
        self.height = geopotential_to_height(self.phi)

        # Compute global average of potential temperature on pressure surfaces
        # above the ground (representative mean) and the perturbations.
        self.theta_avg, self.theta_pbn = self._split_mean_perturbation(self.theta)

        # Compute vertical gradient of potential temperature perturbations
        # (done before filtering to avoid sharp gradients at interfaces)
        self.ddp_theta_pbn = self.vertical_gradient(self.theta_pbn)

        # Parameter ganma to convert from temperature variance to APE
        self.ganma = stability_parameter(self.pressure, self.theta_avg, vertical_axis=-1)

    # -------------------------------------------------------------------------------
    # Methods for computing thermodynamic quantities
    # -------------------------------------------------------------------------------
    def add_metadata(self, data, name, gridtype, **attributes):
        """
            Add metadata and export variables as xr.DataArray
        """
        if gridtype == 'spectral':
            coords = self.sp_coords
            data = recover_spectra(data, self.data_info)
        else:
            coords = self.coords.values()
            data = recover_data(data, self.data_info)

        # create xarray.DataArray
        array = xr.DataArray(data=data, name=name, coords=coords)

        # add attributes to variable
        for attr, value in attributes.items():
            array.attrs[attr] = value

        return array

    # -------------------------------------------------------------------------------
    # Methods for thermodynamic diagnostics
    # -------------------------------------------------------------------------------
    def geopotential(self):
        """
        Computes geopotential at pressure surfaces assuming a hydrostatic atmosphere.
        The geopotential is obtained by integrating the hydrostatic balance equation from the
        surface to the top of the atmosphere. Uses terrain height and surface pressure as lower
        boundary conditions. Uses compiled fortran libraries for higher performance.

        :return: `np.ndarray`
            geopotential (J/kg)
        """

        # Compact horizontal coordinates into one dimension ...
        data_shape = self.temperature.shape
        proc_shape = (-1,) + data_shape[2:]

        temperature = np.moveaxis(self.temperature.reshape(proc_shape), 1, 0)
        temperature = np.ma.filled(temperature, fill_value=0.0)

        sfcp = self.ps.ravel()

        # get surface elevation for the given grid.
        sfch = get_surface_elevation(self.latitude, self.longitude)
        sfch = sfch.values.ravel()

        # Get the surface temperature
        if self.ts is not None:
            sfct = np.moveaxis(self.ts.values.reshape((-1, self.ts.shape[-1])), 1, 0)
        else:
            # compute surface temperature by linear interpolation
            sfct = numeric_tools.surface_temperature(sfcp, temperature, self.pressure)

        # Compute geopotential from the temperature field
        # signature 'geopotential(surface_geopotential, temperature, pressure)'
        phi = numeric_tools.geopotential(self.pressure, temperature, sfch, sfcp, sfct=sfct)

        # back to the original shape (same as temperature)
        return np.moveaxis(phi, 0, 1).reshape(data_shape)

    def geostrophic_wind(self):
        """
        Computes the geostrophically balanced wind
        """
        h_gradient = self.horizontal_gradient(self.height)

        fc = broadcast_1dto(self.fc, h_gradient.shape)

        return (cn.g / fc) * rotate_vector(h_gradient)

    def ageostrophic_wind(self):
        """
        Computes the ageostrophic wind
        """
        return self.wind - self.geostrophic_wind()

    # -------------------------------------------------------------------------------
    # Methods for computing diagnostics: kinetic and available potential energies
    # -------------------------------------------------------------------------------
    def horizontal_kinetic_energy(self):
        """
        Horizontal kinetic energy after Augier and Lindborg (2013), Eq.13
        :return:
        """
        if hasattr(self, 'vrt_spc') and hasattr(self, 'div_spc'):
            vrt_spc = self.cross_spectrum(self.vrt_spc)
            div_spc = self.cross_spectrum(self.div_spc)

            kinetic_energy = (vrt_spc + div_spc) / (2.0 * self.vector_norm)
        else:
            kinetic_energy = self._vector_spectra(self.wind) / 2.0

        #  create dataset
        kinetic_energy = self.add_metadata(kinetic_energy, 'hke',
                                           gridtype='spectral',
                                           units='m**2 s**-2',
                                           standard_name='horizontal_kinetic_energy',
                                           long_name='horizontal kinetic energy per unit mass')
        return kinetic_energy

    def vertical_kinetic_energy(self):
        """
        Vertical kinetic energy calculated from pressure vertical velocity
        :return:
        """
        w = vertical_velocity(self.pressure, self.omega, self.temperature)

        kinetic_energy = self._scalar_spectra(w) / 2.0

        #  create dataset
        kinetic_energy = self.add_metadata(kinetic_energy, 'vke',
                                           gridtype='spectral',
                                           units='m**2 s**-2',
                                           standard_name='vertical_kinetic_energy',
                                           long_name='vertical kinetic energy per unit mass')
        return kinetic_energy

    def available_potential_energy(self):
        """
        Total available potential energy after Augier and Lindborg (2013), Eq.10
        :return:
        """
        potential_energy = self.ganma * self._scalar_spectra(self.theta_pbn) / 2.0

        potential_energy = self.add_metadata(potential_energy, 'ape',
                                             gridtype='spectral',
                                             units='m**2 s**-2',
                                             standard_name='available_potential_energy',
                                             long_name='available potential energy per unit mass')
        return potential_energy

    def vorticity_divergence(self):
        # computes the spectral coefficients of vertical
        # vorticity and horizontal wind divergence
        return self._compute_rotdiv(self.wind)

    # -------------------------------------------------------------------------------
    # Methods for computing spectral fluxes
    # -------------------------------------------------------------------------------
    def ke_nonlinear_transfer(self):
        """
        Kinetic energy spectral transfer due to nonlinear interactions after
        Augier and Lindborg (2013), Eq.A2
        :return:
            Spectrum of KE transfer across scales
        """

        # compute advection of the horizontal wind (using the rotational form)
        advection_term = self._wind_advection(self.wind) + self.div * self.wind / 2.0

        # compute nonlinear spectral transfer related to horizontal advection
        advective_flux = - self._vector_spectra(self.wind, advection_term)

        # This term seems to effectively cancel out after summing over all zonal wavenumber.
        vertical_transport = self._vector_spectra(self.wind_shear, self.omega * self.wind)
        vertical_transport -= self._vector_spectra(self.wind, self.omega * self.wind_shear)

        return advective_flux + vertical_transport / 2.0

    def rke_nonlinear_transfer(self):
        """
        Spectral transfer of rotational kinetic energy due to nonlinear interactions
        after Li et. al. (2023), Eq. 28
        :return:
            Spectrum of RKE transfer across scales
        """

        # This term seems to effectively cancel out after summing over all zonal wavenumber.
        vertical_transport = self._vector_spectra(self.wind_shear, self.omega * self.wind_rot)
        vertical_transport -= self._vector_spectra(self.wind_rot, self.omega * self.wind_shear)

        # cross product of vertical unit vector and horizontal winds (counterclockwise rotation)
        cross_wind = rotate_vector(self.wind)
        cross_wrot = rotate_vector(self.wind_rot)

        # Rotational effect due to the Coriolis force on the spectral
        fc = broadcast_1dto(self.fc, cross_wrot.shape)

        deformation = - self._vector_spectra(self.wind_rot, fc * cross_wind)
        deformation -= self._vector_spectra(self.wind, fc * cross_wrot)
        deformation -= self._vector_spectra(self.wind_rot, self.vrt * cross_wind)
        deformation -= self._vector_spectra(self.wind, self.vrt * cross_wrot)

        return (vertical_transport + deformation) / 2.0

    def dke_nonlinear_transfer(self):
        """
        Spectral transfer of divergent kinetic energy due to nonlinear interactions
        after Li et. al. (2023), Eq. 27. The linear Coriolis effect is included in the
        formulations so that:

        .. math:: T_{D}(l,m) + T_{R}(l,m) = T_{K}(l,m) + L(l,m)

        :return:
            Spectrum of DKE transfer across scales
        """

        # Horizontal kinetic energy per unit mass in grid-point space
        kinetic_energy = np.sum(self.wind * self.wind, axis=0)

        # Horizontal gradient of horizontal kinetic energy
        kinetic_energy_gradient = self.horizontal_gradient(kinetic_energy)

        # compute nonlinear spectral transfer related to horizontal advection
        advective_flux = - self._vector_spectra(self.wind_div, kinetic_energy_gradient)
        advective_flux -= self._vector_spectra(self.wind, self.div * self.wind)

        # This term seems to effectively cancel out after summing over all zonal wavenumber.
        vertical_transport = self._vector_spectra(self.wind_shear, self.omega * self.wind_div)
        vertical_transport -= self._vector_spectra(self.wind_div, self.omega * self.wind_shear)

        # cross product of vertical unit vector and horizontal winds
        cross_wind = rotate_vector(self.wind)
        cross_wdiv = rotate_vector(self.wind_div)

        # Rotational effect due to the Coriolis force on the spectral
        fc = broadcast_1dto(self.fc, cross_wdiv.shape)

        deformation = - self._vector_spectra(self.wind_div, fc * cross_wind)
        deformation -= self._vector_spectra(self.wind, fc * cross_wdiv)
        deformation -= self._vector_spectra(self.wind_div, self.vrt * cross_wind)
        deformation -= self._vector_spectra(self.wind, self.vrt * cross_wdiv)

        return (advective_flux + vertical_transport + deformation) / 2.0

    def ape_nonlinear_transfer(self):
        """
        Available potential energy spectral transfer due to nonlinear interactions
        after Augier and Lindborg (2013), Eq.A3
        :return:
            Spherical harmonic coefficients of APE transfer across scales
        """

        # compute horizontal advection of potential temperature
        theta_advection = self._scalar_advection(self.theta_pbn) + self.div * self.theta_pbn / 2.0

        # compute nonlinear spectral transfer related to horizontal advection
        advection_term = - self._scalar_spectra(self.theta_pbn, theta_advection)

        # compute vertical transfer
        vertical_trans = self._scalar_spectra(self.ddp_theta_pbn, self.omega * self.theta_pbn)
        vertical_trans -= self._scalar_spectra(self.theta_pbn, self.omega * self.ddp_theta_pbn)

        return self.ganma * (advection_term + vertical_trans / 2.0)

    def pressure_flux(self):
        # Pressure flux (Eq.22)
        return - self._scalar_spectra(self.omega, self.height)

    def dke_turbulent_flux(self):
        # Turbulent kinetic energy flux (Eq.22)
        return - self._vector_spectra(self.wind, self.omega * self.wind) / 2.0

    def dke_vertical_flux(self):
        # Vertical flux of total kinetic energy (Eq. A9)
        return self.pressure_flux() + self.dke_turbulent_flux()

    def ape_vertical_flux(self):
        # Total APE vertical flux (Eq. A10)
        ape_flux = self._scalar_spectra(self.theta_pbn, self.omega * self.theta_pbn)

        return - self.ganma * ape_flux / 2.0

    def surface_fluxes(self):
        return

    def conversion_ape_dke(self):
        # Conversion of Available Potential energy into kinetic energy
        # Equivalent to Eq. 19 of A&L, but using potential temperature.
        ape_dke = - self._scalar_spectra(self.omega, self.theta)

        return cn.Rd * self.exner * ape_dke / self.pressure

    def conversion_dke_rke(self):
        """Conversion from divergent to rotational kinetic energy
        """

        # Nonlinear interaction term due to relative vorticity
        vorticity_advection = self.conversion_dke_rke_vorticity()

        # Rotational effect due to the Coriolis force on the spectral
        linear_conversion = self.conversion_dke_rke_coriolis()

        # Vertical transfer
        vertical_transfer = self.conversion_dke_rke_vertical()

        return vorticity_advection + linear_conversion + vertical_transfer

    def conversion_dke_rke_vertical(self):
        """Conversion from divergent to rotational energy due to vertical transfer
        """
        # vertical transfer
        vertical_transfer = self._vector_spectra(self.wind_shear, self.omega * self.wind_rot)
        vertical_transfer += self._vector_spectra(self.wind_rot, self.omega * self.wind_shear)

        return - vertical_transfer / 2.0

    def conversion_dke_rke_coriolis(self):
        """Conversion from divergent to rotational energy
        """

        # cross product of vertical unit vector and horizontal winds
        cw_rot = rotate_vector(self.wind_rot)
        cw_div = rotate_vector(self.wind_div)

        # Rotational effect due to the Coriolis force on the spectral
        # transfer of divergent kinetic energy
        fc = broadcast_1dto(self.fc, cw_rot.shape)

        ct_interaction = self._vector_spectra(self.wind_div, fc * cw_rot)
        ct_interaction -= self._vector_spectra(self.wind_rot, fc * cw_div)

        return ct_interaction / 2.0

    def conversion_dke_rke_vorticity(self):
        """Conversion from divergent to rotational energy
        """

        # cross product of vertical unit vector and horizontal winds
        cw_rot = rotate_vector(self.wind_rot)
        cw_div = rotate_vector(self.wind_div)

        # nonlinear interaction term
        nlt_interaction = self._vector_spectra(self.wind_div, self.vrt * cw_rot)
        nlt_interaction -= self._vector_spectra(self.wind_rot, self.vrt * cw_div)

        return nlt_interaction / 2.0

    def diabatic_conversion(self):
        # need to estimate Latent heat release*
        return

    def coriolis_linear_transfer(self):
        # Linear Coriolis transfer
        cross_wind = rotate_vector(self.wind)

        fc = broadcast_1dto(self.fc, cross_wind.shape)

        linear_term = - self._vector_spectra(self.wind, fc * cross_wind)

        return linear_term

    def non_conservative_term(self):
        # non-conservative term J(p) in Eq. A11
        dlog_gamma = self.vertical_gradient(np.log(self.ganma))

        return - dlog_gamma.reshape(-1) * self.ape_vertical_flux()

    def cumulative_energy_fluxes(self):
        """
        Computes each term in spectral energy budget and return as xr.DataArray objects.
        """

        # ------------------------------------------------------------------------------------------
        # Energy conversions APE --> DKE and DKE --> RKE
        # ------------------------------------------------------------------------------------------
        c_ad = cumulative_flux(self.conversion_ape_dke())

        c_dr = cumulative_flux(self.conversion_dke_rke())

        # Compute cumulative nonlinear spectral energy fluxes
        pi_r = cumulative_flux(self.rke_nonlinear_transfer())
        pi_d = cumulative_flux(self.dke_nonlinear_transfer())
        pi_a = cumulative_flux(self.ape_nonlinear_transfer())

        # Linear transfer due to Coriolis
        lc_k = cumulative_flux(self.coriolis_linear_transfer())

        # add metadata
        attrs = dict(gridtype='spectral', units='W m**-2')

        c_ad = self.add_metadata(c_ad, 'cad', standard_name='conversion_ape_dke',
                                 long_name='conversion from available potential energy '
                                           'to divergent kinetic energy', **attrs)

        c_dr = self.add_metadata(c_dr, 'cdr', standard_name='conversion_dke_rke',
                                 long_name='conversion from divergent to rotational kinetic energy',
                                 **attrs)

        pi_r = self.add_metadata(pi_r, 'pi_rke', standard_name='nonlinear_rke_flux',
                                 long_name='cumulative spectral flux of rotational kinetic energy',
                                 **attrs)

        pi_d = self.add_metadata(pi_d, 'pi_dke', standard_name='nonlinear_dke_flux',
                                 long_name='cumulative spectral flux of divergent kinetic energy',
                                 **attrs)

        pi_a = self.add_metadata(pi_a, 'pi_ape', standard_name='nonlinear_ape_flux',
                                 long_name='cumulative spectral flux of available potential energy',
                                 **attrs)

        lc_k = self.add_metadata(lc_k, 'lc', standard_name='coriolis_transfer',
                                 long_name='coriolis linear transfer', **attrs)

        # ------------------------------------------------------------------------------------------
        # Cumulative vertical fluxes of divergent kinetic energy
        # ------------------------------------------------------------------------------------------
        vf_p = self.pressure_flux()
        vf_m = self.dke_turbulent_flux()

        vf_k = cumulative_flux(self.vertical_gradient(vf_p + vf_m))
        vf_a = cumulative_flux(self.vertical_gradient(self.ape_vertical_flux()))

        # add metadata
        vf_p = self.add_metadata(vf_p, 'pf_dke', gridtype='spectral',
                                 units='W m**-2', standard_name='pressure_dke_flux',
                                 long_name='vertical pressure flux')

        vf_m = self.add_metadata(vf_m, 'mf_dke', gridtype='spectral',
                                 units='W m**-2', standard_name='turbulent_dke_flux',
                                 long_name='vertical turbulent flux of kinetic energy')

        vf_k = self.add_metadata(vf_k, 'vf_dke', gridtype='spectral',
                                 units='W m**-2', standard_name='vertical_dke_flux',
                                 long_name='cumulative vertical flux of kinetic energy')

        vf_a = self.add_metadata(vf_a, 'vf_ape', gridtype='spectral',
                                 units='W m**-2', standard_name='vertical_ape_flux',
                                 long_name='cumulative vertical flux of '
                                           'available potential energy')

        return xr.merge([pi_d, pi_r, lc_k, pi_a, c_ad, c_dr,
                         vf_p, vf_m, vf_k, vf_a], compat="no_conflicts")

    def get_ke_tendency(self, tendency, name=None, cumulative=False):
        r"""
            Compute kinetic energy spectral transfer from parametrized
            or explicit horizontal wind tendencies.

            .. math:: \partial_{t}E_{K}(l) = (\mathbf{u}, \partial_{t}\mathbf{u})_{l}

            where :math:`\boldsymbol{u}=(u, v)` is the horizontal wind vector,
            and :math:`\partial_{t}\boldsymbol{u}` is defined by tendency.

            Parameters
            ----------
                tendency: ndarray with shape (2, nlat, nlon, ...)
                    contains momentum tendencies for each horizontal component
                    stacked along the first axis.
                name: str,
                    name of the tendency
                cumulative: bool,
                    convert to cumulative flux
            Returns
            -------
                Kinetic energy tendency due to any process given by 'tendency'.
        """
        da_flag = isinstance(tendency, xr.DataArray)

        tendency_name = name
        if da_flag:
            if tendency_name is None:
                tendency_name = tendency.name.split("_")[-1]
            info = ''.join([tendency.coords[dim].axis for dim in tendency.dims]).lower()

            tendency, _ = prepare_data(tendency.values, info)
        else:
            tendency = np.asarray(tendency)

            if tendency.shape != self.wind.shape:
                raise ValueError("The shape of 'tendency' array must be "
                                 "consistent with the initialized wind. Expecting {}, "
                                 "but got {}".format(self.wind.shape, tendency.shape))

        ke_tendency = self._vector_spectra(self.wind, tendency)

        if cumulative:
            ke_tendency = cumulative_flux(ke_tendency)

        if da_flag:
            ke_tendency = self.add_metadata(ke_tendency, tendency_name,
                                            gridtype='spectral', units='W m**-2',
                                            standard_name=tendency_name)
        return ke_tendency

    def get_ape_tendency(self, tendency, name=None, cumulative=True):
        r"""
            Compute Available potential energy tendency from
            parametrized or explicit temperature tendencies.

            .. math:: {\partial}_{t}E_{A}(l)= (\theta^{\prime}, \partial_{t}\theta^{\prime})_{l}

            Parameters
            ----------
                tendency: ndarray with shape (nlat, nlon, ...)
                    contains a diabatic temperature tendency.
                name: str,
                    name of the tendency
                cumulative: bool,
                    convert to cumulative flux
            Returns
            -------
                Available potential energy tendency due to diabatic processes.
        """

        da_flag = isinstance(tendency, xr.DataArray)

        tendency_name = name
        if da_flag:
            if tendency_name is None:
                tendency_name = tendency.name.split("_")[-1]
            info = ''.join([tendency.coords[dim].axis for dim in tendency.dims]).lower()

            tendency, _ = prepare_data(tendency.values, info)
        else:
            # check dimensions
            tendency = np.asarray(tendency)

            if tendency.shape != self.theta_pbn.shape:
                raise ValueError("The shape of 'tendency' array must be "
                                 "consistent with the initialized temperature. Expecting {}, "
                                 "but got {}".format(self.wind.shape, tendency.shape))

        # remove representative mean from total temperature tendency
        # tendency -= self._representative_mean(tendency)

        # convert temperature tendency to potential temperature tendency
        theta_tendency = tendency / self.exner / cn.cp  # rate of production of internal energy

        # filtering terrain
        theta_tendency = self.filter_topography(theta_tendency)

        ape_tendency = self.ganma * self._scalar_spectra(self.theta_pbn, theta_tendency)

        if cumulative:
            ape_tendency = cumulative_flux(ape_tendency)

        if da_flag:
            ape_tendency = self.add_metadata(ape_tendency, tendency_name,
                                             gridtype='spectral', units='W m**-2',
                                             standard_name=tendency_name)

        return ape_tendency

    def helmholtz(self):
        """
        Perform a Helmholtz decomposition of the horizontal wind.
        This decomposition splits the horizontal wind vector into
        irrotational and non-divergent components.

        Returns:
            uchi, vchi, upsi, vpsi:
            zonal and meridional components of divergent and
            rotational wind components respectively.
        """

        # streamfunction and velocity potential
        psi_grid, chi_grid = self.streamfunction_potential(self.wind)

        # Compute non-rotational components from streamfunction
        chi_grad = self.horizontal_gradient(chi_grid)

        # Compute non-divergent components from velocity potential
        psi_grad = self.horizontal_gradient(psi_grid)

        return chi_grad, rotate_vector(psi_grad)

    # --------------------------------------------------------------------
    # Helper methods for spectral transformations
    # --------------------------------------------------------------------
    @transform_io
    def _spectral_transform(self, scalar):
        """
        Compute spherical harmonic coefficients of a scalar function on the sphere.
        Wrapper around 'grdtospec' to process inputs and run in parallel.
        """
        return self.sphere.grdtospec(scalar)

    @transform_io
    def _inverse_transform(self, scalar_sp):
        """
            Compute spherical harmonic coefficients of a scalar function on the sphere.
            Wrapper around 'spectogrd' to process inputs and run in parallel.
        """
        return self.sphere.spectogrd(scalar_sp)

    @transform_io
    def _compute_rotdiv(self, vector):
        """
        Compute the spectral coefficients of vorticity and horizontal
        divergence of a vector field on the sphere.
        """
        return self.sphere.getvrtdivspec(*vector)

    @transform_io
    def streamfunction_potential(self, vector):
        """
            Computes the streamfunction and potential of a vector field on the sphere.
        """
        return self.sphere.getpsichi(*vector)

    @transform_io
    def horizontal_gradient(self, scalar):
        """
            Computes horizontal gradient of a scalar function on the sphere.
            Wrapper around 'getgrad' to process inputs and run in parallel.

        Returns:
            Arrays containing gridded zonal and meridional
            components of the vector gradient.
        """
        return self.sphere.getgrad(scalar)

    def _scalar_advection(self, scalar):
        """
        Compute the horizontal advection as dot product between
        the wind vector and scalar gradient.

        scalar: scalar field to be advected
        """
        # computes the components of the scalar advection: (2, nlat, nlon, ...)
        scalar_advection = self.wind * self.horizontal_gradient(scalar)

        return np.sum(scalar_advection, axis=0)

    def _wind_advection(self, wind):
        r"""
        Compute the horizontal advection of the horizontal wind in 'rotation form'

        .. math::

        \mathbf{u}\cdot\nabla_h\mathbf{u}=\nabla_h|\mathbf{u}|^{2}/2+\mathbf{\zeta}\times\mathbf{u}

        where :math:`\mathbf{u}=(u, v)` is the horizontal wind vector,
        and :math:`\mathbf{\zeta}` is the vertical vorticity.

        Notes
        -----
        Advection calculated in rotation form is more robust than the standard convective form
        :math:`(\mathbf{u}\cdot\nabla_h)\mathbf{u}` around sharp discontinuities (Zang, 1991).

        Thomas A. Zang, On the rotation and skew-symmetric forms for incompressible
        flow simulations. [https://doi.org/10.1016/0168-9274(91)90102-6]

        Parameters:
        -----------
            ugrid: `np.ndarray`
                zonal component of the horizontal wind
            vgrid: `np.ndarray`
                meridional component of the horizontal wind
        Returns:
        --------
            advection: `np.ndarray`
                Array containing the zonal and meridional components of advection
        """

        # Horizontal kinetic energy per unit mass in grid-point space
        kinetic_energy = np.sum(wind * wind, axis=0) / 2.0

        # Horizontal gradient of horizontal kinetic energy
        # (components stored along the first dimension)
        kinetic_energy_gradient = self.horizontal_gradient(kinetic_energy)

        # Horizontal advection of zonal and meridional wind components
        # (components stored along the first dimension)
        return kinetic_energy_gradient + self.vrt * rotate_vector(wind)

    def _scalar_spectra(self, scalar_1, scalar_2=None):
        """
        Compute 2D power spectra as a function of spherical harmonic degree
        of a scalar function on the sphere.
        """
        scalar_1sc = self._spectral_transform(scalar_1)

        if scalar_2 is None:
            spectrum = self.cross_spectrum(scalar_1sc)
        else:
            scalar_2sc = self._spectral_transform(scalar_2)
            spectrum = self.cross_spectrum(scalar_1sc, scalar_2sc)

        return spectrum

    def _vector_spectra(self, vector_1, vector_2=None):
        """
        Compute spherical harmonic cross spectra between two vector fields on the sphere.
        """
        rot_1, div_1 = self._compute_rotdiv(vector_1)

        if vector_2 is None:
            spectrum = self.cross_spectrum(rot_1) + self.cross_spectrum(div_1)
        else:
            rot_2, div_2 = self._compute_rotdiv(vector_2)

            spectrum = self.cross_spectrum(rot_1, rot_2) + self.cross_spectrum(div_1, div_2)

        return spectrum / self.vector_norm

    def vertical_gradient(self, scalar, vertical_axis=-1):
        """
            Computes vertical gradient of a scalar function in pressure coordinates: d(scalar)/dp
        """
        return gradient_1d(scalar, self.pressure, axis=vertical_axis)

    def vertical_integration(self, scalar, pressure_range=None, vertical_axis=-1):
        r"""Computes mass-weighted vertical integral of a scalar function.

            .. math:: \Phi = \int_{z_b}^{z_t}\rho(z)\phi(z)~dz
            where :math:`\phi` is any scalar and :math:`\rho` is density.
            In pressure coordinates, assuming a hydrostatic atmosphere, the above can be written as:

            .. math:: \Phi = \int_{p_t}^{p_b}\phi(p)/g~dp
            where :math:`p_{t,b}` is pressure at the top/bottom of the integration interval,
            and :math:`g` is gravity acceleration.

        Parameters
        ----------
        scalar : `np.ndarray`
            Scalar function

        pressure_range: list,
            pressure interval limits: :math:`(p_t, p_b)`
        vertical_axis: `int`
            axis of integration

        Returns
        -------
        `np.ndarray`
            The vertically integrated scalar
        """
        if pressure_range is None:
            pressure_range = [self.pressure[0], self.pressure[-1]]
        else:
            assert pressure_range[0] != pressure_range[1], "Inconsistent pressure levels" \
                                                           " for vertical integration."

        pressure_range = np.sort(pressure_range)

        # find pressure surfaces where integration takes place
        pressure = self.pressure

        level_mask = (pressure >= pressure_range[0]) & (pressure <= pressure_range[1])
        # Excluding boundary points in vertical integration.
        level_mask &= (pressure != pressure[0]) & (pressure != pressure[-1])

        # convert binary mask to array index
        level_mask = np.where(level_mask)[0]

        # Get data inside integration interval along the vertical axis
        scalar = np.take(scalar, level_mask, axis=vertical_axis)

        # Integrate scalar at pressure levels
        scalar_int = - simpson(scalar, x=pressure[level_mask], axis=vertical_axis, even='avg')

        return scalar_int / cn.g

    def global_mean(self, scalar, weights=None, lat_axis=None):
        """
        Computes the global weighted average of a scalar function on the sphere.
        The weights are initialized according to 'grid_type': for grid_type='gaussian' we use
        gaussian quadrature weights. If grid_type='regular' the weights are defined as the
        cosine of latitude. If the grid is regular and latitude points are not available
        it returns global mean with weights = 1 / nlat (not recommended).

        :param scalar: nd-array with data to be averaged
        :param lat_axis: axis of the meridional dimension.
        :param weights: 1D-array containing latitudinal weights
        :return: Global mean of a scalar function
        """
        if lat_axis is None:
            lat_axis = 0
        else:
            lat_axis = normalize_axis_index(lat_axis, scalar.ndim)

        # check array dimensions
        if scalar.shape[lat_axis] != self.nlat:
            raise ValueError("Scalar size along axis must be nlat."
                             "Expected {} and got {}".format(self.nlat, scalar.shape[lat_axis]))

        if scalar.shape[lat_axis + 1] != self.nlon:
            raise ValueError("Dimensions nlat and nlon must be in consecutive order.")

        if weights is None:
            if hasattr(self, 'weights'):
                weights = self.weights
        else:
            weights = np.asarray(weights)

            if weights.size != scalar.shape[lat_axis]:
                raise ValueError("If given, 'weights' must be a 1D array of length 'nlat'."
                                 "Expected length {} but got {}.".format(self.nlat, weights.size))

        # Compute area-weighted average on the sphere (using either gaussian or linear weights)
        # Added masked-arrays support to exclude data below the surface. "np.average" doesn't work!
        scalar_average = np.ma.average(scalar, weights=weights, axis=lat_axis)

        # mean along the longitude dimension (same as lat_axis after array reduction)
        return np.nanmean(scalar_average, axis=lat_axis)

    def integrate_order(self, cs_lm, degrees=None):
        """Accumulates the cross-spectrum as a function of spherical harmonic degree.

        Signature
        ---------
        array = integrate_order(cs_lm, [degrees])

        Parameters
        ----------
        cs_lm : ndarray, shape ((ntrunc+1)*(ntrunc+2)/2, ...)
            contains the cross-spectrum of a set of spherical harmonic coefficients.
        degrees: 1D array, optional, default = None
            Spherical harmonics degree. If not given, degrees are inferred from
            the class definition or calculated from the number of latitude points.
        Returns
        -------
        array : ndarray, shape (len(degrees), ...)
            contains the 1D spectrum as a function of spherical harmonic degree.
        """

        # Get indexes of the triangular matrix with spectral coefficients
        # (move this to class init?)
        sample_shape = cs_lm.shape[1:]

        coeffs_size = cs_lm.shape[0]

        if degrees is None:
            # check if degrees are defined
            if hasattr(self, 'degrees'):
                degrees = self.degrees
            else:
                if hasattr(self, 'truncation'):
                    ntrunc = self.truncation
                else:
                    ntrunc = triangular_truncation(coeffs_size)
                degrees = np.arange(ntrunc + 1, dtype=int)
        else:
            degrees = np.asarray(degrees)
            if (degrees.ndim != 1) or (degrees.size > self.nlat):
                raise ValueError("If given, 'degrees' must be a 1D array of length <= 'nlat'."
                                 "Expected size {} and got {}".format(self.nlat, degrees.size))

        # define wavenumbers locally
        ls = self.sphere.degree
        ms = self.sphere.order

        # Multiplying by 2 to account for symmetric coefficients (ms != 0)
        cs_lm = (np.where(ms == 0, 1.0, 2.0) * cs_lm.T).T

        # Initialize array for the 1D energy/power spectrum shaped (truncation, ...)
        spectrum = np.zeros((degrees.size,) + sample_shape)

        # Compute spectrum as a function of total wavenumber by adding up the zonal wavenumbers.
        for ln, degree in enumerate(degrees):
            # Sum over all zonal wavenumbers <= total wavenumber
            degree_range = (ms <= degree) & (ls == degree)
            spectrum[ln] = np.nansum(cs_lm[degree_range], axis=0)

        # Using the normalization in equation (7) of Lambert [1984].
        # spectrum /= 2.0
        return spectrum

    def cross_spectrum(self, clm1, clm2=None, degrees=None, convention='power', integrate=True):
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
        integrate : bool, default = True
            Option to integrate along the zonal wavenumber (order)
        Returns
        -------
        array : ndarray, shape ((ntrunc+1)*(ntrunc+2)/2, ...)
            contains the cross spectrum as a function of spherical harmonic degree (and order).
        """

        if convention not in ['energy', 'power']:
            raise ValueError("Parameter 'convention' must be one of"
                             " ['energy', 'power']. Given {}".format(convention))

        if clm2 is None:
            clm_sqd = (clm1 * clm1.conjugate()).real
        else:
            assert clm2.shape == clm1.shape, \
                "Arrays 'clm1' and 'clm2' of spectral coefficients must have the same shape. " \
                "Expected 'clm2' shape: {} got: {}".format(clm1.shape, clm2.shape)

            clm_sqd = (clm1 * clm2.conjugate()).real

        if convention.lower() == 'energy':
            clm_sqd *= 4.0 * np.pi

        if integrate:
            return self.integrate_order(clm_sqd, degrees)
        else:
            return clm_sqd

    # Functions for preprocessing data:
    def _pack_levels(self, data, order='C'):
        # pack dimensions of arrays (nlat, nlon, ...) to (nlat, nlon, samples)
        data_length = np.shape(data)[0]

        if data_length == 2:
            new_shape = np.shape(data)[:3]
        elif data_length == self.nlat:
            new_shape = np.shape(data)[:2]
        elif data_length == self.nlm:
            new_shape = np.shape(data)[:1]
        else:
            raise ValueError("Inconsistent array shape: expecting "
                             "first dimension of size {} or {}.".format(self.nlat, self.nlm))
        return np.reshape(data, new_shape + (-1,), order=order).squeeze()

    def _unpack_levels(self, data, order='C'):
        # unpack dimensions of arrays (nlat, nlon, samples)
        if np.shape(data)[-1] == self.samples * self.nlevels:
            new_shape = np.shape(data)[:-1] + (self.samples, self.nlevels)
            return np.reshape(data, new_shape, order=order)
        else:
            return data

    def filter_topography(self, scalar):
        # masks scalar values pierced by the topography
        return self.beta[..., np.newaxis, :] * scalar

    def _transform_data(self, scalar):
        # Helper function for preparing data for the analysis

        # Move dimensions (nlat, nlon) forward and vertical axis last
        # (Useful for cleaner vectorized operations)
        data, data_info = prepare_data(scalar, self.info_coords)

        # Ensure the latitude dimension is ordered north-to-south
        if self.reverse_latitude:
            # Reverse latitude dimension
            data = np.flip(data, axis=0)

        # Ensure that the vertical dimension is ordered from the surface to the model top
        if self.reverse_levels:
            # Reverse data along vertical axis
            data = np.flip(data, axis=-1)

        # Mask data pierced by the topography, if not already masked during the vertical
        # interpolation. This is required to avoid calculating vertical gradients between
        # masked and unmasked neighbours grid points in the vertical.
        if not np.ma.is_masked(data):
            mask = ~self.beta.astype(bool)

            mask = np.broadcast_to(mask[..., np.newaxis, :], self.data_shape)
            data = np.ma.masked_array(data, mask=mask)
        else:
            # Filter out interpolated subterranean data using smoothed Heaviside function
            # convert data to masked array according to not smoothed mask. It only affects
            # the data if the mask 'beta' has a smooth transition at the edges (0 - 1).
            data = self.filter_topography(data)

        # masked elements are filled with zeros before the spectral analysis
        data.set_fill_value(0.0)

        return data, data_info

    def _representative_mean(self, scalar):
        # Computes representative mean of a scalar function:
        # Mean over a constant pressure level for regions above the surface.

        if np.ma.is_masked(scalar):
            # If the scalar is masked the average is performed only over unmasked data
            reduced_points = 1.0
        else:
            # The globally averaged beta as used as a scaling factor to account for zeros in data.
            # Gives the percentage of valid points above the surface at every level.
            reduced_points = self.global_mean(self.beta, lat_axis=0).clip(cn.epsilon, 1.0)

        # Compute weighted average on gaussian grid and scale by the number of reduced points
        return self.global_mean(scalar, lat_axis=0) / reduced_points

    def _split_mean_perturbation(self, scalar):
        # Decomposes a scalar function into the representative mean and perturbations.

        # A&L13 formula for the representative mean
        scalar_avg = self._representative_mean(scalar)

        # Calculate perturbation
        scalar_pbn = scalar - scalar_avg

        return scalar_avg, scalar_pbn

    def _scalar_perturbation(self, scalar):
        # Compute scalar perturbations in spectral space
        scalar_spc = self._spectral_transform(scalar)

        # set mean coefficient (ls=ms=0) to 0.0 and invert transformation
        mean_index = (self.sphere.order == 0) & (self.sphere.degree == 0)
        scalar_spc[mean_index] = 0.0

        return self._inverse_transform(scalar_spc)
