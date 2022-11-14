import functools
import multiprocessing as mp

import numpy as np
import spharm
from numpy.core.numeric import normalize_axis_index
from scipy.integrate import simpson

import constants as cn
from spectral_analysis import triangular_truncation, kappa_from_deg
from thermodynamics import density as _density
from thermodynamics import exner_function as _exner_function
from thermodynamics import geopotential_height as _geopotential_height
from thermodynamics import height_to_geopotential
from thermodynamics import potential_temperature as _potential_temperature
from thermodynamics import pressure_vertical_velocity, vertical_velocity
from tools import terrain_mask, number_chunks, transform_io, prepare_data


class EnergyBudget(object):
    """
        Spectral Energy Budget of a dry hydrostatic Atmosphere.
        Implements the formulation introduced by Augier and Lindborg (2013)

        Augier, P., and E. Lindborg (2013), A new formulation of the spectral energy budget
        of the atmosphere, with application to two high-resolution general circulation models,
        J. Atmos. Sci., 70, 2293–2308.
    """

    def __init__(self, u, v, w, t, p, ps=None, ghsl=None, leveltype='pressure',
                 gridtype='gaussian', truncation=None, rsphere=None, standard_average=False,
                 legfunc='stored', axes=None, filter_terrain=False, jobs=None):

        """
        Initializing EnergyBudget instance.

        Signature
        -----
        energy_budget =  EnergyBudget(u, v, w, t, p, [ps, ghsl, leveltype='pressure',
                 gridtype, truncation, rsphere, legfunc, axes, sample_axis, filter_terrain, jobs])

        Parameters
        ----------
        :param u: horizontal wind component in the zonal direction
        :param v: horizontal wind component in the meridional direction
        :param w: height/pressure vertical velocity depending on leveltype
        :param t: air temperature
        :param p: atmospheric pressure
        :param gridtype: type of horizontal grid ('regular', 'gaussian')
        :param truncation:
            Truncation limit (triangular truncation) for the spherical
            harmonic computation.
        :param rsphere: averaged earth radius (meters)
        :param legfunc: Indicates whether the associated Legendre polynomials are stored or recomputed every time
        :param axes: tuple containing axis of the spatial dimensions (z, lat, lon)
        """

        # For both the input components check if there are missing values by
        # attempting to fill missing values with NaN and detect them. If the
        # inputs are not masked arrays then take copies and check for NaN.
        u = np.asanyarray(u).copy()
        v = np.asanyarray(v).copy()
        w = np.asanyarray(w).copy()
        t = np.asanyarray(t).copy()
        p = np.asanyarray(p).copy()

        if np.isnan(u).any() or np.isnan(v).any():
            raise ValueError('u and v cannot contain missing values')

        data_shape = u.shape

        # Make sure the shapes of the two components match.
        if data_shape != v.shape:
            raise ValueError('u and v must be the same shape')

        if t.shape != data_shape:
            raise ValueError('Temperature must be the same shape as u and v')

        if w.shape != data_shape:
            raise ValueError('w must be the same shape as u and v')

        data_dim = u.ndim

        if data_dim < 3:
            raise ValueError('variables must be at least 3D')

        if axes is None:
            axes = 'tzyx'  # Assuming input arrays with dimensions (time, levels, latitude, longitude)
        elif isinstance(axes, str):
            pass
        else:
            raise ValueError("Wrong type for 'axes', Expecting a string of length {}".format(data_dim))

        if len(axes) != data_dim:
            raise ValueError("Inconsistent number dimensions 'axes'"
                             " must be a string of length {}".format(data_dim))

        # Get the dimensions for levels, latitude and longitudes in the input arrays.
        self.axes = axes.lower()
        self.nlon, self.nlat, self.nlevels = [data_shape[self.axes.find(dim)] for dim in 'xyz']

        # size of the non-spatial dimensions
        self.nsamples = np.prod(data_shape) // (self.nlon * self.nlat * self.nlevels)

        # Get spatial dimensions
        self.leveltype = leveltype.lower()

        if self.leveltype == 'pressure':
            assert p.size == self.nlevels, "Pressure must be a 1D array with" \
                                            "size nlevels when using pressure coordinates"
            omega = w.copy()
            # compute vertical velocity in height coordinates
            w = vertical_velocity(p, omega, t, axis=1)

        elif self.leveltype == 'height':
            assert p.shape == u.shape, "Pressure must have same shape as u" \
                                       "when using height coordinates"
            # compute or load z coordinate
            omega = pressure_vertical_velocity(p, w, t)
            # perform vertical interpolation to pressure levels
        else:
            raise ValueError("Invalid level type: {}. Options are ('pressure', 'height')".format(leveltype))

        self.gridtype = gridtype.lower()

        if self.gridtype not in ('regular', 'gaussian'):
            raise ValueError("Invalid grid type: {0:s}. "
                             "Options are ('regular', 'gaussian')".format(repr(gridtype)))

        # making sure array splitting gives more chunks than jobs for parallel computations
        # if not given, the chunk size is set to the cpu count.
        packed_size = self.nsamples * self.nlevels
        if jobs is None:
            self.jobs = min(mp.cpu_count(), packed_size)
        else:
            self.jobs = min(int(jobs), packed_size)

        self.jobs = number_chunks(packed_size, self.jobs)
        self.chunk_size = 1

        print("Running with {} workers ...".format(self.jobs))

        # Infer direction of the vertical axis and flip accordingly
        self.direction = np.sign(p[0] - p[-1]).astype(int)
        self.p = np.asarray(sorted(p, reverse=True))

        if ps is None:
            # Set first pressure level as surface pressure
            self.ps = p.max()
        elif np.isscalar(ps):
            self.ps = ps
        else:
            if np.ndim(ps) > 2:
                self.ps = np.nanmean(ps, axis=0)
            else:
                self.ps = ps

            if np.shape(self.ps) != (self.nlat, self.nlon):
                raise ValueError('If given, the surface pressure must be a scalar or a'
                                 ' 2D array with shape (nlat, nlon).'
                                 'Expected shape {}, but got {}!'.format((self.nlat, self.nlon),
                                                                         np.shape(self.ps)))

        if ghsl is None:
            self.ghsl = 0.0
        else:
            if np.shape(ghsl) != (self.nlat, self.nlon):
                raise ValueError('If given, the surface height must be a 2D array with shape (nlat, nlon)'
                                 'Expected shape {}, but got {}!'.format((self.nlat, self.nlon),
                                                                         np.shape(ghsl)))
            else:
                self.ghsl = ghsl

        # -----------------------------------------------------------------------------
        # Create a Spharmt object to perform the computations.
        # -----------------------------------------------------------------------------
        if rsphere is None:
            self.rsphere = cn.earth_radius
        elif type(rsphere) == int or type(rsphere) == float:
            self.rsphere = abs(rsphere)
        else:
            raise ValueError("Incorrect value for 'rsphere'. If given, it must be a positive number.")

        # Create sphere object
        self.sphere = spharm.Spharmt(self.nlon, self.nlat, gridtype=self.gridtype,
                                     rsphere=self.rsphere, legfunc=legfunc)

        # define truncation
        if truncation is None:
            self.truncation = self.nlat - 1
        else:
            self.truncation = int(truncation)

            if self.truncation < 0 or self.truncation > self.nlat - 1:
                raise ValueError('Truncation must be between 0 and {:d}'.format(self.nlat - 1, ))

        # Get latitude of the gaussian grid and quadrature weights: weights ~ cosine(lat)
        self.lats, self.weights = spharm.gaussian_lats_wts(self.nlat)

        if standard_average:
            self.weights = None  # using uniform weights (1.0 / nlat)

        # reverse latitude array (weights are symmetric around the equator)
        self.reverse_latitude = self.lats[0] < self.lats[-1]
        if self.reverse_latitude:
            self.lats = self.lats[::-1]

        # Get spectral indexes of the zonal wavenumber and spherical harmonic degree
        self.zonal_wavenumber, self.total_wavenumber = spharm.getspecindx(self.truncation)

        # number of spectral coefficients: (truncation + 1) * (truncation + 2) / 2
        self.ncoeffs = self.total_wavenumber.size

        # get spherical harmonic degree and horizontal wavenumber (rad / meter)
        self.degrees = np.arange(self.truncation + 1, dtype=int)
        self.kappa_h = kappa_from_deg(self.degrees)

        # normalization factor for calculating vector cross-spectrum
        self.vector_norm = np.expand_dims(self.kappa_h ** 2, (-1, -2)).clip(min=cn.epsilon)

        # -----------------------------------------------------------------------------
        # Preprocessing data:
        #  - Exclude interpolated subterranean data from spectral calculations
        #  - Reshape input data to (nlat, nlon, samples * nlevels)
        # -----------------------------------------------------------------------------
        self.filter_terrain = filter_terrain

        if self.filter_terrain:
            self.beta = terrain_mask(self.p, self.ps, smoothed=True, jobs=self.jobs)
        else:
            self.beta = np.ones((self.nlat, self.nlon, self.nlevels))

        # compute divergence with unfiltered winds
        self.wind = np.stack((self._transform_data(u, filtered=False),
                              self._transform_data(v, filtered=False)))

        self.t = self._transform_data(t)
        self.w = self._transform_data(w)
        self.omega = self._transform_data(omega)

        # Compute vorticity and divergence of the wind field
        self.vrt_spc, self.div_spc = self.vorticity_divergence()

        # Transform of divergence/vorticity to grid-point space
        self.div = self._inverse_transform(self.div_spc)
        self.vrt = self._inverse_transform(self.vrt_spc)

        # compute the vertical wind shear before filtering
        self.wind_shear = self._vertical_gradient(self.wind)

        # filtering horizontal wind after computing divergence/vorticity
        self.wind = self.filter_topography(self.wind)

        # -----------------------------------------------------------------------------
        # Thermodynamic diagnostics:
        # -----------------------------------------------------------------------------
        # Compute potential temperature
        self.exner = _exner_function(self.p)
        self.theta = self.potential_temperature()

        # Compute specific volume (volume per unit mass)
        self.alpha = self.specific_volume()

        # Compute geopotential (Compute before applying mask!)
        self.phi = self.geopotential()

        # Compute global average of potential temperature on pressure surfaces
        # above the ground (representative mean) and the perturbations.
        self.theta_avg, self.theta_p = self._split_mean_perturbation(self.theta)

        # Compute vertical gradient of averaged potential temperature profile
        self.ddp_theta_p = self._vertical_gradient(self.theta_p)

        # Factor ganma(p) to convert from temperature variance to APE
        # using d(theta)/d(ln p) gives smoother gradients at the top/bottom boundaries.
        ddlp_theta_avg = self._vertical_gradient(self.theta_avg, z=np.log(self.p))
        self.ganma = - cn.Rd * self.exner / ddlp_theta_avg

    # -------------------------------------------------------------------------------
    # Methods for computing thermodynamic quantities
    # -------------------------------------------------------------------------------
    def potential_temperature(self):
        # computes potential temperature
        return _potential_temperature(self.p, self.t)

    def density(self):
        # computes air density from pressure and temperature using the gas law
        return _density(self.p, self.t)

    def specific_volume(self):
        return 1.0 / self.density()

    def geopotential_height(self):

        # Chunks of arrays along axis=-1 for the mp mapping ...
        data_shape = self.t.shape

        sfc_pressure = self.ps.reshape(-1)
        sfc_height = self.ghsl.reshape(-1)
        temperature = np.reshape(self.t, (-1,) + data_shape[2:])

        # create data chunks for parallel computations
        data_chunks = [chunk for chunk in zip(
            np.array_split(temperature, self.jobs, axis=0),
            np.array_split(sfc_pressure, self.jobs, axis=0),
            np.array_split(sfc_height, self.jobs, axis=0)
        )]

        # Create pool of workers
        pool = mp.Pool(processes=self.jobs)

        # perform computations in parallel
        _geopotential = functools.partial(_geopotential_height, pressure=self.p, axis=-1)

        height = pool.starmap(_geopotential, data_chunks, chunksize=self.chunk_size)

        # close pool of workers
        pool.close()
        pool.join()

        height = np.concatenate(height, axis=0).reshape(data_shape)

        # if not self.filter_terrain:
        return height

    def geopotential(self):
        """
        Computes geopotential at pressure surfaces using the hypsometric equation.
        Assumes a hydrostatic atmosphere. The geopotential at pressure levels
        below the earth surface are undefined therefore set to 0

        :return: `np.ndarray`
            geopotential (J/kg)
        """

        if hasattr(self, 'height'):
            height = self.height
        else:
            # Compute the geopotential height in meters
            # (levels below the surface are set to zero)
            height = self.geopotential_height()

        # Convert geopotential height to geopotential
        return height_to_geopotential(height)

    # -------------------------------------------------------------------------------
    # Methods for computing diagnostics: kinetic and available potential energies
    # -------------------------------------------------------------------------------
    def horizontal_kinetic_energy(self):
        """
        Horizontal kinetic energy after Augier and Lindborg (2013), Eq.13
        :return:
        """
        vrt_sqd = self.cross_spectrum(self.vrt_spc)
        div_sqd = self.cross_spectrum(self.div_spc)

        kinetic_energy = (vrt_sqd + div_sqd) / 2.0

        return kinetic_energy / self.vector_norm

    def vertical_kinetic_energy(self):
        """
        Vertical kinetic energy calculated from pressure vertical velocity
        :return:
        """
        return self._scalar_spectra(self.w) / 2.0

    def available_potential_energy(self):
        """
        Total available potential energy after Augier and Lindborg (2013), Eq.10
        :return:
        """
        return self.ganma * self._scalar_spectra(self.theta_p) / 2.0

    def vorticity_divergence(self):
        # computes the spectral coefficients of vertical
        # vorticity and horizontal wind divergence
        return self._compute_rotdiv(self.wind)

    # -------------------------------------------------------------------------------
    # Methods for computing spectral fluxes
    # -------------------------------------------------------------------------------
    def ke_nonlinear_transfer(self):
        """
        Kinetic energy spectral transfer due to nonlinear interactions
        after Augier and Lindborg (2013), Eq.A2
        :return:
            Spherical harmonic coefficients of KE transfer across scales
        """

        # compute advection of the horizontal wind
        advection_term = self._wind_advection(self.wind) + self.div * self.wind / 2.0

        # compute nonlinear spectral fluxes
        advective_flux = - self._vector_spectra(self.wind, advection_term)

        turbulent_flux = self._vector_spectra(self.wind_shear, self.omega * self.wind)
        turbulent_flux -= self._vector_spectra(self.wind, self.omega * self.wind_shear)

        return advective_flux + turbulent_flux / 2.0

    def ape_nonlinear_transfer(self):
        """
        Available potential energy spectral transfer due to nonlinear interactions
        after Augier and Lindborg (2013), Eq.A3
        :return:
            Spherical harmonic coefficients of APE transfer across scales
        """

        # compute horizontal advection of potential temperature
        theta_advection = self._scalar_advection(self.theta_p) + self.div * self.theta_p / 2.0

        # compute turbulent horizontal transfer
        advection_term = - self._scalar_spectra(self.theta_p, theta_advection)

        # compute vertical turbulent transfer
        vertical_trans = self._scalar_spectra(self.ddp_theta_p, self.omega * self.theta_p)
        vertical_trans -= self._scalar_spectra(self.theta_p, self.omega * self.ddp_theta_p)

        return self.ganma * (advection_term + vertical_trans / 2.0)

    def pressure_flux(self):
        # Pressure flux (Eq.22)
        return - self._scalar_spectra(self.omega, self.phi)

    def ke_turbulent_flux(self):
        # Turbulent kinetic energy flux (Eq.22)
        return - self._vector_spectra(self.wind, self.omega * self.wind) / 2.0

    def ke_vertical_flux(self):
        # Vertical flux of total kinetic energy (Eq. A9)
        return self.pressure_flux() + self.ke_turbulent_flux()

    def ape_vertical_flux(self):
        # Total APE vertical flux (Eq. A10)
        vertical_flux = self._scalar_spectra(self.theta_p, self.omega * self.theta_p)

        return - self.ganma * vertical_flux / 2.0

    def surface_fluxes(self):
        return

    def energy_conversion(self):
        # Conversion of APE to KE
        return - self._scalar_spectra(self.omega, self.alpha)

    def diabatic_conversion(self):
        # need to estimate Latent heat release*
        return

    def coriolis_linear_transfer(self):
        # relevance?
        sin_lat = np.sin(np.deg2rad(self.lats))
        cos_lat = np.cos(np.deg2rad(self.lats))

        sin_lat = np.expand_dims(sin_lat, (1, 2, 3))
        cos_lat = np.expand_dims(cos_lat, (1, 2, 3))

        # Compute the streamfunction and velocity potential
        sf, vp = self.sfvp(self.wind)

        # compute meridional gradients
        _, sf_grad = self.horizontal_gradient(sf)
        _, vp_grad = self.horizontal_gradient(vp)

        vp_grad *= cos_lat / cn.earth_radius ** 2
        sf_grad *= cos_lat / cn.earth_radius ** 2

        linear_term = self._scalar_spectra(sf, sin_lat * self.div + vp_grad)
        linear_term += self._scalar_spectra(vp, sin_lat * self.vrt - sf_grad)

        return cn.Omega * linear_term

    def non_conservative_term(self):
        # non-conservative term J(p) in Eq. A11
        dlog_gamma = self._vertical_gradient(np.log(self.ganma))

        return - dlog_gamma.reshape(-1) * self.ape_vertical_flux()

    def accumulated_fluxes(self, pressure_range=None):

        # Compute spectral energy fluxes accumulated over zonal wavenumbers
        tk_l = self.ke_nonlinear_transfer()
        ta_l = self.ape_nonlinear_transfer()

        # Accumulate fluxes from the smallest resolved scale (l=truncation+1) to wavenumber l.
        pi_k = np.nansum(tk_l, axis=0) - np.cumsum(tk_l, axis=0)
        pi_a = np.nansum(ta_l, axis=0) - np.cumsum(ta_l, axis=0)

        # Perform vertical integration along last axis
        pik_p = self.vertical_integration(pi_k, pressure_range=pressure_range)
        pia_p = self.vertical_integration(pi_a, pressure_range=pressure_range)

        if pi_k.ndim > 1:
            # compute mean over samples (time or any other dimension)
            return pik_p.mean(-1), pia_p.mean(-1)
        else:
            return pik_p, pia_p

    def global_diagnostics(self):

        wind_theta = self.wind * self.theta_p ** 2

        _, div_spc = self._compute_rotdiv(wind_theta)
        divh_theta = self.global_average(self._inverse_transform(div_spc), axis=0)

        return - self.ganma * divh_theta / 2.0

    @transform_io
    def ke_tendency(self, tendency):
        r"""
            Compute kinetic energy spectral transfer from parametrized
            or explicit horizontal wind tendencies.

            .. math:: \partial_{t}E_{K}(l) = (\boldsymbol{u}, \partial_{t}\boldsymbol{u})_{l}

            where :math:`\boldsymbol{u}=(u, v)` is the horizontal wind vector,
            and :math:`\partial_{t}\boldsymbol{u}` is defined by tendency.

            Parameters
            ----------
                tendency: ndarray with shape (2, nlat, nlon, ...)
                    contains momentum tendencies for each horizontal component.
            Returns
            -------
                Kinetic energy tendency due to any process described by 'tendency'
        """
        tendency = np.asarray(tendency)

        if tendency.shape != self.wind.shape:
            raise ValueError("The shape of array 'tendency' must be consistent with initialized data."
                             "Expecting {} but got {}".format(self.wind.shape, tendency.shape))

        return self._vector_spectra(self.wind, tendency)

    @transform_io
    def ape_tendency(self, tendency):
        """
            Compute Available potential energy tendency from
            parametrized or explicit temperature tendencies.
        """
        # check dimensions
        tendency = np.asarray(tendency)

        if tendency.shape != self.theta_p.shape:
            raise ValueError("The shape of array 'tendency' must be consistent with initialized data."
                             "Expecting {} but got {}".format(self.theta_p.shape, tendency.shape))

        # remove representative mean from total temperature tendency
        tendency_prn = tendency - self._representative_mean(tendency)

        # convert temperature tendency to potential temperature tendency
        theta_tendency = tendency_prn / self.exner

        # filter terrain
        theta_tendency *= np.expand_dims(self.beta, -2)

        return self._scalar_spectra(self.theta_p, theta_tendency)

    @transform_io
    def sfvp(self, vector):
        """
            Computes the streamfunction and potential of a vector field on the sphere.
        """
        # Create iterable for multiprocessing
        data_chunks = np.array_split(vector, self.jobs, axis=-1)

        # Wrapper for spherepack function: 'getvrtdivspec'
        _getpsichi = functools.partial(self.sphere.getpsichi, ntrunc=self.truncation)

        # Create pool of workers
        pool = mp.Pool(processes=self.jobs)

        # perform computations in parallel
        result = pool.starmap(_getpsichi, data_chunks, chunksize=self.chunk_size)

        # Freeing all the workers
        pool.close()
        pool.join()

        return np.concatenate(result, axis=-1)

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
        psi_grid, chi_grid = self.sfvp(self.wind)

        # Compute non-rotational components from streamfunction
        uchi, vchi = self.horizontal_gradient(chi_grid)
        chi_vector = np.stack([uchi, vchi])

        # Compute non-divergent components from velocity potential
        vpsi, upsi = self.horizontal_gradient(psi_grid)
        psi_vector = np.stack([-upsi, vpsi])

        return chi_vector, psi_vector

    # --------------------------------------------------------------------
    # Helper methods
    # --------------------------------------------------------------------
    @transform_io
    def _spectral_transform(self, scalar):
        """
        Compute spherical harmonic coefficients of a scalar function on the sphere.
        Modified for multiprocessing
        """

        # Chunks of arrays along axis=-1 for the mp mapping ...
        data_chunks = np.array_split(scalar, self.jobs, axis=-1)

        # Create pool of workers
        pool = mp.Pool(processes=self.jobs)

        # perform computations in parallel
        _grdtospec = functools.partial(self.sphere.grdtospec, ntrunc=self.truncation)

        result = pool.map(_grdtospec, data_chunks, chunksize=self.chunk_size)

        # Close pool of workers
        pool.close()
        pool.join()

        return np.concatenate(result, axis=-1)

    @transform_io
    def _inverse_transform(self, scalar_sp):
        """
            Compute spherical harmonic coefficients of a scalar function on the sphere.
            Modified for multiprocessing
        """

        # Chunks of arrays along axis=-1 for the mp mapping ...
        data_chunks = np.array_split(scalar_sp, self.jobs, axis=-1)

        # Create pool of workers
        pool = mp.Pool(processes=self.jobs)

        # perform computations in parallel
        result = pool.map(self.sphere.spectogrd, data_chunks, chunksize=self.chunk_size)

        # Close pool of workers
        pool.close()
        pool.join()

        return np.concatenate(result, axis=-1)

    @transform_io
    def horizontal_gradient(self, scalar):
        """
            Computes horizontal gradient of a scalar function on the sphere.

        Returns:
            Arrays containing gridded zonal and meridional
            components of the vector gradient.
        """

        # compute spherical harmonic coefficients:
        scalar_ml = self._spectral_transform(scalar)

        scalar_ml = self._pack_levels(scalar_ml)
        data_chunks = np.array_split(scalar_ml, self.jobs, axis=-1)

        # Create pool of workers
        pool = mp.Pool(processes=self.jobs)

        # perform computations in parallel
        result = pool.map(self.sphere.getgrad, data_chunks, chunksize=self.chunk_size)

        # Close pool of workers
        pool.close()
        pool.join()

        return np.concatenate(result, axis=-1)

    def horizontal_advection(self, scalar):
        """
        Compute the horizontal advection
        scalar: scalar field to be advected
        """
        # computes the two components of the horizontal gradient: (2, nlat, nlon, ...)
        scalar_gradient = self.horizontal_gradient(scalar)

        return self.wind * np.stack(scalar_gradient)

    def _scalar_advection(self, scalar):
        """
        Compute the horizontal advection
        scalar: scalar field to be advected
        """
        return np.nansum(self.horizontal_advection(scalar), axis=0)

    def _wind_advection(self, wind):
        r"""
        Compute the horizontal advection of the horizontal wind in 'rotation form'

        .. math:: \boldsymbol{u}\cdot\nabla_h\boldsymbol{u}=\nabla_h|\boldsymbol{u}|^2
        .. math:: + \boldsymbol{\zeta}\times\boldsymbol{u}

        where :math:`\boldsymbol{u}=(u, v)` is the horizontal wind vector,
        and :math:`\boldsymbol{\zeta}` is the vertical vorticity.

        Notes
        -----
        Advection calculated in rotation form is more robust than the standard convective form
        :math:`(\boldsymbol{u}\cdot\nabla_h)\boldsymbol{u}` around sharp discontinuities (Zang, 1991).

        Thomas A. Zang, On the rotation and skew-symmetric forms for incompressible flow simulations.
        https://doi.org/10.1016/0168-9274(91)90102-6.

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

        # Horizontal kinetic energy per unit mass in physical space
        kinetic_energy = np.sum(wind * wind, axis=0) / 2.0

        # Horizontal gradient of horizontal kinetic energy
        # (components stored along the first dimension)
        kinetic_energy_grad = self.horizontal_gradient(kinetic_energy)

        # Horizontal advection of zonal and meridional wind components
        # (components stored along the first dimension)
        return kinetic_energy_grad + self.vrt * np.stack((-wind[1], wind[0]))

    @transform_io
    def _compute_rotdiv(self, vector):
        """
        Compute the spectral coefficients of vorticity and horizontal
        divergence of a vector field with components ugrid and vgrid on the sphere.
        """

        # Create iterable for multiprocessing
        # the second dimension corresponds to the arguments of 'getvrtdivspec'
        # or the horizontal components of 'vector'
        data_chunks = np.array_split(vector, self.jobs, axis=-1)

        # Wrapper for spherepack function: 'getvrtdivspec'
        _getvrtdivspec = functools.partial(self.sphere.getvrtdivspec, ntrunc=self.truncation)

        # Create pool of workers
        pool = mp.Pool(processes=self.jobs)

        # perform computations in parallel
        result = pool.starmap(_getvrtdivspec, data_chunks, chunksize=self.chunk_size)

        # Freeing all the workers
        pool.close()
        pool.join()

        return np.concatenate(result, axis=-1)

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
        rot_asc, div_asc = self._compute_rotdiv(vector_1)

        if vector_2 is None:
            spectrum = self.cross_spectrum(rot_asc) + self.cross_spectrum(div_asc)
        else:
            rot_bsc, div_bsc = self._compute_rotdiv(vector_2)

            spectrum = self.cross_spectrum(rot_asc, rot_bsc) + \
                       self.cross_spectrum(div_asc, div_bsc)

        return spectrum / self.vector_norm

    def _vertical_gradient(self, scalar, z=None, axis=-1):
        """
            Computes vertical gradient of a scalar function d(scalar)/dz
        """

        if z is None:
            if self.leveltype == 'pressure':
                z = self.p
            else:
                raise ValueError('Height based vertical coordinate not implemented')

        return np.gradient(scalar, z, axis=axis, edge_order=2)

    def vertical_integration(self, scalar, pressure_range=None, axis=-1):
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
        axis: `int`
            axis of integration

        Returns
        -------
        `np.ndarray`
            The vertically integrated scalar
        """
        if pressure_range is None:
            pressure_range = np.sort([self.p[0], self.p[-1]])

        assert pressure_range[0] != pressure_range[1], "Inconsistent pressure levels for vertical integration"

        # find pressure surfaces where integration takes place
        press_lmask = (self.p >= pressure_range[0]) & (self.p <= pressure_range[1])
        press_layer = self.p[press_lmask]

        # Get data inside integration interval along the vertical axis
        scalar = np.take(scalar, np.where(press_lmask)[0], axis=axis)

        # Integrate scalar at pressure levels
        integrated_scalar = simpson(scalar, x=press_layer, axis=axis, even='avg') / cn.g

        return self.direction * integrated_scalar

    def global_average(self, scalar, weights=None, axis=None):
        """
        Computes the global weighted average of a scalar function on the sphere.
        The weights are initialized according to 'standard_average':
        For 'standard_average=True,' uniform weights (1/nlat) are used, giving the global mean;
        Gaussian weights (cosine of latitude) are used otherwise.

        :param scalar: nd-array with data to be averaged
        :param axis: axis of the meridional dimension.
        :param weights: 1D-array containing latitudinal weights
        :return: Global average of a scalar function
        """
        if axis is None:
            axis = 0
        else:
            axis = normalize_axis_index(axis, scalar.ndim)

        if weights is None:
            if hasattr(self, 'weights'):
                weights = self.weights
        else:
            weights = np.asarray(weights)

            if weights.size != scalar.shape[axis]:
                raise ValueError("If given, 'weights' must be a 1D array of length 'nlat'."
                                 "Expected length {} but got {}.".format(self.nlat, weights.size))

        if scalar.shape[axis] != self.nlat:
            raise ValueError("Scalar size along axis must be nlat."
                             "Expected {} and got {}".format(self.nlat, scalar.shape[axis]))

        if scalar.shape[axis + 1] != self.nlon:
            raise ValueError("Dimensions nlat and nlon must be in consecutive order.")

        # Compute area-weighted average on the sphere (using either gaussian or linear weights)
        return np.average(scalar, weights=weights, axis=axis).mean(axis=axis)

    def cross_spectrum(self, clm1, clm2=None, degrees=None, convention='power'):
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
        Returns
        -------
        array : ndarray, shape (len(degrees), ...)
            contains the 1D spectrum as a function of spherical harmonic degree.
        """

        # Get indexes of the triangular matrix with spectral coefficients
        # (move this to class init?)
        sample_shape = clm1.shape[1:]
        coeffs_size = clm1.shape[0]

        if degrees is None:
            # check if degrees are defined
            if hasattr(self, 'degrees'):
                degrees = self.degrees
            else:
                if hasattr(self, 'truncation'):
                    ntrunc = self.truncation + 1
                else:
                    ntrunc = triangular_truncation(coeffs_size)
                degrees = np.arange(ntrunc, dtype=int)
        else:
            degrees = np.asarray(degrees)
            if (degrees.ndim != 1) or (degrees.size > self.nlat):
                raise ValueError("If given, 'degrees' must be a 1D array of length <= 'nlat'."
                                 "Expected size {} and got {}".format(self.nlat, degrees.size))

        if clm2 is None:
            clm_sqd = (clm1 * clm1.conjugate()).real
        else:
            assert clm2.shape == clm1.shape, \
                "Arrays 'clm1' and 'clm2' of spectral coefficients must have the same shape. " \
                "Expected 'clm2' shape: {} got: {}".format(clm1.shape, clm2.shape)

            clm_sqd = (clm1 * clm2.conjugate()).real

        # define wavenumbers locally
        ls = self.total_wavenumber
        ms = self.zonal_wavenumber

        # Spectrum scaled by 2 to account for symmetric coefficients (ms != 0)
        # Using the normalization in equation (7) of Lambert [1984].
        clm_sqd = (np.where(ms == 0, 0.5, 1.0) * clm_sqd.T).T

        # Initialize array for the 1D energy/power spectrum shaped (truncation, ...)
        spectrum = np.zeros((degrees.size,) + sample_shape)

        # Compute spectrum as a function of total wavenumber by adding up the zonal wavenumbers.
        for ln, degree in enumerate(degrees):
            # Sum over all zonal wavenumbers <= total wavenumber
            degree_range = (ms <= degree) & (ls == degree)
            spectrum[ln] = clm_sqd[degree_range].sum(axis=0)

        if convention.lower() == 'energy':
            spectrum *= 4.0 * np.pi

        return spectrum.squeeze()

    # Functions for preprocessing data:
    def _pack_levels(self, data, order='C'):
        # pack dimensions of arrays (nlat, nlon, ...) to (nlat, nlon, samples)
        data_length = np.shape(data)[0]

        if data_length == 2:
            new_shape = np.shape(data)[:3]
        elif data_length == self.nlat:
            new_shape = np.shape(data)[:2]
        elif data_length == self.ncoeffs:
            new_shape = np.shape(data)[:1]
        else:
            raise ValueError("Inconsistent array shape: expecting "
                             "first dimension of size {} or {}.".format(self.nlat, self.ncoeffs))
        return np.reshape(data, new_shape + (-1,), order=order).squeeze()

    def _unpack_levels(self, data, order='C'):
        # unpack dimensions of arrays (nlat, nlon, samples)
        if np.shape(data)[-1] == self.nsamples * self.nlevels:
            new_shape = np.shape(data)[:-1] + (self.nsamples, self.nlevels)
            return np.reshape(data, new_shape, order=order).squeeze()
        else:
            return data

    def filter_topography(self, scalar):
        # masks scalar values pierced by the topography
        return np.expand_dims(self.beta, -2) * scalar

    def _transform_data(self, scalar, filtered=True):
        # Helper function

        # Move dimensions (nlat, nlon) forward and vertical axis last
        # (Useful for cleaner vectorized operations)
        data, data_info = prepare_data(scalar, self.axes)

        # Ensure the latitude dimension is ordered north-to-south
        if self.reverse_latitude:
            # Reverse latitude dimension
            data = np.flip(data, axis=0)

        # Ensure the surface is at index 0
        if self.direction < 0:
            # Reverse data along vertical axis
            data = np.flip(data, axis=-1)

        # Filter out interpolated subterranean data using smoothed Heaviside function
        if filtered:
            return self.filter_topography(data)
        else:
            return data

    def _representative_mean(self, scalar):
        # Computes representative mean of a scalar function
        # excluding subterranean interpolated data

        # Use global averaged beta as a normalization factor
        norm = self.global_average(self.beta, axis=0).clip(1.0e-12, None)

        # compute weighted average on gaussian grid and divide by norm
        scalar_beta = np.expand_dims(self.beta, -2) * scalar

        return self.global_average(scalar_beta, axis=0) / norm

    def _split_mean_perturbation(self, scalar):
        # Decomposes a scalar function into the representative mean
        # and perturbations with respect to the mean
        scalar_m = self._representative_mean(scalar)

        scalar_p = scalar - np.expand_dims(self.beta, -2) * scalar_m
        return scalar_m, scalar_p
