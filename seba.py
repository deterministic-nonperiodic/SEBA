import multiprocessing as mp
from functools import partial

import numpy as np
import spharm
import xarray as xr
from cdo import *
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
from tools import _find_latitude, _find_longitude, _find_levels
from tools import parse_dataset, prepare_data, recover_data, cumulative_flux
from tools import terrain_mask, number_chunks, transform_io, infer_gridtype, getspecindx

_private_vars = ['nlon', 'nlat', 'nlevels', 'gridtype', 'legfunc', 'rsphere']

# cdo backend
cdo_cmd = Cdo()


class EnergyBudget:
    """
        Spectral Energy Budget of a dry hydrostatic Atmosphere.
        Implements the formulation introduced by Augier and Lindborg (2013)

        Augier, P., and E. Lindborg (2013), A new formulation of the spectral energy budget
        of the atmosphere, with application to two high-resolution general circulation models,
        J. Atmos. Sci., 70, 2293–2308.
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

    def __init__(self, dataset, variables=None, ps=None, ghsl=None, leveltype='pressure',
                 truncation=None, rsphere=None, legfunc='stored', filter_terrain=False, jobs=None):

        """
        Initializing EnergyBudget instance.

        Signature
        ---------
        energy_budget =  EnergyBudget(dataset, [ps, ghsl, leveltype='pressure',
                 gridtype, truncation, rsphere, legfunc, axes, sample_axis, filter_terrain, jobs])

        Parameters
        ----------
        :param dataset: xarray.Dataset, contains 3D analysis fields
            u: horizontal wind component in the zonal direction
            v: horizontal wind component in the meridional direction
            w: height/pressure vertical velocity depending on leveltype
            t: air temperature
            p: atmospheric pressure

        # :param gridtype: type of horizontal grid ('regular', 'gaussian', 'spectral')
        :param truncation:
            Truncation limit (triangular truncation) for the spherical harmonic computation.
        :param rsphere: averaged earth radius (meters)
        :param legfunc: Indicates whether the associated Legendre polynomials are stored
                        [using O(nlat**3) memory] or recomputed on the fly when transforms are
                        requested [O(nlat**2) memory] but slower.
        """

        if not isinstance(dataset, xr.Dataset):
            raise TypeError("Input 'dataset' must be xarray.Dataset instance.")

        if variables is None:
            print("No variable names given: I am just guessing now ...")

        # Initialize variables
        data, self.coords = parse_dataset(dataset, variables=variables)

        # string needed for data preparation
        self.info_coords = ''.join([c.axis for c in self.coords]).lower()

        # Get 3D fields
        t = data.get('t')
        p = data.get('p')
        w = data.get('w')
        omega = data.get('omega')

        if w is None and omega is None:

            raise ValueError("Vertical velocity not found!")

        elif (omega is not None) and (w is None):
            # compute vertical velocity in height coordinates
            w = vertical_velocity(p, omega, t, axis=1).transpose(*t.dims)

        elif (w is not None) and (omega is None):
            # compute or load z coordinate
            omega = pressure_vertical_velocity(p, w, t).transpose(*t.dims)

        else:
            pass

        u = data.get('u').values
        v = data.get('v').values
        w = w.values
        omega = omega.values
        t = t.values
        p = p.values

        # find coordinates
        self.latitude, self.nlat = _find_latitude(dataset)
        self.longitude, self.nlon = _find_longitude(dataset)
        self.levels, self.nlevels = _find_levels(dataset)

        # Get the dimensions for levels, latitude and longitudes in the input arrays.
        self.grid_shape = (self.nlat, self.nlon)

        # size of the non-spatial dimensions
        self.samples = np.prod(u.shape) // (np.prod(self.grid_shape) * self.nlevels)

        # Get spatial dimensions
        self.leveltype = leveltype.lower()

        if self.leveltype == 'pressure':
            assert p.size == self.nlevels, "Pressure must be a 1D field with" \
                                           "size nlevels when using pressure coordinates"
        elif self.leveltype == 'height':
            assert p.shape == u.shape, "Pressure must be a 3D field when using height coordinates"
        else:
            raise ValueError("Wrong level type specification")

        # making sure array splitting gives more chunks than jobs for parallel computations
        # if not given, the chunk size is set to the cpu count.
        packed_size = self.samples * self.nlevels
        if jobs is None:
            self.jobs = min(mp.cpu_count(), packed_size)
        else:
            self.jobs = min(int(jobs), packed_size)

        self.jobs = number_chunks(packed_size, self.jobs)
        self.chunk_size = 2

        print("Running with {} workers ...".format(self.jobs))

        # Infer direction of the vertical axis and flip accordingly
        self.direction = np.sign(p[0] - p[-1]).astype(int)
        self.p = np.asarray(sorted(p, reverse=True))

        if ps is None:
            # Set first pressure level as surface pressure
            dp = np.mean(abs(np.diff(self.p)))  # extrapolate linearly from first level
            self.ps = np.broadcast_to(p.max() + dp, (self.nlat, self.nlon))
        elif np.isscalar(ps):
            self.ps = np.broadcast_to(ps, (self.nlat, self.nlon))
        else:
            if np.ndim(ps) > 2:
                self.ps = np.nanmean(ps, axis=0)
            else:
                self.ps = ps

            if np.shape(self.ps) != self.grid_shape:
                raise ValueError('If given, the surface pressure must be a scalar or a '
                                 '2D array with shape (nlat, nlon). Expected shape {}, '
                                 'but got {}!'.format(self.grid_shape, np.shape(self.ps)))

        if ghsl is None:
            self.ghsl = 0.0
        else:
            if np.shape(ghsl) != self.grid_shape:
                raise ValueError(
                    'If given, the surface height must be a 2D array with shape (nlat, nlon)'
                    'Expected shape {}, but got {}!'.format(self.grid_shape, np.shape(ghsl)))
            self.ghsl = ghsl

        # -----------------------------------------------------------------------------
        # Create SPHEREPACK object to perform the spectral computations.
        # -----------------------------------------------------------------------------
        # Inspect grid type and latitude sampling
        self.gridtype, self.latitude, self.weights = infer_gridtype(self.latitude)

        # nodes used in the Gauss-Legendre quadrature
        self.nodes = np.sin(np.deg2rad(self.latitude))

        if rsphere is None:
            self.rsphere = cn.earth_radius
        elif type(rsphere) == int or type(rsphere) == float:
            self.rsphere = abs(rsphere)
        else:
            raise ValueError("Incorrect value for 'rsphere'.")

        # Create sphere object for common functions
        self.sphere = spharm.Spharmt(self.nlon, self.nlat, gridtype=self.gridtype,
                                     rsphere=self.rsphere, legfunc=legfunc)

        # define the triangular truncation
        if truncation is None:
            self.truncation = self.nlat - 1
        else:
            self.truncation = int(truncation)

            if self.truncation < 0 or self.truncation > self.nlat - 1:
                raise ValueError('Truncation must be between 0 and {:d}'.format(self.nlat - 1, ))

        # reverse latitude array (weights are symmetric around the equator)
        self.reverse_latitude = self.latitude[0] < self.latitude[-1]
        if self.reverse_latitude:
            self.latitude = self.latitude[::-1]

        # Get spectral indexes of the zonal wavenumber and spherical harmonic degree
        self.zonal_wavenumber, self.total_wavenumber = getspecindx(self.truncation)

        # number of spectral coefficients: (truncation + 1) * (truncation + 2) / 2
        self.ncoeffs = self.total_wavenumber.size

        # get spherical harmonic degree and horizontal wavenumber (rad / meter)
        self.degrees = np.arange(self.truncation + 1, dtype=int)
        self.kappa_h = kappa_from_deg(self.degrees)

        # normalization factor for calculating vector cross-spectrum
        self.vector_norm = np.expand_dims(self.kappa_h ** 2, (-1, -2))
        self.vector_norm[0] = 1.0

        # -----------------------------------------------------------------------------
        # Preprocessing data:
        #  - Exclude interpolated subterranean data from spectral calculations.
        #  - Ensure latitude axis is oriented north-to-south.
        #  - Reverse vertical axis from the surface to the model top.
        # -----------------------------------------------------------------------------
        self.filter_terrain = filter_terrain

        if self.filter_terrain:
            self.beta = terrain_mask(self.p, self.ps, smooth=True, jobs=self.jobs)
        else:
            self.beta = np.ones((self.nlat, self.nlon, self.nlevels))

        # Reshape and reorder data dimensions for computations
        # self.data_info stores information to recover the original shape.
        self.w, self.data_info = self._transform_data(w)

        # create wind array from unfiltered wind components
        self.wind = np.stack((self._transform_data(u, filtered=False)[0],
                              self._transform_data(v, filtered=False)[0]))

        # compute thermodynamic quantities with unfiltered temperature
        self.t = self._transform_data(t, filtered=False)[0]
        self.omega = self._transform_data(omega, filtered=False)[0]

        # free up some memory
        del u, v, t, w, omega

        # Compute vorticity and divergence of the wind field
        self.vrt_spc, self.div_spc = self.vorticity_divergence()

        # Transform of divergence and vorticity to the gaussian/regular grid
        self.div = self._inverse_transform(self.div_spc)
        self.vrt = self._inverse_transform(self.vrt_spc)

        # compute the vertical wind shear before filtering to avoid sharp gradients.
        self.wind_shear = self._vertical_gradient(self.wind)

        # filtering horizontal wind after computing divergence/vorticity
        self.wind = self.filter_topography(self.wind)
        self.wind_shear = self.filter_topography(self.wind_shear)

        self.div_wind, self.rot_wind = self.helmholtz()

        # -----------------------------------------------------------------------------
        # Thermodynamic diagnostics:
        # -----------------------------------------------------------------------------
        # Compute potential temperature
        self.exner = _exner_function(self.p)
        self.theta = self.potential_temperature()

        # Compute specific volume (volume per unit mass)
        self.alpha = self.specific_volume()
        self.alpha = self.filter_topography(self.alpha)

        # Compute geopotential (Compute before applying mask!)
        self.phi = self.geopotential()

        # Compute global average of potential temperature on pressure surfaces
        # above the ground (representative mean) and the perturbations.
        # Using A&L13 formula for the representative mean results in unstable
        # profiles in most cases! We use global average instead.

        self.theta_avg = self._representative_mean(self.theta)
        self.theta_pbn = self.theta - self.theta_avg

        # self.theta_pbn = self._scalar_perturbation(self.theta)
        # self.theta_avg = self.global_average(self.theta)

        # Compute vertical gradient of potential temperature perturbations
        # (done before filtering to avoid sharp gradients at interfaces)
        self.ddp_theta_pbn = self._vertical_gradient(self.theta_pbn)

        # Apply filter to 'theta_pbn' and 'ddp_theta_pbn'
        self.theta_pbn = self.filter_topography(self.theta_pbn)
        self.ddp_theta_pbn = self.filter_topography(self.ddp_theta_pbn)

        # Static stability parameter ganma to convert from temperature variance to APE
        # using d(theta)/d(ln p) gives smoother gradients at the top/bottom boundaries.
        # ddlp_theta_avg = self._vertical_gradient(self.theta_avg, z=np.log(self.p))
        ddlp_theta_avg = self._vertical_gradient(self.theta_avg) * self.p
        self.ganma = - cn.Rd * self.exner / ddlp_theta_avg

    # -------------------------------------------------------------------------------
    # Methods for computing thermodynamic quantities
    # -------------------------------------------------------------------------------
    def _add_metadata(self, data, name, **attributes):
        """
        Add metadata and export variables as xr.DataArray
        """
        data = recover_data(data, self.data_info)
        data = xr.DataArray(data, coords=self.coords, name=name)

        for attr, value in attributes.items():
            data.attrs[attr] = value
        return data

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
        _geopotential = partial(_geopotential_height, pressure=self.p, axis=-1)

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
        return self.ganma * self._scalar_spectra(self.theta_pbn) / 2.0

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
            Spectrum of KE transfer across scales
        """

        # compute advection of the horizontal wind (using the rotational form)
        advection_term = self._wind_advection(self.wind) + self.div * self.wind / 2.0

        # compute nonlinear spectral transfer related to horizontal advection
        advective_flux = - self.get_ke_tendency(advection_term)

        # Does this term effectively cancels out after summing over all zonal wavenumber?
        vertical_trans = self._vector_spectra(self.wind_shear, self.omega * self.wind)
        vertical_trans -= self._vector_spectra(self.wind, self.omega * self.wind_shear)

        return advective_flux + vertical_trans / 2.0

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
        return - self._scalar_spectra(self.omega, self.phi)

    def ke_turbulent_flux(self):
        # Turbulent kinetic energy flux (Eq.22)
        return - self._vector_spectra(self.wind, self.omega * self.wind) / 2.0

    def ke_vertical_flux(self):
        # Vertical flux of total kinetic energy (Eq. A9)
        return self.pressure_flux() + self.ke_turbulent_flux()

    def ape_vertical_flux(self):
        # Total APE vertical flux (Eq. A10)
        vertical_flux = self._scalar_spectra(self.theta_pbn, self.omega * self.theta_pbn)

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
        sin_lat = np.sin(np.deg2rad(self.latitude))
        cos_lat = np.cos(np.deg2rad(self.latitude))

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

    def cumulative_energy_fluxes(self):

        # Compute spectral energy fluxes accumulated over zonal wavenumbers
        tk_l = self.ke_nonlinear_transfer()
        ta_l = self.ape_nonlinear_transfer()

        # Accumulate fluxes from the smallest resolved scale (truncation+1) to wavenumber l.
        pi_k = cumulative_flux(tk_l)
        pi_a = cumulative_flux(ta_l)

        if pi_k.ndim > 2:
            # compute mean over samples (time or any other dimension if needed)
            pi_k = pi_k.mean(axis=1)
            pi_a = pi_a.mean(axis=1)

        # add metadata

        return pi_k, pi_a

    def global_diagnostics(self):

        wind_theta = 0.5 * self.wind * self.theta_pbn ** 2

        _, div_spc = self._compute_rotdiv(wind_theta)

        divh_theta = self.global_average(self._inverse_transform(div_spc), axis=0)

        return - self.ganma * divh_theta

    def get_ke_tendency(self, tendency):
        r"""
            Compute kinetic energy spectral transfer from parametrized
            or explicit horizontal wind tendencies.

            .. math:: \partial_{t}E_{K}(l) = (\boldsymbol{u}, \partial_{t}\boldsymbol{u})_{l}

            where :math:`\boldsymbol{u}=(u, v)` is the horizontal wind vector,
            and :math:`\partial_{t}\boldsymbol{u}` is defined by tendency.

            Parameters
            ----------
                tendency: ndarray with shape (2, nlat, nlon, ...)
                    contains momentum tendencies for each horizontal component
                    stacked along the first axis.
            Returns
            -------
                Kinetic energy tendency due to any process given by 'tendency'.
        """
        tendency = np.asarray(tendency)

        if tendency.shape != self.wind.shape:
            raise ValueError("The shape of 'tendency' array must be "
                             "consistent with the initialized wind. Expecting {}, "
                             "but got {}".format(self.wind.shape, tendency.shape))

        return self._vector_spectra(self.wind, tendency)

    def get_ape_tendency(self, tendency):
        """
            Compute Available potential energy tendency from
            parametrized or explicit temperature tendencies.
        """
        # check dimensions
        tendency = np.asarray(tendency)

        if tendency.shape != self.theta_pbn.shape:
            raise ValueError("The shape of 'tendency' array must be "
                             "consistent with the initialized temperature. Expecting {}, "
                             "but got {}".format(self.wind.shape, tendency.shape))

        # remove representative mean from total temperature tendency
        tendency_pbn = tendency - self._representative_mean(tendency)

        # convert temperature tendency to potential temperature tendency
        theta_tendency = tendency_pbn / self.exner

        # filtering terrain
        theta_tendency = self.filter_topography(theta_tendency)

        return self.ganma * self._scalar_spectra(self.theta_pbn, theta_tendency)

    @transform_io
    def sfvp(self, vector):
        """
            Computes the streamfunction and potential of a vector field on the sphere.
        """
        # Create iterable for multiprocessing
        data_chunks = np.array_split(vector, self.jobs, axis=-1)

        # Wrapper for spherepack function: 'getvrtdivspec'
        _getpsichi = partial(self.sphere.getpsichi, ntrunc=self.truncation)

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
    def _grdtospec(self, grid):
        # helper method to perform spectral transformation using shtools
        samples = grid.shape[-1]

        clm = np.empty((self.ncoeffs, samples))

        return 2.0 * clm

    def _spectogrd(self, csp):
        # helper method to perform spectral transformation using shtools
        samples = csp.shape[-1]

        grid = np.empty((self.nlat, self.nlon, samples))

        return grid / 2.0

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
        _grdtospec = partial(self.sphere.grdtospec, ntrunc=self.truncation)

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

    def _scalar_advection(self, scalar):
        """
        Compute the horizontal advection as dot product between
        the wind vector and scalar gradient.

        scalar: scalar field to be advected
        """
        # computes the two components of the horizontal gradient: (2, nlat, nlon, ...)
        scalar_gradient = self.horizontal_gradient(scalar)

        # components of the scalar advection
        scalar_advection = self.wind * np.stack(scalar_gradient)

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
        return kinetic_energy_gradient - self.vrt * np.stack((wind[1], -wind[0]))

    @transform_io
    def _compute_rotdiv(self, vector):
        """
        Compute the spectral coefficients of vorticity and horizontal
        divergence of a vector field on the sphere.
        """

        # Create iterable for multiprocessing
        # the first dimension corresponds to the arguments of 'getvrtdivspec'
        # or the horizontal components of 'vector'
        data_chunks = np.array_split(vector, self.jobs, axis=-1)

        # Wrapper for spherepack function: 'getvrtdivspec'
        _getvrtdivspec = partial(self.sphere.getvrtdivspec, ntrunc=self.truncation)

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

        return np.gradient(scalar, z, axis=axis, edge_order=1)

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
            pressure_range = [self.p[0], self.p[-1]]
        else:
            assert pressure_range[0] != pressure_range[1], "Inconsistent pressure levels" \
                                                           " for vertical integration."

        pressure_range = np.sort(pressure_range)

        # find pressure surfaces where integration takes place
        pressure = self.p
        level_mask = (pressure >= pressure_range[0]) & (pressure <= pressure_range[1])
        # Excluding boundary points in vertical integration.
        level_mask &= (pressure != pressure[0]) & (pressure != pressure[-1])

        # convert boolean mask to array index
        level_mask = np.where(level_mask)[0]

        # Get data inside integration interval along the vertical axis
        scalar = np.take(scalar, level_mask, axis=axis)

        # Integrate scalar at pressure levels
        integrated_scalar = - simpson(scalar, x=pressure[level_mask], axis=axis, even='avg')

        return integrated_scalar / cn.g

    def global_average(self, scalar, weights=None, axis=None):
        """
        Computes the global weighted average of a scalar function on the sphere.
        The weights are initialized according to 'grid_type':
        for grid_type='gaussian' we use gaussian quadrature weights. If grid_type='regular'
        the weights are defined as the cosine of latitude. If the grid is regular and latitude
        points are not given it returns global mean with weights = 1/nlat (not recommended).

        :param scalar: nd-array with data to be averaged
        :param axis: axis of the meridional dimension.
        :param weights: 1D-array containing latitudinal weights
        :return: Global average of a scalar function
        """
        if axis is None:
            axis = 0
        else:
            axis = normalize_axis_index(axis, scalar.ndim)

        # check array dimensions
        if scalar.shape[axis] != self.nlat:
            raise ValueError("Scalar size along axis must be nlat."
                             "Expected {} and got {}".format(self.nlat, scalar.shape[axis]))

        if scalar.shape[axis + 1] != self.nlon:
            raise ValueError("Dimensions nlat and nlon must be in consecutive order.")

        if weights is None:
            if hasattr(self, 'weights'):
                weights = self.weights
        else:
            weights = np.asarray(weights)

            if weights.size != scalar.shape[axis]:
                raise ValueError("If given, 'weights' must be a 1D array of length 'nlat'."
                                 "Expected length {} but got {}.".format(self.nlat, weights.size))

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
                    ntrunc = self.truncation
                else:
                    ntrunc = triangular_truncation(coeffs_size)
                degrees = np.arange(ntrunc + 1, dtype=int)
        else:
            degrees = np.asarray(degrees)
            if (degrees.ndim != 1) or (degrees.size > self.nlat):
                raise ValueError("If given, 'degrees' must be a 1D array of length <= 'nlat'."
                                 "Expected size {} and got {}".format(self.nlat, degrees.size))

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

        # define wavenumbers locally
        ls = self.total_wavenumber
        ms = self.zonal_wavenumber

        # Multiplying by 2 to account for symmetric coefficients (ms != 0)
        clm_sqd = (np.where(ms == 0, 1.0, 2.0) * clm_sqd.T).T

        # Initialize array for the 1D energy/power spectrum shaped (truncation, ...)
        spectrum = np.zeros((degrees.size,) + sample_shape)

        # Compute spectrum as a function of total wavenumber by adding up the zonal wavenumbers.
        for ln, degree in enumerate(degrees):
            # Sum over all zonal wavenumbers <= total wavenumber
            degree_range = (ms <= degree) & (ls == degree)
            spectrum[ln] = clm_sqd[degree_range].sum(axis=0)

        # Using the normalization in equation (7) of Lambert [1984].
        spectrum /= 2.0

        if convention.lower() == 'energy':
            spectrum *= 4.0 * np.pi

        return spectrum

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
        if np.shape(data)[-1] == self.samples * self.nlevels:
            new_shape = np.shape(data)[:-1] + (self.samples, self.nlevels)
            return np.reshape(data, new_shape, order=order)
        else:
            return data

    def filter_topography(self, scalar):
        # masks scalar values pierced by the topography
        return np.expand_dims(self.beta, -2) * scalar

    def _transform_data(self, scalar, filtered=True):
        # Helper function

        # Move dimensions (nlat, nlon) forward and vertical axis last
        # (Useful for cleaner vectorized operations)
        data, data_info = prepare_data(scalar, self.info_coords)

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
            data = self.filter_topography(data)

        return data, data_info

    def _representative_mean(self, scalar):
        # Computes representative mean of a scalar function:
        # Mean over a constant pressure level excluding subterranean data.

        # Use globally averaged beta as a normalization factor
        # gives a profile of the % of points above the surface at a given level.
        norm = self.global_average(self.beta, axis=0).clip(cn.epsilon, 1.0)

        # compute weighted average on gaussian grid and divide by norm
        weighted_scalar = self.filter_topography(scalar)

        return self.global_average(weighted_scalar, axis=0) / norm

    def _split_mean_perturbation(self, scalar):
        # Decomposes a scalar function into the representative mean
        # and perturbations with respect to the mean
        scalar_avg = self._representative_mean(scalar)

        return scalar_avg, scalar - scalar_avg

    def _scalar_perturbation(self, scalar):
        # Compute scalar perturbations
        scalar_spc = self._spectral_transform(scalar)

        # set mean coefficient (ls=ms=0) to 0.0
        mean_index = (self.zonal_wavenumber == 0) & (self.total_wavenumber == 0)
        scalar_spc[mean_index] = 0.0

        return self._inverse_transform(scalar_spc)
