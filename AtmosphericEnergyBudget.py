import multiprocessing as mp
from functools import partial

import numpy as np
import spharm
from _spherepack import onedtotwod
from numpy.core.numeric import normalize_axis_tuple, normalize_axis_index
from scipy.integrate import simpson

import constants as cn
from spectral_analysis import cross_spectrum, spectrum
from thermodynamics import density as _density
from thermodynamics import exner_function as _exner_function
from thermodynamics import height_to_geopotential, geopotential_height
from thermodynamics import potential_temperature as _potential_temperature
from thermodynamics import pressure_vertical_velocity, vertical_velocity
from tools import terrain_mask, _getvrtdiv, kappa_from_deg


class EnergyBudget(object):
    """
        Spectral Energy Budget of the Atmosphere.
        Implements the formulation introduced by Augier and Lindborg (2013)

        Augier, P., and E. Lindborg (2013), A new formulation of the spectral energy budget
        of the atmosphere, with application to two high-resolution general circulation models,
        J. Atmos. Sci., 70, 2293–2308.
    """

    def __init__(self, u, v, w, t, p, ps=None, ghsl=None, leveltype='pressure',
                 gridtype='gaussian', truncation=None, rsphere=cn.earth_radius,
                 legfunc='stored', axes=None, sample_axis=None, filter_terrain=False, jobs=None):

        """
        Initializing class EnergyBudget for computing Spectral Energy Budget of the Atmosphere

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
        self.rsphere = rsphere

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

        # Make sure the shapes of the two components match.
        if u.shape != v.shape:
            raise ValueError('u and v must be the same shape')

        if t.shape != u.shape:
            raise ValueError('Temperature must be the same shape as u and v')

        if w.shape != u.shape:
            raise ValueError('w must be the same shape as u and v')

        self.datadim = u.ndim

        if self.datadim not in (2, 3, 4):
            raise ValueError('variables must be rank 2, 3 or 4 arrays')

        if axes is None:
            axes = (0, 1, 2)
        else:
            axes = normalize_axis_tuple(axes, self.datadim)

            if len(axes) not in (2, 3):
                raise ValueError('Axes must be at rank 2 or 3')

        if len(axes) == 3:
            self.vaxis = axes[0]
        else:
            self.vaxis = None

        # Get spatial dimensions
        self.leveltype = leveltype.lower()

        if leveltype == 'pressure':
            assert p.size == u.shape[axes[0]], "Pressure must be a 1D array with" \
                                               "size nlevels when using pressure coordinates"
            omega = w.copy()
            # compute vertical velocity in height coordinates
            w = vertical_velocity(p, omega, t, axis=1)

        elif leveltype == 'height':
            assert p.shape == u.shape, "Pressure must have same shape as u" \
                                       "when using height coordinates"
            # compute or load z coordinate
            omega = pressure_vertical_velocity(p, w, t)
        else:
            raise ValueError('Invalid level type: {}'.format(leveltype))

        self.gridtype = gridtype.lower()
        if self.gridtype not in ('regular', 'gaussian'):
            raise ValueError('invalid grid type: {0:s}'.format(repr(gridtype)))

        # The dimensions for levels, latitude and longitudes must be in consecutive order.
        self.vaxis = 0
        self.axes = axes

        self.nlevels, self.nlat, self.nlon = [u.shape[axis] for axis in axes]

        if sample_axis is None:
            self.samples = 1
        else:
            self.samples = u.shape[sample_axis]

        # making sure array splitting gives more chunks than jobs
        if jobs is None:
            self.jobs = min(mp.cpu_count(), self.samples * self.nlevels)
        else:
            self.jobs = min(int(jobs), self.samples * self.nlevels)

        self.direction = np.sign(p[0] - p[-1]).astype(int)
        self.p = np.asarray(sorted(p, reverse=True))

        assert self.nlevels == self.p.size, "Array p must have size nlevels"

        if ps is None:
            # Set first pressure level as surface pressure
            self.ps = p.max()
        elif np.isscalar(ps):
            self.ps = ps
        else:
            if np.shape(ps) != (self.nlat, self.nlon):
                raise ValueError('Surface pressure must be a scalar or a'
                                 ' 2D array with shape (nlat, nlon)')
            else:
                self.ps = ps

        if ghsl is None:
            self.ghsl = 0.0
        else:
            if np.shape(ghsl) != (self.nlat, self.nlon):
                raise ValueError('Surface pressure must be a scalar or a'
                                 ' 2D array with shape (nlat, nlon)')
            else:
                self.ghsl = ghsl

        # -----------------------------------------------------------------------------
        # Create a Spharmt object to perform the computations.
        # -----------------------------------------------------------------------------
        # Get latitude and gaussian quadrature weights: weights ~ cosine(lat)
        self.lats, self.weights = spharm.gaussian_lats_wts(self.nlat)

        self.sphere = spharm.Spharmt(self.nlon, self.nlat,
                                     gridtype=self.gridtype, rsphere=rsphere,
                                     legfunc=legfunc)

        if truncation is None:
            self.truncation = self.nlat - 1
        else:
            self.truncation = truncation

        if self.truncation < 0 or self.truncation + 1 > self.nlat:
            raise ValueError('truncation must be between 0 and %d' % (self.nlat - 1,))

        self.degrees = np.arange(self.truncation + 1, dtype=float)
        self.kappa = kappa_from_deg(self.degrees)

        # Compute scale for vector cross spectra (1 / kappa^2)
        self.scale = 1.0 / np.atleast_2d(self.kappa ** 2).clip(1.0e-20, None).T

        # self.l, self.m = spharm.getspecindx(self.truncation)

        # -----------------------------------------------------------------------------
        # Preprocessing data:
        #  - Exclude interpolated subterranean data from spectral calculations
        #  - Reshape input data to (nlat, nlon, samples * nlevels)
        # -----------------------------------------------------------------------------
        if filter_terrain:
            self.beta = terrain_mask(self.p, self.ps, smoothed=True, jobs=self.jobs)
        else:
            self.beta = np.ones((self.nlat, self.nlon, self.nlevels))

        self.u = self._transform_data(u)
        self.v = self._transform_data(v)
        self.t = self._transform_data(t)
        self.w = self._transform_data(w)
        self.omega = self._transform_data(omega)

        # Compute vorticity and divergence of the wind field
        self.vrt, self.div = self.vorticity_divergence()

        self.div_grd = self._inverse_transform(self.div)

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
        self.theta_avg, self.theta_p = self._split_mean_perturbations(self.theta)

        # Compute vertical gradient of averaged potential temperature profile
        self.ddp_theta = np.gradient(self.theta_avg, self.p, axis=-1, edge_order=2)

        self.ganma = - cn.Rd * self.exner / (self.p * self.ddp_theta)
        self.ganma = self.ganma.reshape(-1)  # pack samples and vertical levels

    # -------------------------------------------------------------------------------
    # Methods for computing physical processes and diagnostics
    # -------------------------------------------------------------------------------

    # Diagnose thermodynamic variables
    def potential_temperature(self):
        # computes potential temperature
        return self._pack_levels(_potential_temperature(self.p, self._unpack_levels(self.t)))

    def density(self):
        # computes air density from pressure and temperature using the gas law
        return self._pack_levels(_density(self.p, self._unpack_levels(self.t)))

    def specific_volume(self):
        return 1.0 / self.density()

    def geopotential(self):
        """
        Computes geopotential at pressure surfaces using the hypsometric equation.
        Assumes a hydrostatic atmosphere. The geopotential at pressure levels
        below the earth surface are undefined therefore set to 0

        :return: `np.ndarray`
            geopotential (J/kg)
        """

        # Topographic height in meters above sea level
        surface_height = self.ghsl[..., np.newaxis, np.newaxis]
        temperature = self._unpack_levels(self.t)

        # Compute geopotential height (implement in parallel)
        height = surface_height + geopotential_height(temperature, self.p, self.ps, axis=-1)

        # Apply smoothed terrain mask
        height = self.beta[..., np.newaxis] * np.moveaxis(height, -2, -1)
        height = self._pack_levels(np.moveaxis(height, -1, -2))

        # Convert geopotential height to geopotential
        return height_to_geopotential(height)

    # diagnose kinetic and potential energies
    def horizontal_kinetic_energy(self):
        """
        Horizontal kinetic energy after Augier and Lindborg (2013), Eq.13
        :return:
        """
        vrt_sqd = self._cspectra(self.vrt)
        div_sqd = self._cspectra(self.div)

        return self.scale * (vrt_sqd + div_sqd) / 2.0

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
        # computes the vertical vorticity and horizontal wind divergence
        return self._compute_rotdiv(self.u, self.v)

    def ke_nonlinear_transfer(self):
        """
        Kinetic energy spectral transfer due to nonlinear interactions
        after Augier and Lindborg (2013), Eq.A2
        :return:
            Spherical harmonic coefficients of KE transfer across scales
        """
        wind = np.stack((self.u, self.v))

        adv_u = self._advect_scalar(self.u)
        adv_v = self._advect_scalar(self.v)

        der_u = self.u * self.div_grd / 2.0
        der_v = self.v * self.div_grd / 2.0

        # create advection vector
        advection_term = np.stack((adv_u + der_u, adv_v + der_v))

        advective_flux = - self._vector_cross_spectra(wind, advection_term)

        # create wind shear vector
        wind_shear = np.stack((self._vertical_gradient(self.u), self._vertical_gradient(self.v)))

        turbulent_flux = self._vector_cross_spectra(wind_shear, self.omega * wind)
        turbulent_flux -= self._vector_cross_spectra(wind, self.omega * wind_shear)

        return advective_flux + turbulent_flux / 2.0

    def ape_nonlinear_transfer(self):
        """
        Available potential energy spectral transfer due to nonlinear interactions
        after Augier and Lindborg (2013), Eq.A3
        :return:
            Spherical harmonic coefficients of APE transfer across scales
        """
        # compute horizontal advection of potential temperature
        theta_advection = self._advect_scalar(self.theta_p)

        der_theta = self.theta_p * self.div_grd / 2.0

        # compute turbulent horizontal transfer
        advection_term = - self._scalar_cross_spectra(self.theta_p, theta_advection + der_theta)

        # compute turbulent vertical transfer
        theta_gradient = self._vertical_gradient(self.theta_p)

        vertical_transport = self._scalar_cross_spectra(theta_gradient, self.omega * self.theta_p)
        vertical_transport -= self._scalar_cross_spectra(self.theta_p, self.omega * theta_gradient)

        return self.ganma * (advection_term + vertical_transport / 2.0)

    def pressure_flux(self):
        # Pressure flux (Eq.22)
        return - self._scalar_cross_spectra(self.omega, self.phi)

    def ke_turbulent_flux(self):
        # Turbulent kinetic energy flux (Eq.22)
        wind = np.stack((self.u, self.v))
        return - self._vector_cross_spectra(wind, self.omega * wind) / 2.0

    def ape_turbulent_flux(self):
        # Turbulent APE vertical flux (Eq.16)
        return - self._scalar_cross_spectra(self.theta_p, self.omega * self.theta_p) / 2.0

    def ke_vertical_fluxes(self):
        # Vertical flux of total kinetic energy (Eq. A9)
        return self.pressure_flux() + self.ke_turbulent_flux()

    def ape_vertical_flux(self):
        # Total APE vertical flux (Eq. A10)
        return self.ganma * self.ape_turbulent_flux()

    def surface_fluxes(self):
        return

    def energy_conversion(self):
        # Conversion of APE to KE
        return - self._scalar_cross_spectra(self.omega, self.alpha)

    def diabatic_conversion(self):
        # need to estimate Latent heat release*
        return

    def coriolis_linear_transfer(self):
        # relevance?
        return

    def non_conservative_term(self):
        # non-conservative term J(p) in Eq. A11
        dlog_gamma = self._vertical_gradient(np.log(self.ganma))

        return - dlog_gamma.reshape(-1) * self.ape_vertical_flux()

    def sfvp(self):
        """
            The streamfunction and velocity potential
            of the flow field on the sphere.
        """
        return self.sphere.getpsichi(self.u, self.v, ntrunc=self.truncation)

    def helmholtz(self):
        """
        Compute the irrotational and non-divergent components of the wind field

        Returns:
            uchi, vchi, upsi, vpsi:
            Zonal and meridional components of irrotational and
            non-divergent wind components respectively.
        """

        # compute the streamfunction and velocity potential
        psigrid, chigrid = self.sfvp()

        # Compute non-divergent components from velocity potential
        vpsi, upsi = self._horizontal_gradient(psigrid)

        # Compute non-rotational components from streamfunction
        uchi, vchi = self._horizontal_gradient(chigrid)

        return uchi, vchi, -upsi, vpsi

    # --------------------------------------------------------------------
    # Helper methods
    # --------------------------------------------------------------------
    def _spectral_transform(self, scalar):
        """
        Compute spherical harmonic coefficients of a scalar function on the sphere.
        Modified for multiprocessing
        """

        # Chunks of arrays along axis=-1 for the mp mapping ...
        chunks = np.array_split(scalar, self.jobs, axis=-1)

        # Create pool of workers
        pool = mp.Pool(processes=self.jobs)

        # perform computations in parallel
        result = pool.map(partial(self.sphere.grdtospec, ntrunc=self.truncation), chunks)

        # Close pool of workers
        pool.close()
        pool.join()

        return np.concatenate(result, axis=-1)

    def _inverse_transform(self, scalar_sp):
        """
        Compute spherical harmonic coefficients of a scalar function on the sphere.
        Modified for multiprocessing
        """

        # Chunks of arrays along axis=-1 for the mp mapping ...
        chunks = np.array_split(scalar_sp, self.jobs, axis=-1)

        # Create pool of workers
        pool = mp.Pool(processes=self.jobs)

        # perform computations in parallel
        result = pool.map(self.sphere.spectogrd, chunks)

        # Close pool of workers
        pool.close()
        pool.join()

        return np.concatenate(result, axis=-1)

    def _horizontal_gradient(self, scalar):
        """
            Computes gradient vector of a scalar function on the sphere.

        Returns:
            Arrays containing gridded zonal and meridional
            components of the vector gradient.
        """

        # compute spherical harmonic coefficients:
        scalar_ml = self._spectral_transform(scalar)

        # Create pool of workers
        pool = mp.Pool(processes=self.jobs)

        # perform computations in parallel
        result = pool.map(self.sphere.getgrad, np.array_split(scalar_ml, self.jobs, axis=-1))

        # Close pool of workers
        pool.close()
        pool.join()

        return np.concatenate(result, axis=-1)

    def _horizontal_advection(self, scalar):
        """
        Compute the horizontal advection
        scalar: scalar field to be advected
        """
        # compute horizontal gradient
        ds_dx, ds_dy = self._horizontal_gradient(scalar)

        return self.u * ds_dx, self.v * ds_dy

    def _advect_scalar(self, scalar):
        """
        Compute the horizontal advection
        scalar: scalar field to be advected
        """
        return np.sum(self._horizontal_advection(scalar), axis=0)

    def _compute_rotdiv(self, ugrid, vgrid):
        """
        Compute the spectral coefficients of vorticity and horizontal
        divergence of a vector field with components ugrid and vgrid on the sphere.
        """

        # Chunks of arrays along axis=-1 for the mp mapping ...
        chunks = [chunk for chunk in
                  zip(np.array_split(ugrid, self.jobs, axis=-1),
                      np.array_split(vgrid, self.jobs, axis=-1))]

        # Wrapper for spherepack function: 'getvrtdivspec'
        getvrtdiv = partial(_getvrtdiv, func=self.sphere.getvrtdivspec, ntrunc=self.truncation)

        # Create pool of workers
        pool = mp.Pool(processes=self.jobs)

        # perform computations in parallel
        result = pool.map(getvrtdiv, chunks)

        # Freeing all the workers
        pool.close()
        pool.join()

        return np.concatenate(result, axis=-1)

    def _scalar_spectra(self, scalar):
        """
        Compute power spectra of a scalar function on the sphere.
        """
        return self._cspectra(self._spectral_transform(scalar))

    def _scalar_cross_spectra(self, scalar1, scalar2):
        """
        Compute spherical harmonic coefficients of a scalar function on the sphere.
        """
        s1_ml = self._spectral_transform(scalar1)
        s2_ml = self._spectral_transform(scalar2)

        return self._cspectra(s1_ml, s2_ml)

    def _vector_spectra(self, u, v):

        rot_ml, div_ml = self._compute_rotdiv(u, v)

        return self.scale * (self._cspectra(rot_ml) + self._cspectra(div_ml))

    def _vector_cross_spectra(self, a, b):
        """
        Compute spherical harmonic cross spectra between two vector fields on the sphere.
        """
        rot_uml, div_uml = self._compute_rotdiv(*a)
        rot_vml, div_vml = self._compute_rotdiv(*b)

        c_ml = self._cspectra(rot_uml, rot_vml) + self._cspectra(div_uml, div_vml)

        return self.scale * c_ml.real

    def _vertical_gradient(self, scalar, z=None):
        """
            Computes vertical gradient of a scalar function d(scalar)/dz
        """

        if z is None:
            if self.leveltype == 'pressure':
                z = self.p
            else:
                raise ValueError('Height based vertical coordinate not implemented')

        # unpack vertical dimension before computing vertical gradient
        scalar = self._unpack_levels(scalar)

        ddz_scalar = np.gradient(scalar, z, axis=-1, edge_order=2)

        return self._pack_levels(ddz_scalar)

    def vertical_integration(self, scalar, prange=None):
        """
            Computes vertical gradient of a scalar function d(scalar)/dz
        """
        if prange is None:
            prange = np.sort([self.p[0], self.p[-1]])

        assert prange[0] != prange[1], "Inconsistent pressure levels for vertical integration"

        pressure = self.p  # Check for orientation: sorted(self.p, reverse=True)

        # find pressure surfaces where integration takes place
        press_index = (pressure >= prange[0]) & (pressure <= prange[1])
        press_layer = pressure[press_index]

        # unpack vertical dimension before computing vertical gradient
        scalar = self._unpack_levels(scalar)

        integrated_scalar = simpson(scalar[..., press_index], x=press_layer, axis=-1, even='avg')

        return self.direction * self._pack_levels(integrated_scalar)

    def _cspectra(self, clm1, clm2=None):
        # Computes 1D cross spectra as a function of spherical harmonic degree
        # from spectral coefficients

        # Expand spectral coefficients from size (lmax+1)(lmax+2)/2 to (2, l, m)
        clm1_2d = np.asarray(onedtotwod(clm1, self.nlat))

        if clm2 is None:
            # Accumulate along meridional wavenumber m
            cl_sqd = spectrum(clm1_2d,
                              normalization='4pi', lmax=self.truncation,
                              convention='power', unit='per_l')
        else:
            assert clm2.shape == clm1.shape, "Arrays clm1 and clm2 of spectral" \
                                             " coefficients must have the same shape"

            clm2_2d = np.asarray(onedtotwod(clm2, self.nlat))

            cl_sqd = cross_spectrum(clm1_2d, clm2_2d,
                                    normalization='4pi', lmax=self.truncation,
                                    convention='power', unit='per_l')

        return cl_sqd.real / 2.0

    # Functions for data preprocessing:
    def _transform_data(self, data, beta=None):
        # Helper function

        # Move dimensions (nlat, nlon) forward and vertical axis last
        data = np.moveaxis(data, self.axes, (-1, 0, 1))

        # reverse data along vertical axis so the surface is at index 0
        if self.direction < 0:
            data = np.flip(data, axis=-1)

        # Filter out interpolated subterranean data using smoothed Heaviside function
        if beta is None:
            beta = self.beta[..., np.newaxis]

        data = beta * np.moveaxis(data, -2, -1)

        return np.moveaxis(data, -1, -2).reshape((self.nlat, self.nlon, -1))

    def _unpack_levels(self, data):
        trn_shape = data.shape[:-1] + (self.samples, self.nlevels)
        return data.reshape(trn_shape).squeeze()

    def _pack_levels(self, data):
        if data.ndim > 3:
            trn_shape = (self.nlat, self.nlon, -1)
        else:
            trn_shape = (data.shape[0], -1)
        return data.reshape(trn_shape).squeeze()

    def _global_average(self, scalar, axis=None):
        """
            Computes global weighted average of a scalar function on the sphere.

        param scalar: nd-array with data to be averaged
        :param axis: axis of the meridional dimension.
        :return: Global average of a scalar function
        """
        if axis is None:
            axis = 0
        else:
            axis = normalize_axis_index(axis, scalar.ndim)

        if scalar.shape[axis] != self.nlat:
            raise ValueError("Variable scalar must have size nlat along axis."
                             "Expected {} and got {}".format(self.nlat, scalar.shape[axis]))

        # Compute area-weighted average on the sphere
        # (assumes meridional and zonal dimensions are in consecutive order)
        return np.average(scalar, weights=self.weights, axis=axis).mean(axis=axis)

    def _representative_mean(self, scalar):
        # Computes representative mean of a scalar function
        # excluding subterranean interpolated data

        # Use global averaged beta as a normalization factor
        norm = self._global_average(self.beta, axis=0).clip(1.0e-12, None)

        # compute weighted average on gaussian grid and divide by norm
        weighted_scl = np.expand_dims(self.beta, -2) * self._unpack_levels(scalar)

        return self._global_average(weighted_scl, axis=0) / norm

    def _split_mean_perturbations(self, scalar):
        # Decomposes a scalar function into the representative mean
        # and perturbations with respect to the mean
        scalar_p = scalar.copy()
        scalar_m = self._representative_mean(scalar)

        print(scalar_p.shape, self.beta.shape, scalar_m.shape)

        for ij in np.ndindex(self.nlat, self.nlon):
            scalar_p[ij] -= (self.beta[ij] * scalar_m).reshape(-1)

        return scalar_m, scalar_p
