from datetime import date

import numpy as np
import xarray as xr
from numpy.core.numeric import normalize_axis_index

import constants as cn
from io_tools import parse_dataset, SebaDataset
from kinematics import coriolis_parameter
from spectral_analysis import triangular_truncation, kappa_from_deg
from spherical_harmonics import Spharmt
from thermodynamics import exner_function, potential_temperature
from thermodynamics import stability_parameter, vertical_velocity
from tools import inspect_gridtype, prepare_data, recover_data, recover_spectra
from tools import rotate_vector, broadcast_1dto, gradient_1d, transform_io

# declare global read-only variables
_private_vars = ['nlon', 'nlat', 'nlevels', 'gridtype']

_global_attrs = {'source': 'git@github.com:deterministic-nonperiodic/SEBA.git',
                 'institution': 'Max Planck Institute for Meteorology',
                 'title': 'Spectral Energy Budget of the Atmosphere',
                 'history': date.today().strftime('Created on %c'),
                 'references': ''}


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
            p: Atmospheric pressure. A 1D array for isobaric levels or a ND array for arbitrary
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

        # Parsing input dataset to search for required analysis fields.
        data = parse_dataset(dataset, variables=variables,
                             surface_data={'ps': ps, 'ts': None},
                             p_levels=p_levels)

        # Get the size of every dimension. The analysis is performed over 3D slices of data
        # (lat, lon, pressure) by simply iterating over the sample axis.
        self.nlat = data.latitude.size
        self.nlon = data.longitude.size
        self.nlevels = data.pressure.size
        self.samples = data.time.size

        # -----------------------------------------------------------------------------
        # Initialize a sphere object to perform the spectral transformations.
        # -----------------------------------------------------------------------------

        # Inspect grid type based on the latitude sampling
        self.gridtype, self.latitude, self.weights = inspect_gridtype(data.latitude)

        # define the triangular truncation
        if truncation is None:
            self.truncation = self.nlat - 1
        else:
            self.truncation = int(truncation)

            if self.truncation < 0 or self.truncation > self.nlat - 1:
                raise ValueError(f'Truncation must be between 0 and {self.nlat - 1}')

        # Create sphere object for spectral transforms
        self.sphere = Spharmt(self.nlon, self.nlat,
                              gridtype=self.gridtype, rsphere=rsphere,
                              ntrunc=self.truncation, jobs=jobs)

        # number of spectral coefficients: (truncation + 1) * (truncation + 2) / 2
        self.nlm = self.sphere.nlm

        # get spherical harmonic degree and horizontal wavenumber (rad / meter)
        self.degrees = np.arange(self.truncation + 1, dtype=int)
        self.kappa_h = kappa_from_deg(self.degrees)

        # Create dictionary with axis/coordinate pairs (ensure dimension order is preserved)
        # These coordinates are used to export data as xarray objects
        self.coords = data.coordinates_by_axes()

        # create horizontal wavenumber coordinates for spectral quantities.
        self.sp_coords = [c for ic, c in self.coords.items() if ic not in 'xy']

        self.sp_coords.append(xr.Coordinate('kappa', self.kappa_h,
                                            attrs={'standard_name': 'wavenumber',
                                                   'long_name': 'horizontal wavenumber',
                                                   'axis': 'X', 'units': 'm**-1'}))

        # ------------------------------------------------------------------------------------------
        # Initialize dynamic fields
        # ------------------------------------------------------------------------------------------
        self.pressure = data.pressure.values

        self.omega, self.data_info = data.get_field('omega')

        # define data mask once for consistency
        self.data_mask = self.omega.mask

        # create wind array from masked wind components (preserving mask)
        self.wind = np.ma.stack((data.get_field('u_wind')[0],
                                 data.get_field('v_wind')[0]))

        # compute thermodynamic quantities from masked temperature
        self.temperature = data.get_field('temperature')[0]

        # Compute vorticity and divergence of the wind field.
        self.vrt, self.div = self.vorticity_divergence()

        # compute the vertical wind shear before filtering to avoid sharp gradients.
        self.wind_shear = self.vertical_gradient(self.wind)

        # Perform Helmholtz decomposition
        self.wind_div, self.wind_rot = self.helmholtz()

        # Coriolis parameter (broadcast to the shape of the wind vector)
        self.fc = broadcast_1dto(coriolis_parameter(self.latitude), self.wind.shape)

        # ------------------------------------------------------------------------------------------
        # Thermodynamic diagnostics
        # ------------------------------------------------------------------------------------------
        self.exner = exner_function(self.pressure)
        self.theta = potential_temperature(self.pressure, self.temperature)

        # Get geopotential field and compute geopotential height
        self.phi = data.get_field('geopotential')[0]

        # Compute global average of potential temperature on pressure surfaces
        # above the ground (representative mean) and the perturbations.
        self.theta_avg, self.theta_prime = self._split_mean_perturbation(self.theta)

        # Compute vertical gradient of potential temperature perturbations
        self.ddp_theta_prime = self.vertical_gradient(self.theta_prime)

        # Parameter ganma to convert from temperature variance to APE
        self.ganma = stability_parameter(self.pressure, self.theta_avg, vertical_axis=-1)

    # ------------------------------------------------------------------------------------------
    # Helper function for adding metadata to fields and convert to DataArray
    # ------------------------------------------------------------------------------------------
    def add_field(self, data, name=None, gridtype='spectral', **attributes):
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

    # ------------------------------------------------------------------------------------------
    # Methods for computing diagnostics: kinetic and available potential energies
    # ------------------------------------------------------------------------------------------
    def horizontal_kinetic_energy(self):
        """
        Horizontal kinetic energy after Augier and Lindborg (2013), Eq.13
        :return:
        """
        kinetic_energy = self._vector_spectra(self.wind) / 2.0

        #  create dataset
        kinetic_energy = self.add_field(kinetic_energy, 'hke',
                                        gridtype='spectral',
                                        units='m**2 s**-2',
                                        standard_name='horizontal_kinetic_energy',
                                        long_name='horizontal kinetic energy per unit mass')
        return kinetic_energy

    def vertical_kinetic_energy(self):
        """
        Horizontal wavenumber spectra of vertical kinetic energy
        """
        if hasattr(self, 'w_wind'):
            w_wind = self.w_wind
        else:
            w_wind = vertical_velocity(self.omega, self.temperature, self.pressure)

        kinetic_energy = self._scalar_spectra(w_wind) / 2.0

        #  create dataset
        kinetic_energy = self.add_field(kinetic_energy, 'vke',
                                        gridtype='spectral', units='m**2 s**-2',
                                        standard_name='vertical_kinetic_energy',
                                        long_name='vertical kinetic energy per unit mass')
        return kinetic_energy

    def available_potential_energy(self):
        """
        Total available potential energy after Augier and Lindborg (2013), Eq.10
        """
        potential_energy = self.ganma * self._scalar_spectra(self.theta_prime) / 2.0

        potential_energy = self.add_field(potential_energy, 'ape',
                                          gridtype='spectral', units='m**2 s**-2',
                                          standard_name='available_potential_energy',
                                          long_name='available potential energy per unit mass')
        return potential_energy

    def vorticity_divergence(self):
        """
        Computes the vertical vorticity and horizontal divergence
        """
        # Spectral coefficients of vertical vorticity and horizontal wind divergence.
        vrt_spc, div_spc = self._compute_rotdiv(self.wind)

        # transform back to grid-point space preserving mask
        vrt = self._inverse_transform(vrt_spc)
        div = self._inverse_transform(div_spc)

        vrt = np.ma.masked_array(vrt, mask=self.data_mask, fill_value=0.0)
        div = np.ma.masked_array(div, mask=self.data_mask, fill_value=0.0)

        return vrt, div

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
        advection_term = self._wind_advection() + self.div * self.wind / 2.0

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
        vertical_transfer = self._vector_spectra(self.wind_shear, self.omega * self.wind_rot)
        vertical_transfer -= self._vector_spectra(self.wind_rot, self.omega * self.wind_shear)

        # Rotational effect due to Coriolis
        deformation = - self._vector_spectra(self.wind_rot, self.fc * rotate_vector(self.wind))
        deformation -= self._vector_spectra(self.wind, self.fc * rotate_vector(self.wind_rot))

        # Relative vorticity term
        deformation -= self._vector_spectra(self.wind_rot, self.vrt * rotate_vector(self.wind))
        deformation -= self._vector_spectra(self.wind, self.vrt * rotate_vector(self.wind_rot))

        return (vertical_transfer + deformation) / 2.0

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
        vertical_transfer = self._vector_spectra(self.wind_shear, self.omega * self.wind_div)
        vertical_transfer -= self._vector_spectra(self.wind_div, self.omega * self.wind_shear)

        # cross product of vertical unit vector and horizontal winds
        cross_wind = rotate_vector(self.wind)
        cross_wdiv = rotate_vector(self.wind_div)

        # Coriolis effect
        deformation = - self._vector_spectra(self.wind_div, self.fc * cross_wind)
        deformation -= self._vector_spectra(self.wind, self.fc * cross_wdiv)

        # Relative vorticity effect
        deformation -= self._vector_spectra(self.wind_div, self.vrt * cross_wind)
        deformation -= self._vector_spectra(self.wind, self.vrt * cross_wdiv)

        return (advective_flux + vertical_transfer + deformation) / 2.0

    def ape_nonlinear_transfer(self):
        """
        Available potential energy spectral transfer due to nonlinear interactions
        after Augier and Lindborg (2013), Eq.A3
        :return:
            Spherical harmonic coefficients of APE transfer across scales
        """

        # compute horizontal advection of potential temperature
        theta_advection = self._scalar_advection(self.theta_prime)
        theta_advection += self.div * self.theta_prime / 2.0

        # compute nonlinear spectral transfer related to horizontal advection
        advection_term = - self._scalar_spectra(self.theta_prime, theta_advection)

        # compute vertical transfer
        vertical_trans = self._scalar_spectra(self.ddp_theta_prime, self.omega * self.theta_prime)
        vertical_trans -= self._scalar_spectra(self.theta_prime, self.omega * self.ddp_theta_prime)

        return self.ganma * (advection_term + vertical_trans / 2.0)

    def pressure_flux(self):
        # Pressure flux (Eq.22)
        return - self._scalar_spectra(self.omega, self.phi)

    def dke_turbulent_flux(self):
        # Turbulent kinetic energy flux (Eq.22)
        return - self._vector_spectra(self.wind, self.omega * self.wind) / 2.0

    def dke_vertical_flux(self):
        # Vertical flux of total kinetic energy (Eq. A9)
        return self.pressure_flux() + self.dke_turbulent_flux()

    def ape_vertical_flux(self):
        # Total APE vertical flux (Eq. A10)
        ape_flux = self._scalar_spectra(self.theta_prime, self.omega * self.theta_prime)

        return - self.ganma * ape_flux / 2.0

    def dke_vertical_flux_divergence(self):
        # Vertical flux divergence of Divergent kinetic energy.
        # This term enters directly the energy budget formulation.
        return self.vertical_gradient(self.dke_vertical_flux())

    def ape_vertical_flux_divergence(self):
        # Vertical flux divergence of Available potential energy.
        # This term enters directly the energy budget formulation.
        return self.vertical_gradient(self.ape_vertical_flux())

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
        dke_rke_omega = -self._vector_spectra(self.wind_shear, self.omega * self.wind_rot)
        dke_rke_omega -= self._vector_spectra(self.wind_rot, self.omega * self.wind_shear)

        return dke_rke_omega / 2.0

    def conversion_dke_rke_coriolis(self):
        """Conversion from divergent to rotational energy
        """

        # Rotational effect due to the Coriolis force on the spectral
        # transfer of divergent kinetic energy
        div_term = self._vector_spectra(self.wind_div, self.fc * rotate_vector(self.wind_rot))
        rot_term = self._vector_spectra(self.wind_rot, self.fc * rotate_vector(self.wind_div))

        return (div_term - rot_term) / 2.0

    def conversion_dke_rke_vorticity(self):
        """Conversion from divergent to rotational energy
        """

        # nonlinear interaction terms
        div_term = self._vector_spectra(self.wind_div, self.vrt * rotate_vector(self.wind_rot))
        rot_term = self._vector_spectra(self.wind_rot, self.vrt * rotate_vector(self.wind_div))

        return (div_term - rot_term) / 2.0

    def diabatic_conversion(self):
        # need to estimate Latent heat release*
        return

    def coriolis_linear_transfer(self):
        # Coriolis linear transfer
        return - self._vector_spectra(self.wind, self.fc * rotate_vector(self.wind))

    def non_conservative_term(self):
        # non-conservative term J(p) in Eq. A11
        dlog_gamma = self.vertical_gradient(np.log(self.ganma))

        heat_trans = self._scalar_spectra(self.theta_prime, self.omega * self.theta_prime)

        return dlog_gamma.reshape(-1) * heat_trans

    def energy_diagnostics(self):
        """
        Computes spectral energy budget and return as dataset objects.
        """
        # Compute diagnostics
        hke = self.horizontal_kinetic_energy()
        ape = self.available_potential_energy()
        vke = self.vertical_kinetic_energy()

        return SebaDataset(xr.merge([hke, ape, vke], compat="no_conflicts"))

    def nonlinear_energy_fluxes(self):
        """
        Computes each term in spectral energy budget and return as xr.DataArray objects.
        """

        # ------------------------------------------------------------------------------------------
        # Energy conversions APE --> DKE and DKE --> RKE
        # ------------------------------------------------------------------------------------------
        c_ape_dke = self.conversion_ape_dke()

        # different contributions to DKE --> RKE
        cdr_w = self.conversion_dke_rke_vertical()
        cdr_v = self.conversion_dke_rke_vorticity()
        cdr_c = self.conversion_dke_rke_coriolis()

        # Total conversion from divergent to rotational kinetic energy.
        c_dr = cdr_w + cdr_v + cdr_c

        # Compute cumulative nonlinear spectral energy fluxes
        pi_rke = self.rke_nonlinear_transfer()
        pi_dke = self.dke_nonlinear_transfer()
        pi_ape = self.ape_nonlinear_transfer()

        # Linear transfer due to Coriolis
        lc_ke = self.coriolis_linear_transfer()

        # Create dataset to export nonlinear fluxes
        data_fluxes = SebaDataset()

        # add some relevant info to dataset
        data_fluxes.attrs.update(_global_attrs)

        # add data and metadata
        units = 'watt / kilogram'

        data_fluxes['cad'] = self.add_field(c_ape_dke, units=units,
                                            standard_name='conversion_ape_dke',
                                            long_name='conversion from available potential '
                                                      'energy to divergent kinetic energy')

        data_fluxes['cdr_w'] = self.add_field(cdr_w, units=units,
                                              standard_name='conversion_dke_rke_vertical_velocity',
                                              long_name='conversion from divergent to rotational '
                                                        'kinetic energy due to vertical velocity')

        data_fluxes['cdr_v'] = self.add_field(cdr_v, units=units,
                                              standard_name='conversion_dke_rke_vorticity',
                                              long_name='conversion from divergent to rotational '
                                                        'kinetic energy due to relative vorticity')

        data_fluxes['cdr_c'] = self.add_field(cdr_c, units=units,
                                              standard_name='conversion_dke_rke_coriolis',
                                              long_name='conversion from divergent to rotational '
                                                        'kinetic energy due to the coriolis effect')

        data_fluxes['cdr'] = self.add_field(c_dr, units=units,
                                            standard_name='conversion_dke_rke',
                                            long_name='conversion from divergent to '
                                                      'rotational kinetic energy')

        data_fluxes['pi_rke'] = self.add_field(pi_rke, units=units,
                                               standard_name='nonlinear_rke_flux',
                                               long_name='spectral transfer of rotational'
                                                         ' kinetic energy')

        data_fluxes['pi_dke'] = self.add_field(pi_dke, units=units,
                                               standard_name='nonlinear_dke_flux',
                                               long_name='spectral transfer of divergent'
                                                         ' kinetic energy')

        data_fluxes['pi_ape'] = self.add_field(pi_ape, units=units,
                                               standard_name='nonlinear_ape_flux',
                                               long_name='spectral transfer of available'
                                                         ' potential energy')

        data_fluxes['lc'] = self.add_field(lc_ke, units=units,
                                           standard_name='coriolis_transfer',
                                           long_name='coriolis linear transfer')

        # ------------------------------------------------------------------------------------------
        # Cumulative vertical fluxes of divergent kinetic energy
        # ------------------------------------------------------------------------------------------
        pf_dke = self.pressure_flux()
        tf_dke = self.dke_turbulent_flux()

        vf_dke = self.vertical_gradient(pf_dke + tf_dke)  # same as 'dke_vertical_flux_divergence'
        vf_ape = self.ape_vertical_flux_divergence()

        # add data and metadata to vertical fluxes
        data_fluxes['pf_dke'] = self.add_field(pf_dke, units='Pa * ' + units,
                                               standard_name='pressure_dke_flux',
                                               long_name='pressure flux')

        data_fluxes['tf_dke'] = self.add_field(tf_dke, units='Pa * ' + units,
                                               standard_name='turbulent_dke_flux',
                                               long_name='vertical turbulent flux'
                                                         ' of kinetic energy')

        data_fluxes['vf_dke'] = self.add_field(vf_dke, units=units,
                                               standard_name='vertical_dke_flux',
                                               long_name='vertical flux divergence'
                                                         ' of horizontal kinetic energy')

        data_fluxes['vf_ape'] = self.add_field(vf_ape, units=units,
                                               standard_name='vertical_ape_flux',
                                               long_name='vertical flux divergence'
                                                         ' of available potential energy')
        return data_fluxes

    def get_ke_tendency(self, tendency, name=None):
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

        if da_flag:
            ke_tendency = self.add_field(ke_tendency, tendency_name,
                                         gridtype='spectral', units='W m**-2',
                                         standard_name=tendency_name)
        return ke_tendency

    def get_ape_tendency(self, tendency, name=None):
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

            if tendency.shape != self.theta_prime.shape:
                raise ValueError("The shape of 'tendency' array must be "
                                 "consistent with the initialized temperature. Expecting {}, "
                                 "but got {}".format(self.wind.shape, tendency.shape))

        # remove representative mean from total temperature tendency
        # tendency -= self._representative_mean(tendency)

        # convert temperature tendency to potential temperature tendency
        theta_tendency = tendency / self.exner / cn.cp  # rate of production of internal energy

        ape_tendency = self.ganma * self._scalar_spectra(self.theta_prime, theta_tendency)

        if da_flag:
            ape_tendency = self.add_field(ape_tendency, tendency_name,
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

        # apply mask to computed gradients
        mask = self.wind.mask

        chi_grad = np.ma.masked_array(chi_grad, mask=mask, fill_value=0.0)
        psi_grad = np.ma.masked_array(psi_grad, mask=mask, fill_value=0.0)

        return chi_grad, rotate_vector(psi_grad)

    # --------------------------------------------------------------------
    # Low-level methods for spectral transformations
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
            Arrays containing gridded zonal and meridional components of the gradient vector.
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

        # returns the summed horizontal components
        return np.ma.sum(scalar_advection, axis=0)

    def _wind_advection(self):
        r"""
        Compute the horizontal advection of the horizontal wind in 'rotation form'

        .. math:: \frac{1}{2}\nabla_h|\mathbf{u}|^{2} + \mathbf{\zeta}\times\mathbf{u}

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
        kinetic_energy = np.ma.sum(self.wind * self.wind, axis=0) / 2.0

        # Horizontal gradient of horizontal kinetic energy
        # (components stored along the first dimension)
        ke_gradient = self.horizontal_gradient(kinetic_energy)

        # Horizontal advection of zonal and meridional wind components
        # (components stored along the first dimension)
        return ke_gradient + self.vrt * rotate_vector(self.wind)

    def _scalar_spectra(self, scalar_1, scalar_2=None):
        """
        Compute 2D power spectrum as a function of spherical harmonic degree of a scalar function.
        """
        scalar_1 = self._spectral_transform(scalar_1)

        if scalar_2 is not None:
            scalar_2 = self._spectral_transform(scalar_2)

        return self.cross_spectrum(scalar_1, scalar_2)

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

        # normalization factor for vector analysis n * (n + 1) / re ** 2
        norm = broadcast_1dto(self.kappa_h ** 2, spectrum.shape).clip(cn.epsilon, None)

        return spectrum / norm

    def vertical_gradient(self, scalar, vertical_axis=-1):
        """
            Computes vertical gradient of a scalar function in pressure coordinates: d(scalar)/dp
        """
        scalar_grad = gradient_1d(scalar, self.pressure, axis=vertical_axis)

        # preserve the mask of scalar if masked
        if np.ma.is_masked(scalar):
            scalar_grad = np.ma.masked_array(scalar_grad, mask=scalar.mask, fill_value=0.0)

        return scalar_grad

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

        lat_axis = normalize_axis_index(lat_axis or 0, scalar.ndim)

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
        return np.ma.mean(scalar_average, axis=lat_axis)

    def integrate_order(self, cs_lm, degrees=None):
        """Accumulates over spherical harmonic order and returns spectrum as a function of
        spherical harmonic degree.

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
        cs_lm *= broadcast_1dto(np.where(ms == 0, 1.0, 2.0), cs_lm.shape)

        # Initialize array for the 1D energy/power spectrum shaped (truncation, ...)
        spectrum = np.zeros((degrees.size,) + sample_shape)

        # Compute spectrum as a function of total wavenumber by adding up the zonal wavenumbers.
        for ln, degree in enumerate(degrees):
            # Sum over all zonal wavenumbers <= total wavenumber
            degree_range = (ms <= degree) & (ls == degree)
            spectrum[ln] = np.nansum(cs_lm[degree_range], axis=0)

        # Normalize as in equation (7) of Lambert [1984]? i.e. spectrum /= 2.0
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

    def _representative_mean(self, scalar):
        # Computes representative mean of a scalar function: weighted average
        # on gaussian or regular grid excluding masked values over a constant
        # pressure level for regions above the surface.
        return self.global_mean(scalar, lat_axis=0)

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
