from datetime import date

import numpy as np
from numpy.core.numeric import normalize_axis_index
from xarray import IndexVariable, DataArray

import constants as cn
from fortran_libs import numeric_tools
from io_tools import parse_dataset, SebaDataset
from kinematics import coriolis_parameter
from spectral_analysis import kappa_from_deg
from spherical_harmonics import Spharmt
from thermodynamics import exner_function, potential_temperature
from thermodynamics import lorenz_parameter, vertical_velocity
from tools import prepare_data, transform_io, rotate_vector, broadcast_1dto, gradient_1d

# declare global read-only variables
_global_attrs = {'source': 'git@github.com:deterministic-nonperiodic/SEBA.git',
                 'institution': 'Leibniz Institute of Atmospheric Physics (IAP)',
                 'title': 'Spectral Energy Budget of the Atmosphere',
                 'history': date.today().strftime('Created on %c'),
                 'references': '', 'Conventions': 'CF-1.6'}


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

    def __init__(self, dataset, variables=None, p_levels=None, ps=None, hs=None,
                 truncation=None, rsphere=None):
        """
        Initializing class EnergyBudget.

        Signature
        ---------
        energy_budget =  EnergyBudget(dataset, [variables, ps, p_levels, truncation, rsphere])

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
            used names 'U' and 'temp' are not conventional CF names.

        :param truncation: int, optional, default None
            Triangular truncation for the spherical harmonic transforms. If truncation is not
            specified then 'truncation=nlat-1' is used, where 'nlat' is the number of
            latitude points.

        :param rsphere: float, optional,
            Averaged earth radius (meters), default 'rsphere = 6371200'.

        :param p_levels: iterable, optional
            Contains the pressure levels in (Pa) for vertical interpolation.
            Ignored if the data is already in pressure coordinates.
        """

        # Parsing input dataset to search for required analysis fields.
        data = parse_dataset(dataset, variables=variables, p_levels=p_levels, ps=ps, hs=hs)

        # Get the size of every dimension. The analysis is performed over 3D slices of data
        # (lat, lon, pressure) by simply iterating over the temporal axis.
        self.nlat = data.latitude.size
        self.nlon = data.longitude.size
        self.nlevels = data.pressure.size
        self.samples = data.time.size

        # -----------------------------------------------------------------------------
        # Initialize a sphere object to perform the spectral transformations.
        # -----------------------------------------------------------------------------
        self.sphere = Spharmt(self.nlat, self.nlon, ntrunc=truncation, rsphere=rsphere)

        # Compute scale factor for vector spectra: l (l + 1) / Re ** 2 -> kappa ** -2
        scale = - self.sphere.rsphere * self.sphere.inv_lap
        self.vector_scale = broadcast_1dto(scale, (self.sphere.nlm, self.samples, self.nlevels))

        # Create dictionary with name/coordinate pairs (ensure dimension order is preserved)
        # These coordinates are used to export data as SebaDataset
        self.gp_coords = data.coordinates_by_names()

        # Create coordinates for spectral quantities.
        self.sp_coords = {name: coord for name, coord in self.gp_coords.items()
                          if name not in ['latitude', 'longitude']}

        # compute horizontal wavenumber (rad / meter)
        self.kappa_h = kappa_from_deg(np.arange(self.sphere.truncation + 1, dtype=int))

        self.sp_coords['kappa'] = IndexVariable('kappa', self.kappa_h,
                                                attrs={'standard_name': 'wavenumber',
                                                       'long_name': 'horizontal wavenumber',
                                                       'axis': 'X', 'units': 'm**-1'})

        # ------------------------------------------------------------------------------------------
        # Initialize dynamic fields for the analysis
        # ------------------------------------------------------------------------------------------
        self.pressure = data.pressure.values

        # get vertical velocity
        self.omega = data.get_field('omega', masked=True)

        # create wind array from masked wind components (preserving mask)
        self.wind = np.ma.stack((data.get_field('u_wind', masked=True),
                                 data.get_field('v_wind', masked=True)))

        # Get geopotential field
        self.phi = data.get_field('geopotential', masked=False)

        # compute thermodynamic quantities. Using unmasked temperature field if possible
        # Using masked temperature causes issues computing the spectral transfers of APE,
        # as well as neutral stratification close to the surface resulting in singular APE.
        self.temperature = data.get_field('temperature', masked=False)

        # ------------------------------------------------------------------------------------------
        # Infer data mask for spectral corrections due to missing data. Here, β(lat, lon, p)
        # is a heavy side function the terrain mask containing 0 for levels satisfying p >= ps
        # and 1 otherwise (Boer 1982).
        # ------------------------------------------------------------------------------------------
        # define data mask once for consistency
        self.mask = self.omega.mask if hasattr(self.omega, 'mask') else np.zeros_like(self.omega)
        self.mask = self.mask.astype(bool)

        # Compute fraction of valid points at every level for Spectral Mode-Coupling Correction
        # same as the power-spectrum of the mask integrated along spherical harmonic degree. This
        # approximation is not accurate for big masked regions, the proper approach is to
        # compute the inverse of the Mode-Coupling matrix for many realizations of masked
        # power spectra (e.g., Cooray et al. 2012), however this is computationally too expensive!
        self.beta_correction = 1.0 / self.representative_mean((~self.mask).astype(float))

        # ------------------------------------------------------------------------------------------
        # Kinematics
        # ------------------------------------------------------------------------------------------
        # Compute vorticity and divergence from the wind field (ignore if given for consistency)
        self.vrt, self.div = self.vorticity_divergence()

        # self.vrt = data.get_field('vorticity', self.vrt)
        # self.div = data.get_field('divergence', self.div)

        # compute the vertical wind shear before filtering to avoid sharp gradients.
        self.wind_shear = self.vertical_gradient(self.wind)

        # Perform Helmholtz decomposition
        self.wind_rot, self.wind_div = self.helmholtz()

        # Horizontal gradient of kinetic energy per unit mass in physical space
        self.hke_grad = self.horizontal_gradient(np.ma.sum(self.wind ** 2, axis=0))

        # Coriolis parameter (broadcast to the shape of the wind vector)
        self.fc = broadcast_1dto(coriolis_parameter(data.latitude.values), self.vrt.shape)

        # clear some memory
        del data

        # Absolute vorticity
        self.abs_vrt = self.vrt + self.fc

        # ------------------------------------------------------------------------------------------
        # Thermodynamics
        # ------------------------------------------------------------------------------------------
        self.exner = exner_function(self.pressure)
        self.theta = potential_temperature(self.pressure, self.temperature)

        # Compute global average of potential temperature on pressure surfaces
        # above the ground (representative mean) and the perturbations.
        self.theta_avg, self.theta_prime = self._split_mean_perturbation(self.theta)

        # Lorenz's stability parameter 'ganma' used to convert from temperature variance to APE
        self.ganma = lorenz_parameter(self.pressure, self.theta_avg, vertical_axis=-1)

    # ----------------------------------------------------------------------------------------------
    # Helper function for adding metadata to fields and convert to DataArray
    # ----------------------------------------------------------------------------------------------
    def add_field(self, data, name=None, gridtype='spectral', cumulative_flux=True, **attrs):
        """
            Add metadata and export variables as xr.DataArray
        """
        expected_gridtype = ['spectral', 'gaussian', 'regular']
        if gridtype not in expected_gridtype:
            raise ValueError(f"Unknown grid type! Must be one of: {expected_gridtype}")

        if gridtype == 'spectral':
            coords = self.sp_coords
            dims = ['kappa', 'time', 'level']

            # Accumulate along spherical harmonic order (m) and return
            # spectrum as a function of spherical harmonic degree (l)
            if data.shape[0] == self.sphere.nlm:
                data = self.cumulative_spectrum(data, cumulative_flux=cumulative_flux)
        else:
            coords = self.gp_coords
            dims = ['latitude', 'longitude', 'time', 'level']

        # create xarray.DataArray... dimensions are sorted according to dims
        array = DataArray(data=data, name=name, dims=dims, coords=coords)

        # add attributes to variable
        attrs.update(gridtype=gridtype)
        array.attrs.update(attrs)

        return array.transpose('time', 'level', ...)

    # ------------------------------------------------------------------------------------------
    # Methods for computing diagnostics: kinetic and available potential energies
    # ------------------------------------------------------------------------------------------
    def horizontal_kinetic_energy(self):
        """
        Horizontal kinetic energy Augier and Lindborg (2013), Eq.13
        :return:
        """

        # Same as: kinetic_energy = self._vector_spectra(self.wind) / 2.0
        rke = self.vector_scale * self._scalar_spectrum(self.vrt) / 2.0
        dke = self.vector_scale * self._scalar_spectrum(self.div) / 2.0

        hke = rke + dke

        #  create dataset
        rke = self.add_field(rke, 'rke', cumulative_flux=False,
                             gridtype='spectral', units='m**2 s**-2',
                             standard_name='rotational_kinetic_energy',
                             long_name='horizontal kinetic energy'
                                       ' of the non-divergent wind')

        dke = self.add_field(dke, 'dke', cumulative_flux=False,
                             gridtype='spectral', units='m**2 s**-2',
                             standard_name='divergent_kinetic_energy',
                             long_name='horizontal kinetic energy'
                                       ' of the non-rotational wind')

        hke = self.add_field(hke, 'hke', cumulative_flux=False,
                             gridtype='spectral', units='m**2 s**-2',
                             standard_name='horizontal_kinetic_energy',
                             long_name='horizontal kinetic energy')

        return rke, dke, hke

    def vertical_kinetic_energy(self):
        """
        Horizontal wavenumber spectra of vertical kinetic energy
        """
        if hasattr(self, 'w_wind'):
            w_wind = self.w_wind
        else:
            w_wind = vertical_velocity(self.omega, self.temperature, self.pressure)

        vke = self._scalar_spectrum(w_wind) / 2.0

        #  create dataset
        vke = self.add_field(vke, 'vke', cumulative_flux=False,
                             gridtype='spectral', units='m**2 s**-2',
                             standard_name='vertical_kinetic_energy',
                             long_name='vertical kinetic energy')
        return vke

    def available_potential_energy(self):
        """
        Total available potential energy after Augier and Lindborg (2013), Eq.10
        """
        ape = self.ganma * self._scalar_spectrum(self.theta_prime) / 2.0

        ape = self.add_field(ape, 'ape', cumulative_flux=False,
                             gridtype='spectral', units='m**2 s**-2',
                             standard_name='available_potential_energy',
                             long_name='available potential energy')
        return ape

    def energy_diagnostics(self):
        """
        Computes kinetic energy components and potential energy and return as SebaDataset.
        """
        energy_components = SebaDataset()
        energy_components.attrs.update(_global_attrs)

        for variable in self.horizontal_kinetic_energy():
            energy_components[variable.name] = variable

        energy_components['vke'] = self.vertical_kinetic_energy()
        energy_components['ape'] = self.available_potential_energy()

        return energy_components

    def cumulative_energy_fluxes(self):
        """
        Computes each term in spectral energy budget and return as SebaDataset.
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
        c_dke_rke = cdr_w + cdr_v + cdr_c

        # Compute cumulative nonlinear spectral energy fluxes
        pi_rke = self.rke_nonlinear_transfer()
        pi_dke = self.dke_nonlinear_transfer()
        pi_ape = self.ape_nonlinear_transfer()

        pi_hke = pi_rke + pi_dke
        # Linear transfer due to Coriolis
        pi_lke = self.coriolis_linear_transfer()
        pi_nke = pi_hke - pi_lke

        # Create dataset to export nonlinear fluxes
        fluxes = SebaDataset()

        # add some relevant info to dataset
        fluxes.attrs.update(_global_attrs)

        # add data and metadata
        units = 'watt / kilogram'

        fluxes['cad'] = self.add_field(c_ape_dke, units=units, standard_name='conversion_ape_dke',
                                       long_name='conversion from available potential '
                                                 'energy to divergent kinetic energy')

        fluxes['cdr_w'] = self.add_field(cdr_w, units=units,
                                         standard_name='conversion_dke_rke_vertical_velocity',
                                         long_name='conversion from divergent to rotational '
                                                   'kinetic energy due to vertical velocity')

        fluxes['cdr_v'] = self.add_field(cdr_v, units=units,
                                         standard_name='conversion_dke_rke_vorticity',
                                         long_name='conversion from divergent to rotational '
                                                   'kinetic energy due to relative vorticity')

        fluxes['cdr_c'] = self.add_field(cdr_c, units=units,
                                         standard_name='conversion_dke_rke_coriolis',
                                         long_name='conversion from divergent to rotational '
                                                   'kinetic energy due to the coriolis effect')

        fluxes['cdr'] = self.add_field(c_dke_rke, units=units, standard_name='conversion_dke_rke',
                                       long_name='conversion from divergent to '
                                                 'rotational kinetic energy')

        fluxes['pi_rke'] = self.add_field(pi_rke, units=units, standard_name='rke_transfer',
                                          long_name='spectral transfer of rotational'
                                                    ' kinetic energy')

        fluxes['pi_dke'] = self.add_field(pi_dke, units=units, standard_name='dke_transfer',
                                          long_name='spectral transfer of divergent '
                                                    'kinetic energy')

        fluxes['pi_hke'] = self.add_field(pi_hke, units=units,
                                          standard_name='hke_transfer',
                                          long_name='spectral transfer of horizontal '
                                                    'kinetic energy')

        fluxes['pi_lke'] = self.add_field(pi_lke, units=units, standard_name='coriolis_transfer',
                                          long_name='linear coriolis transfer')

        fluxes['pi_nke'] = self.add_field(pi_nke, units=units, standard_name='nonlinear_transfer',
                                          long_name='nonlinear spectral transfer of kinetic energy')

        fluxes['pi_ape'] = self.add_field(pi_ape, units=units, standard_name='nonlinear_ape_flux',
                                          long_name='nonlinear spectral transfer of available'
                                                    ' potential energy')

        # ------------------------------------------------------------------------------------------
        # Cumulative vertical fluxes of divergent kinetic energy
        # ------------------------------------------------------------------------------------------
        # Approximation for the vertical pressure flux using mass continuity and the hydrostatic
        # approximation to avoid computing vertical gradients of spectral quantities.
        # self.geopotential_flux() + self.dke_turbulent_flux_divergence() - c_ape_dke

        vf_dke = self.dke_vertical_flux()
        vf_ape = self.ape_vertical_flux()

        vfd_dke = self.vertical_gradient(vf_dke, order=2)
        vfd_ape = self.vertical_gradient(vf_ape, order=2)

        # add data and metadata to vertical fluxes
        fluxes['vf_dke'] = self.add_field(vf_dke, units="pascal * " + units,
                                          standard_name='vertical_dke_flux',
                                          long_name='vertical flux of horizontal kinetic energy')

        fluxes['vfd_dke'] = self.add_field(vfd_dke, units=units,
                                           standard_name='vertical_dke_flux_divergence',
                                           long_name='vertical flux divergence'
                                                     ' of horizontal kinetic energy')

        fluxes['vf_ape'] = self.add_field(vf_ape, units="pascal * " + units,
                                          standard_name='vertical_ape_flux',
                                          long_name='vertical flux of available potential energy')

        fluxes['vfd_ape'] = self.add_field(vfd_ape, units=units,
                                           standard_name='vertical_ape_flux_divergence',
                                           long_name='vertical flux divergence'
                                                     ' of available potential energy')

        fluxes['vf'] = self.add_field(vf_dke + vf_ape, units="pascal * " + units,
                                      standard_name='total_vertical_flux',
                                      long_name='total vertical flux')

        fluxes['vfd'] = self.add_field(vfd_dke + vfd_ape, units=units,
                                       standard_name='total_vertical_flux_divergence',
                                       long_name='total vertical flux divergence')

        # Compute energy dissipation assuming quasi-stationary atmospheric state.
        dis_rke = - (pi_rke + c_dke_rke)
        dis_dke = - (pi_dke + c_ape_dke - c_dke_rke + vfd_dke)

        fluxes['dis_rke'] = self.add_field(dis_rke, units=units,
                                           standard_name='rke_dissipation',
                                           long_name='dissipation of rotational kinetic energy')

        fluxes['dis_dke'] = self.add_field(dis_dke, units=units,
                                           standard_name='dke_dissipation',
                                           long_name='dissipation of divergent kinetic energy')

        fluxes['dis_hke'] = self.add_field(dis_rke + dis_dke, units=units,
                                           standard_name='hke_dissipation',
                                           long_name='dissipation of horizontal kinetic energy')

        return fluxes

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
        da_flag = isinstance(tendency, DataArray)

        tendency_name = name
        if da_flag:
            if tendency_name is None:
                tendency_name = tendency.name.split("_")[-1]
            info = ''.join([tendency.gp_coords[dim].axis for dim in tendency.dims]).lower()

            tendency, _ = prepare_data(tendency.values, info)
        else:
            tendency = np.asarray(tendency)

            if tendency.shape != self.wind.shape:
                raise ValueError(f"The shape of 'tendency' array must be "
                                 f"consistent with the initialized wind."
                                 f"Expecting {self.wind.shape}, but got {tendency.shape}")

        ke_tendency = self._vector_spectrum(self.wind, tendency)

        if da_flag:
            ke_tendency = self.add_field(ke_tendency, tendency_name,
                                         gridtype='spectral', units='watt / kilogram',
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

        da_flag = isinstance(tendency, DataArray)

        tendency_name = name
        if da_flag:
            if tendency_name is None:
                tendency_name = tendency.name.split("_")[-1]
            info = ''.join([tendency.gp_coords[dim].axis for dim in tendency.dims]).lower()

            tendency, _ = prepare_data(tendency.values, info)
        else:
            # check dimensions
            tendency = np.asarray(tendency)

            if tendency.shape != self.theta_prime.shape:
                raise ValueError(f"The shape of 'tendency' array must be "
                                 f"consistent with the initialized temperature."
                                 f"Expecting {self.wind.shape}, but got {tendency.shape}")

        # remove representative mean from total temperature tendency
        # tendency -= self._representative_mean(tendency)

        # convert temperature tendency to potential temperature tendency
        theta_tendency = tendency / self.exner / cn.cp  # rate of production of internal energy

        ape_tendency = self.ganma * self._scalar_spectrum(self.theta_prime, theta_tendency)

        if da_flag:
            ape_tendency = self.add_field(ape_tendency, tendency_name,
                                          gridtype='spectral', units='watt / kilogram',
                                          standard_name=tendency_name)

        return ape_tendency

    def helmholtz(self):
        """
        Perform a Helmholtz decomposition of the horizontal wind.
        This decomposition splits the horizontal wind vector into
        irrotational and non-divergent components.

        Returns:
            upsi, vpsi, uchi, vchi:
            zonal and meridional components of the rotational and divergent winds.
        """

        # streamfunction and velocity potential
        psi_grid, chi_grid = self.streamfunction_potential(self.wind)

        # Compute non-divergent components from velocity potential
        rot_wind = rotate_vector(self.horizontal_gradient(psi_grid))

        # Compute non-rotational components from streamfunction
        div_wind = self.horizontal_gradient(chi_grid)

        return rot_wind, div_wind

    def vorticity_divergence(self):
        """
        Computes the vertical vorticity and horizontal divergence
        """
        # Spectral coefficients of vertical vorticity and horizontal wind divergence.
        vrt_spc, div_spc = self._spectral_vrtdiv(self.wind)

        # transform back to grid-point space preserving mask
        vrt = self._inverse_transform(vrt_spc, mask=self.mask)
        div = self._inverse_transform(div_spc, mask=self.mask)

        return vrt, div

    def get_divergence(self, vector, mask=None):
        """
        Computes the horizontal divergence of a vector field on the sphere
        """
        # Spectral coefficients of horizontal wind divergence.
        _, div_spc = self._spectral_vrtdiv(vector)

        return self._inverse_transform(div_spc, mask=mask)

    # -------------------------------------------------------------------------------
    # Methods for computing spectral fluxes
    # -------------------------------------------------------------------------------
    def hke_nonlinear_transfer(self):
        """
        Kinetic energy spectral transfer due to nonlinear interactions after
        Augier and Lindborg (2013), Eq.A2
        :return:
            Spectrum of KE transfer across scales
        """

        # compute advection of the horizontal wind (using the rotational form)
        # Horizontal advection of zonal and meridional wind components
        advection = self.hke_grad + 2.0 * self.vrt * rotate_vector(self.wind)
        advection += self.div * self.wind + self.omega * self.wind_shear

        # compute nonlinear spectral transfer related to horizontal advection
        hke_transfer = - self._vector_spectrum(self.wind, advection)
        hke_transfer += self._vector_spectrum(self.wind_shear, self.omega * self.wind)

        return hke_transfer / 2.0

    def rke_nonlinear_transfer(self):
        """
        Spectral transfer of rotational kinetic energy due to nonlinear interactions
        after Li et al. (2023), Eq. 28
        :return:
            Spectrum of RKE transfer across scales
        """
        # Advection by the rotational effect due to absolute vorticity
        advection_rot = self.abs_vrt * rotate_vector(self.wind_rot)
        advection_tot = self.abs_vrt * rotate_vector(self.wind) + self.omega * self.wind_shear

        # nonlinear rotational kinetic energy transfer
        rke_transfer = -self._vector_spectrum(self.wind, advection_rot)
        rke_transfer -= self._vector_spectrum(self.wind_rot, advection_tot)

        # add vertical transfer term
        rke_transfer += self._vector_spectrum(self.wind_shear, self.omega * self.wind_rot)

        return rke_transfer / 2.0

    def dke_nonlinear_transfer(self):
        """
        Spectral transfer of divergent kinetic energy due to nonlinear interactions
        after Li et al. (2023), Eq. 27. The linear Coriolis effect is included in the
        formulations so that:

        .. math:: T_{D}(l,m) + T_{R}(l,m) = T_{K}(l,m) + L(l,m)

        This implementation is optimized to reduce the number of calls to "vector_spectrum"

        :return:
            Spectrum of DKE transfer across scales
        """
        # Advection by the divergent wind in grid space
        advection_div = self.hke_grad + self.abs_vrt * rotate_vector(self.wind)
        advection_div += self.omega * self.wind_shear

        # Horizontal advection of absolute vorticity.
        advection_vrt = self.div * self.wind + self.abs_vrt * rotate_vector(self.wind_div)

        # compute nonlinear spectral transfer related to advection by the divergent wind.
        dke_transfer = - self._vector_spectrum(self.wind_div, advection_div)
        dke_transfer -= self._vector_spectrum(self.wind, advection_vrt)

        # add vertical transfer term
        dke_transfer += self._vector_spectrum(self.wind_shear, self.omega * self.wind_div)

        return dke_transfer / 2.0

    def ape_nonlinear_transfer(self):
        """
        Available potential energy spectral transfer due to nonlinear interactions
        after Augier and Lindborg (2013), Eq.A3

        :return:
            Spherical harmonic coefficients of APE transfer across scales
        """

        # Compute vertical gradient of potential temperature perturbations
        theta_gradient = self.vertical_gradient(self.theta_prime)

        # compute advection of potential temperature perturbation
        theta_advection = 2.0 * self._scalar_advection(self.theta_prime)
        theta_advection += self.div * self.theta_prime + self.omega * theta_gradient

        # compute nonlinear spectral transfer due to horizontal advection
        ape_transfer = - self._scalar_spectrum(self.theta_prime, theta_advection)
        ape_transfer += self._scalar_spectrum(theta_gradient, self.omega * self.theta_prime)

        return self.ganma * ape_transfer / 2.0

    def ape_nonlinear_transfer_1(self):
        """
        Available potential energy spectral transfer due to nonlinear interactions.
        This Function is a modified version of Eq.A3 in Augier and Lindborg (2013),
        to use simulated omega explicitly instead of mass continuity.

        :return:
            Spherical harmonic coefficients of APE transfer across scales
        """

        # Compute vertical gradient of potential temperature perturbations
        theta_omega = self.theta_prime * self.omega
        theta_gradient = self.vertical_gradient(self.theta_prime)
        theta_omega_gradient = self.vertical_gradient(theta_omega)

        # compute 3D advection of potential temperature perturbation
        theta_advection = self._scalar_advection(self.theta_prime) + self.omega * theta_gradient

        # compute nonlinear spectral transfer due to horizontal advection
        ape_transfer = - self._scalar_spectrum(self.theta_prime, theta_advection)

        # add vertical terms
        ape_transfer += self._scalar_spectrum(theta_gradient, theta_omega) / 2.0
        ape_transfer += self._scalar_spectrum(self.theta_prime, theta_omega_gradient) / 2.0

        return self.ganma * ape_transfer

    def geopotential_flux(self):
        # The geopotential flux should equal the pressure flux plus the APE to DKE conversion
        # Can be used to calculate the vertical pressure flux indirectly to avoid vertical
        # gradients of spectral coefficients.
        return self._scalar_spectrum(self.div, self.phi)

    def pressure_flux(self):
        # Pressure flux (Eq.22)
        return - self._scalar_spectrum(self.omega, self.phi)

    def dke_turbulent_flux(self):
        # Turbulent kinetic energy flux (Eq.22)
        return - self._vector_spectrum(self.wind, self.omega * self.wind) / 2.0

    def dke_vertical_flux(self):
        # Vertical flux of total kinetic energy (Eq. A9)
        return self.pressure_flux() + self.dke_turbulent_flux()

    def ape_vertical_flux(self):
        # Total APE vertical flux (Eq. A10)
        ape_flux = self._scalar_spectrum(self.theta_prime, self.omega * self.theta_prime)

        return - self.ganma * ape_flux / 2.0

    def dke_vertical_flux_divergence(self):
        # Vertical flux divergence of divergent kinetic energy.
        # This term enters directly the energy budget formulation.
        return self.vertical_gradient(self.dke_vertical_flux(), order=2)

    def dke_turbulent_flux_divergence(self):
        return self.vertical_gradient(self.dke_turbulent_flux(), order=2)

    def ape_vertical_flux_divergence(self):
        # Vertical flux divergence of Available potential energy.
        # This term enters directly the energy budget formulation.
        return self.vertical_gradient(self.ape_vertical_flux(), order=2)

    def conversion_ape_dke(self):
        # Conversion of Available Potential energy into kinetic energy (Eq. 19 of A&L)
        return - cn.Rd * self._scalar_spectrum(self.omega, self.temperature) / self.pressure

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
        dke_rke_omega = self._vector_spectrum(self.wind_shear, self.omega * self.wind_rot)
        dke_rke_omega += self._vector_spectrum(self.wind_rot, self.omega * self.wind_shear)

        return - dke_rke_omega / 2.0

    def conversion_dke_rke_coriolis(self):
        """Conversion from divergent to rotational energy
        """

        # Rotational effect due to the Coriolis force on the spectral
        # transfer of divergent kinetic energy
        div_term = self._vector_spectrum(self.wind_div, self.fc * rotate_vector(self.wind_rot))
        rot_term = self._vector_spectrum(self.wind_rot, self.fc * rotate_vector(self.wind_div))

        return (div_term - rot_term) / 2.0

    def conversion_dke_rke_vorticity(self):
        """Conversion from divergent to rotational energy
        """

        # nonlinear interaction terms
        div_term = self._vector_spectrum(self.wind_div, self.vrt * rotate_vector(self.wind_rot))
        rot_term = self._vector_spectrum(self.wind_rot, self.vrt * rotate_vector(self.wind_div))

        return (div_term - rot_term) / 2.0

    def diabatic_conversion(self):
        # need to estimate Latent heat release*
        return

    def coriolis_linear_transfer(self):
        # Coriolis linear transfer
        return - self._vector_spectrum(self.wind, self.fc * rotate_vector(self.wind))

    def non_conservative_term(self):
        # non-conservative term J(p) in Eq. A11
        dlog_gamma = self.vertical_gradient(np.log(self.ganma))

        heat_trans = self._scalar_spectrum(self.theta_prime, self.omega * self.theta_prime)

        return dlog_gamma.reshape(-1) * heat_trans

    # --------------------------------------------------------------------
    # Low-level methods for spectral transformations
    # --------------------------------------------------------------------
    @transform_io
    def _spectral_transform(self, scalar):
        """
        Compute spherical harmonic coefficients of a scalar function on the sphere.
        Wrapper around 'grdtospec' to process inputs and run in parallel.
        """
        return self.sphere.analysis(scalar)

    @transform_io
    def _inverse_transform(self, scalar_sp, mask=None):
        """
        Compute a scalar function on the sphere from spherical harmonic coefficients.
        Wrapper around 'spectogrd' to process inputs and run in parallel.
        """
        scalar = self.sphere.synthesis(scalar_sp)

        # apply mask if needed
        if np.ma.is_mask(mask) and np.shape(mask) == scalar.shape:
            scalar = np.ma.masked_array(scalar, mask=mask, fill_value=0.0)

        return scalar

    @transform_io
    def _spectral_vrtdiv(self, vector):
        """
        Compute the spectral coefficients of vorticity and horizontal divergence
        of a vector field on the sphere.
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

        # Compute horizontal gradient on grid-point space
        scalar_gradient = self.sphere.getgrad(scalar)

        # Recovering masked regions.
        mask = self.mask if not np.ma.is_masked(scalar) else scalar.mask

        return np.ma.masked_array(scalar_gradient, mask=[mask, mask], fill_value=0.0)

    def vertical_gradient(self, scalar, order=2):
        """
            Computes vertical gradient (∂φ/∂p) of a scalar function (φ) in pressure coordinates.
            Using high-order compact finite difference scheme (Lele 1992). Contiguous masked
            regions are excluded from the calculation and the original scalar mask is recovered.
        """
        return gradient_1d(scalar, self.pressure, axis=-1, order=order)

    def _scalar_advection(self, scalar):
        r"""
        Compute the horizontal advection of a scalar field on the sphere.

        Advection is computed in flux form for better conservation properties.
        This approach also has higher performance than computing horizontal gradients.

        .. math:: \mathbf{u}\cdot\nabla_h\phi = \nabla_h\cdot(\phi\mathbf{u}) - \delta\phi

        where :math:`\phi` is an arbitrary scalar, :math:`\mathbf{u}=(u, v)` is the horizontal
        wind vector, and :math:`\delta` is the horizontal wind divergence.

        Parameters:
        -----------
            scalar: `np.ndarray`
                scalar field to be advected
        Returns:
        --------
            advection: `np.ndarray`
                Array containing the advection of a scalar field.
        """
        # Same as but faster than: np.ma.sum(self.wind * self.horizontal_gradient(scalar), axis=0)

        # Spectral coefficients of the scalar flux divergence: ∇⋅(φ u)
        flux_divergence = self._spectral_vrtdiv(scalar * self.wind)[1]

        # back to grid-point space
        flux_divergence = self._inverse_transform(flux_divergence, mask=scalar.mask)

        # recover scalar advection: u⋅∇φ = ∇⋅(φu) - δφ
        scalar_advection = flux_divergence - self.div * scalar

        return scalar_advection

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

        # Horizontal advection of zonal and meridional wind components
        return self.hke_grad / 2.0 + self.vrt * rotate_vector(self.wind)

    def _scalar_spectrum(self, scalar_1, scalar_2=None):
        """
        Compute cross/spectrum of a scalar field as a function of spherical harmonic degree.
        Augier and Lindborg (2013), Eq.9
        """

        # Compute the spectral coefficients of scalar_1
        spectrum = self._spectral_transform(scalar_1)

        if scalar_2 is None:
            spectrum *= spectrum.conjugate()
        else:
            assert scalar_2.shape == scalar_1.shape, "Rank mismatch between input scalars."
            spectrum *= self._spectral_transform(scalar_2).conjugate()

        return spectrum.real

    def _vector_spectrum(self, vector_1, vector_2=None):
        """
        Compute cross/spectrum of two vector fields as a function of spherical harmonic degree.
        Augier and Lindborg (2013), Eq.12
        """

        # Compute the spectral coefficients of the vorticity and divergence of vector_1
        vector_sp1 = self._spectral_vrtdiv(vector_1)

        if vector_2 is not None:
            assert vector_2.shape == vector_1.shape, "Rank mismatch between input vectors."
            # Compute the spectral coefficients of the vorticity and divergence of vector_2
            vector_sp2 = self._spectral_vrtdiv(vector_2)
        else:
            vector_sp2 = vector_sp1

        spectrum = np.sum(vector_sp1 * vector_sp2.conjugate(), axis=0)

        return self.vector_scale * spectrum.real

    def cumulative_spectrum(self, cs_lm, cumulative_flux=False, mask_correction=True, axis=0):
        """Accumulates over spherical harmonic order and returns
           spectrum as a function of spherical harmonic degree. The resulting spectrum can be
           converted to cumulative flux if required, i.e, adding the values corresponding to
           all wavenumbers n >= l for each l.

        Signature
        ---------
        array = integrate_order(cs_lm, [degrees])

        Parameters
        ----------
        cs_lm : ndarray, shape ((ntrunc+1)*(ntrunc+2)/2, ...)
            contains the cross-spectrum of a set of spherical harmonic coefficients.
        cumulative_flux: bool, default False,
            whether to accumulate as a flux-like quantity
        mask_correction: bool,
            When true, a correction is applied to the power spectrum to eliminate
            the masking effects
        axis : int, optional
            axis of the spectral coefficients
        Returns
        -------
        array : ndarray, shape (len(degrees), ...)
            contains the 1D spectrum as a function of spherical harmonic degree.
        """

        clm_shape = list(cs_lm.shape)
        nml_shape = clm_shape.pop(axis)

        if hasattr(self.sphere, 'truncation'):
            truncation = self.sphere.truncation + 1
        else:
            truncation = numeric_tools.truncation(nml_shape)

        # reshape coefficients for consistency with fortran routine
        cs_lm = np.moveaxis(cs_lm, axis, 0).reshape((nml_shape, -1))

        # Compute spectrum as a function of spherical harmonic degree (total wavenumber).
        spectrum = numeric_tools.cumulative_spectrum(cs_lm, truncation, flux_form=cumulative_flux)

        # back to original shape
        spectrum = np.moveaxis(spectrum.reshape([truncation] + clm_shape), 0, axis)

        # Spectral Mode-Coupling Correction for masked regions (Cooray et al. 2012)
        if mask_correction:
            spectrum *= self.beta_correction

        return spectrum

    def representative_mean(self, scalar, weights=None, lat_axis=None):
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
            raise ValueError(f"Scalar size along axis must be nlat."
                             f"Expected {self.nlat} and got {scalar.shape[lat_axis]}")

        if scalar.shape[lat_axis + 1] != self.nlon:
            raise ValueError("Dimensions nlat and nlon must be in consecutive order.")

        if weights is None:
            weights = self.sphere.weights
        else:
            weights = np.asarray(weights)

            if weights.size != scalar.shape[lat_axis]:
                raise ValueError(f"If given, 'weights' must be a 1D array of length 'nlat'."
                                 f"Expected length {self.nlat} but got {weights.size}.")

        # Compute area-weighted average on the sphere (using either gaussian or linear weights)
        # Added masked-arrays support to exclude data below the surface.
        scalar_average = np.ma.average(scalar, weights=weights, axis=lat_axis)

        # mean along the longitude dimension (same as lat_axis after array reduction)
        return np.ma.mean(scalar_average, axis=lat_axis)

    def _split_mean_perturbation(self, scalar):
        # Decomposes a scalar function into the representative mean and perturbations.

        # A&L13 formula for the representative mean
        scalar_avg = self.representative_mean(scalar, lat_axis=0)

        return scalar_avg, scalar - scalar_avg
