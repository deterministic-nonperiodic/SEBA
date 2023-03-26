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
