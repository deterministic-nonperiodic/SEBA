import numpy as np

# avoid dividing by zero (using machine precision eps)
epsilon = np.finfo(np.float64).eps  # fun fact: eps = abs(7./3 - 4./3 - 1).

earth_radius = 6.3712e6            # Radius of Earth (m)
Omega = 2.0 * np.pi / 24. / 3600.  # Earth's rotation rate, (s**(-1))
g = 9.80665                        # gravitational acceleration (m / s**2)

Rd = 287.058     # gas constant for dry air (J / kg / K)
Rv = 461.5       # gas constant for water vapor (J / kg / K)

cp = 1004.       # specific heat at constant pressure for dry air (J / kg / K)
cpv = 1875.      # specific heat at constant pressure for water vapor (J / kg / K)

eps = Rd / Rv    # molecular_weight_ratio (dimensionless)
chi = Rd / cp    # ~2/7 (dimensionless)
gamma = 6.5      # Temperature lapse rate units('K/km')

ps = 1000e2     # reference surface pressure (Pa)
t0 = 288.        # reference surface temperature (K)
rho_w = 1000.    # density of water (kg / m**3)
cw = 4181.3      # specific heat of liquid water (J / kg / K)

Lh_vap = 2.5E6    # Latent heat of vaporization (J / kg)
Lh_sub = 2.834E6  # Latent heat of sublimation (J / kg)
Lh_fus = Lh_sub - Lh_vap  # Latent heat of fusion (J / kg)
