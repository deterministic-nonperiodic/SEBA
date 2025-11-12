### Description
    -----------
    A collection of tools to compute the Spectral Energy Budget of a dry hydrostatic
    Atmosphere (SEBA). This package is developed for application to global numerical
    simulations of General Circulation Models (GCMs). SEBA is implemented based on the
    formalism developed by Augier and Lindborg (2013) and includes the Helmholtz decomposition
    into the rotational and divergent kinetic energy contributions to the nonlinear energy
    fluxes introduced by Li et al. (2023). The Spherical Harmonic Transforms are carried out
    with the high-performance SHTns C library. The analysis supports data sampled on a
    regular (equally spaced in longitude and latitude) or Gaussian (equally spaced in
    longitude, latitudes located at roots of ordinary Legendre polynomial of degree nlat)
    horizontal grids. The vertical grid can be arbitrary; if data is not sampled on
    pressure levels, it is interpolated to isobaric levels before the analysis.

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

## Examples

This example demonstrates how to compute and visualize **spectral energy diagnostics and nonlinear energy transfers** using **SEBA** with reanalysis or model data.

### 1. Load atmospheric data

```python
import xarray as xr
from seba.seba import EnergyBudget

# Load atmospheric 3D fields and surface pressure
file_name = "/path/to/simulations/data.nc"

# Define external surface variables. Optional: read from input model data if available, override if specified
sfc_pres = xr.open_dataset("/path/to/simulations/surface_data.nc")["sfc_pressure"]
```

---

### 2. Create an energy budget object

```python
# Create budget object from data path or directly from loaded xarray.Dataset
budget = EnergyBudget(file_name, truncation=511, ps=sfc_pres)
```

The `truncation` parameter sets the **spectral resolution** of the spherical harmonic transform used internally.
Default `None`: truncation is based on Gaussian grid resolution, e.g., n512

---

### 3. Compute energy diagnostics

```python
dataset_energy = budget.energy_diagnostics()
```

This computes **Horizontal kinetic energy (HKE)**, **Vertical kinetic energy (VKE)**, **Rotational/Divergent kinetic energy ([R/D]KE)**, and **Available potential energy (APE)** spectra for each time and pressure level.

---

### 4. Visualize energy spectra

```python
layers = {"Troposphere": [250e2, 450e2], "Stratosphere": [50e2, 250e2]}
dataset_energy.visualize_energy(model=model, layers=layers, fig_name=f"figures/energy_spectra.pdf")
```

This produces a subplot per layer of the energy spectra. Variables can be specified via the `variables` argument, e.g., `variables=['hke', 'vke', 'ape']`.

---

### 5. Compute and visualize nonlinear energy fluxes

```python
dataset_fluxes = budget.cumulative_energy_fluxes()

layers = {
    "Lower troposphere": [450e2, 850e2],
    "Free troposphere": [250e2, 450e2],
    "Stratosphere": [20e2, 250e2],
}
y_limits = {
    "Stratosphere": [-0.6, 1.0],
    "Free troposphere": [-0.6, 1.0],
    "Lower troposphere": [-0.9, 1.5],
}

dataset_fluxes.visualize_fluxes(
    model=model,
    variables=["pi_hke+pi_ape", "pi_hke", "pi_ape"],
    layers=layers, y_limits=y_limits,
    fig_name=f"figures/energy_fluxes.pdf"
)
```

This generates layer-wise flux plots showing **Cumulative nonlinear spectral energy transfers**.

---

### 6. Cross-section visualization

```python
dataset_fluxes.visualize_sections(
    variables=["cad", "vfd_dke"],
    y_limits=[1000., 98.],
    fig_name=f"figures/fluxes_section.pdf"
)
```

Plots **vertical cross-sections** of specific HKE budget terms (e.g., conversion from APE to HKE, vertical flux divergence) as a function of pressure and horizontal wavenumber.
