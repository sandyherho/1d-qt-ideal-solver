# `1d-qt-ideal-solver`: Idealized 1D Quantum Tunneling Solver

[![DOI](https://zenodo.org/badge/1072081371.svg)](https://doi.org/10.5281/zenodo.17299767)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Numba](https://img.shields.io/badge/accelerated-numba-orange.svg)](https://numba.pydata.org/)
[![PyPI](https://img.shields.io/pypi/v/1d-qt-ideal-solver.svg)](https://pypi.org/project/1d-qt-ideal-solver/)

High-performance 1D quantum tunneling solver with **absorbing boundary conditions** for realistic open-system dynamics. Implements split-operator Fourier method with adaptive time stepping, Numba-accelerated computation, and optional stochastic noise for studying coherent tunneling in nanoscale barrier systems.

## Physics

Solves the time-dependent Schrödinger equation in natural units ($\hbar = m_e = 1$):

$$i\frac{\partial \psi}{\partial t} = \hat{H}\psi = \left[-\frac{1}{2}\nabla^2 + V(x,t)\right]\psi$$

The wavefunction evolves via the split-operator method with **absorbing boundaries** to prevent spurious reflections:

$$\psi(x, t+\delta t) = \mathcal{M}_{\text{abs}} \cdot e^{-iV\delta t/2} \cdot \mathcal{F}^{-1}\left[e^{-ik^2\delta t/2}\mathcal{F}[\psi]\right] \cdot e^{-iV\delta t/2}$$

where $\mathcal{M}_{\text{abs}}$ is a smooth absorbing mask at domain boundaries.

**Key Observables:**
- Transmission coefficient: $T = \int_{x>x_b} |\psi(x,t_f)|^2 dx$
- Reflection coefficient: $R = \int_{x<x_a} |\psi(x,t_f)|^2 dx$
- Absorbed probability: $A$ = probability lost at boundaries
- Conservation: $T + R + A \approx 1$

## Features

- **Absorbing Boundaries**: Smooth damping layers prevent unphysical reflections from computational domain edges
- **Adaptive Time Stepping**: Automatically adjusts $\delta t$ based on wavefunction dynamics
- **Numba JIT Compilation**: 10-100× speedup via parallel CPU execution
- **Stochastic Environments**: Ornstein-Uhlenbeck noise ($\tau_{\text{corr}}$) and decoherence ($\gamma$)
- **Physically Realistic Test Cases**: Field emission, RTD, multi-QW, STM tunneling
- **Conservation Monitoring**: Real-time validation of norm and energy conservation
- **Professional Visualization**: Parallel-rendered GIF animations
- **Compact NetCDF4 Output**: Self-describing format with full wavefunction (Re/Im) and probability density

## Installation

**From PyPI:**
```bash
pip install 1d-qt-ideal-solver
```

**From source:**
```bash
git clone https://github.com/sandyherho/1d-qt-ideal-solver.git
cd 1d-qt-ideal-solver
pip install -e .
```

## Quick Start

**Command-line interface:**
```bash
# Run individual test case
qt1d-simulate case1                    # Field emission
qt1d-simulate case2 --cores 8          # Resonant tunneling diode

# Run all 4 cases sequentially
qt1d-simulate --all

# Custom boundary parameters
qt1d-simulate case3 --boundary-width 3.0 --boundary-strength 0.15

# Custom configuration
qt1d-simulate --config myconfig.txt
```

**Python API:**
```python
from qt1d_ideal import QuantumTunneling1D, GaussianWavePacket

# Initialize solver with absorbing boundaries
solver = QuantumTunneling1D(
    nx=2048, 
    x_min=-10, 
    x_max=10,
    boundary_width=2.0,      # Absorption layer width (nm)
    boundary_strength=0.1    # Damping strength
)

# Prepare initial Gaussian wave packet
psi0 = GaussianWavePacket(x0=-5.0, k0=5.0, sigma=0.5)(solver.x)

# Define rectangular barrier: V₀ = 2 eV, width = 2 nm
V = solver.rectangular_barrier(height=2.0, width=2.0)

# Solve dynamics
result = solver.solve(
    psi0=psi0, 
    V=V, 
    t_final=5.0,
    n_snapshots=200,
    noise_amplitude=0.05,      # 50 meV stochastic noise
    decoherence_rate=0.001     # T₂ ≈ 1 ps
)

print(f"T = {result['transmission_coefficient']:.2%}")
print(f"R = {result['reflection_coefficient']:.2%}")
print(f"A = {result['absorbed_probability']:.2%}")
```

## Test Cases

Four physically realistic scenarios based on experimental systems:

| Case | System | Barrier | Application |
|------|--------|---------|-------------|
| **1** | Field Emission | W work function (4.5 eV, 1 nm) | Electron microscopy, displays |
| **2** | Resonant Tunneling Diode | GaAs/AlGaAs (0.3 eV, 1.5 nm) | High-frequency oscillators |
| **3** | Multi-Quantum Well | InGaAs/InAlAs (0.5 eV, 2 nm) | Quantum cascade lasers |
| **4** | STM Tunneling | Vacuum gap (4 eV, 0.8 nm) | Surface imaging, manipulation |

## Output Files

**Simulation data** (in `outputs/`):
- `*.nc` — NetCDF4 format with Re(ψ), Im(ψ), |ψ|², V(x), and metadata
- `*.gif` — High-quality animations with statistics overlay

**Diagnostics** (in `logs/`):
- `*.log` — Complete simulation records with conservation diagnostics

## Reading NetCDF Output

**Python:**
```python
import netCDF4 as nc
import numpy as np

data = nc.Dataset('outputs/case1_field_emission.nc')

# Load coordinates and wavefunction
x = data['x'][:]
t = data['t'][:]
psi_real = data['psi_real'][:, :]
psi_imag = data['psi_imag'][:, :]
prob = data['probability'][:, :]

# Reconstruct complex wavefunction
psi = psi_real + 1j * psi_imag

# Extract phase and current density
phase = np.angle(psi)
dpsi_dx = np.gradient(psi, x, axis=1)
J = np.imag(np.conj(psi) * dpsi_dx)  # Probability current

# Read results
T = data.transmission
R = data.reflection
A = data.absorbed

data.close()
```

**MATLAB:**
```matlab
x = ncread('outputs/case1_field_emission.nc', 'x');
psi_real = ncread('outputs/case1_field_emission.nc', 'psi_real');
psi_imag = ncread('outputs/case1_field_emission.nc', 'psi_imag');
psi = psi_real + 1i * psi_imag;
```

## Physical Applications

Relevant for studying quantum tunneling in:
- **Field Emission**: Metal surfaces, electron sources (nm–μm scale)
- **Semiconductor Devices**: RTDs, quantum cascade lasers (nm scale)
- **Surface Science**: STM imaging, atomic manipulation (Å–nm scale)
- **Nuclear Physics**: α-decay barrier penetration (fm scale, rescaled)

## Absorbing Boundary Conditions

Smooth cosine-squared damping at domain edges:

$$\mathcal{M}(x) = \begin{cases}
\cos^2\left(\frac{\pi(x-x_{\text{min}})}{2w}\right) & x < x_{\text{min}} + w \\
1 & x_{\text{min}} + w \leq x \leq x_{\text{max}} - w \\
\cos^2\left(\frac{\pi(x_{\text{max}}-x)}{2w}\right) & x > x_{\text{max}} - w
\end{cases}$$

where $w$ is the boundary width. This **prevents reflections** while maintaining stability.

## Citation

If you use this solver in your research, please cite:

```bibtex
@software{qt1d_solver_2025,
  author = {Kaban, Siti N. and Herho, Sandy H. S. and Prayogo, Sonny and Anwar, Iwan P.},
  title = {1D Quantum Tunneling Solver with Absorbing Boundaries},
  year = {2025},
  version = {0.0.5},
  url = {https://github.com/sandyherho/1d-qt-ideal-solver},
  license = {MIT}
}
```

## Authors

- Siti N. Kaban
- Sandy H. S. Herho (sandy.herho@email.ucr.edu)
- Sonny Prayogo
- Iwan P. Anwar

## License

MIT License — See [LICENSE](LICENSE) for details.
