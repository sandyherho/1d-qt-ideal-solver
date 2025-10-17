# `1d-qt-ideal-solver`: 1D Quantum Tunneling Solver

[![DOI](https://zenodo.org/badge/1072081371.svg)](https://doi.org/10.5281/zenodo.17299767)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Numba](https://img.shields.io/badge/accelerated-numba-orange.svg)](https://numba.pydata.org/)
[![PyPI](https://img.shields.io/pypi/v/1d-qt-ideal-solver.svg)](https://pypi.org/project/1d-qt-ideal-solver/)

High-performance 1D quantum tunneling solver with absorbing boundary conditions. Implements split-operator Fourier method with Numba acceleration.

## Physics

Solves the time-dependent Schrödinger equation:

$i\hbar\frac{\partial \psi}{\partial t} = \hat{H}\psi = \left[-\frac{\hbar^2}{2m}\nabla^2 + V(x,t)\right]\psi$

In natural units ($\hbar = 1$, $m = 1$), this simplifies to:

$i\frac{\partial \psi}{\partial t} = \left[-\frac{1}{2}\nabla^2 + V(x,t)\right]\psi$

Split-operator evolution with absorbing boundaries:

$\psi(x, t+\delta t) = \mathcal{M}_{\text{abs}} \cdot e^{-iV\delta t/2} \cdot \mathcal{F}^{-1}\left[e^{-ik^2\delta t/2}\mathcal{F}[\psi]\right] \cdot e^{-iV\delta t/2}$

Absorbing mask (cos⁴ profile):

$\mathcal{M}(x) = \begin{cases}
1 - s\left[1 - \cos^4\left(\frac{\pi i}{2n_b}\right)\right] & \text{left boundary} \\
1 & \text{safe zone} \\
1 - s\left[1 - \cos^4\left(\frac{\pi(N-i)}{2n_b}\right)\right] & \text{right boundary}
\end{cases}$

**Natural Units Convention:**
- ℏ = 1 (reduced Planck constant)
- Electron mass m_e = 1
- Energy in eV, length in nm, time in fs

**Observables:**
- Transmission coefficient: T
- Reflection coefficient: R
- Absorbed probability: A
- Conservation: T + R + A ≈ 1

## Features

- Absorbing boundary conditions prevent spurious reflections
- Adaptive time stepping
- Numba JIT compilation
- Stochastic environments with noise and decoherence
- Visualization with zone highlighting
- NetCDF4 output

## Installation

```bash
# From PyPI
pip install 1d-qt-ideal-solver

# From source
git clone https://github.com/sandyherho/1d-qt-ideal-solver.git
cd 1d-qt-ideal-solver
pip install -e .
```

## Quick Start

**Command line:**
```bash
# Run single case
qt1d-simulate case1

# Run both cases
qt1d-simulate --all

# Custom parameters
qt1d-simulate case1 --boundary-width 3.0 --cores 8
```

**Python API:**
```python
from qt1d_ideal import QuantumTunneling1D, GaussianWavePacket

# Initialize solver
solver = QuantumTunneling1D(
    nx=2048, 
    x_min=-30.0,
    x_max=30.0,
    boundary_width=3.0,
    boundary_strength=0.03
)

# Create initial wavepacket
psi0 = GaussianWavePacket(x0=-8.0, k0=4.0, sigma=0.8)(solver.x)

# Define barrier
V = solver.rectangular_barrier(height=4.5, width=1.0)

# Solve
result = solver.solve(psi0=psi0, V=V, t_final=6.0, n_snapshots=200)

# Results
T = result['transmission_coefficient']
R = result['reflection_coefficient']
A = result['absorbed_probability']
print(f"T = {T:.4f}, R = {R:.4f}, A = {A:.4f}, T+R+A = {T+R+A:.4f}")
```

## Test Cases

| Case | Barrier Type | Height | Width | Domain |
|------|--------------|--------|-------|--------|
| 1 | Rectangular | 4.5 eV | 1.0 nm | ±30 nm |
| 2 | Gaussian | 4.0 eV | 0.8 nm | ±30 nm |

## Configuration

Key parameters (see `configs/` for examples):

```text
# Spatial grid
nx = 2048
x_min = -30.0
x_max = 30.0

# Time integration
t_final = 6.0
n_snapshots = 200

# Absorbing boundaries
boundary_width = 3.0
boundary_strength = 0.03

# Environment
noise_amplitude = 0.0
decoherence_rate = 0.0
```

## Output

**Generated files:**
- `outputs/*.nc` - NetCDF4 with wavefunction data
- `outputs/*.gif` - Animated visualization
- `logs/*.log` - Simulation diagnostics

**Reading NetCDF data:**
```python
import netCDF4 as nc

data = nc.Dataset('outputs/case1_rectangular_barrier.nc')
x = data['x'][:]
t = data['t'][:]
psi_real = data['psi_real'][:, :]
psi_imag = data['psi_imag'][:, :]
probability = data['probability'][:, :]
potential = data['potential'][:]

T = data.transmission
R = data.reflection
A = data.absorbed
```

## Citation

```bibtex
@article{qt1d_solver_2025,
  author = {Herho, Sandy H. S. and Kaban, Siti N. and Anwar, Iwan P. and
             and Trilaksono, Nurjanna J., Suwarman, Rusmawan},
  title = {{\texttt{1d-qt-ideal-solver}: 1D idealized quantum tunneling solver with absorbing boundaries}},
  journal = {xxxx},
  volume = {xxxx},
  pages = {xxxx},
  year = {xxxx},
  doi = {10.5281/xxxxx},
  url = {https://github.com/sandyherho/1d-qt-ideal-solver}
}
```

## Authors

- Sandy H. S. Herho (sandy.herho@email.ucr.edu)
- Siti N. Kaban
- Iwan P. Anwar
- Nurjanna J. Trilaksono
- Rusmawan Suwarman

## License

MIT License - See [LICENSE](LICENSE) for details.
