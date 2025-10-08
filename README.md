# `1d-qt-ideal-solver`: Idealized 1D Quantum Tunneling Solver

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Numba](https://img.shields.io/badge/accelerated-numba-orange.svg)](https://numba.pydata.org/)

**Educational and research tool for simulating idealized 1D quantum tunneling phenomena.**

⚠️ **IMPORTANT**: This package simulates **IDEALIZED** quantum tunneling for educational purposes. 
The models are simplified representations capturing essential physics, not full quantum mechanical calculations.

## Features

- **Adaptive time stepping** for optimal accuracy and efficiency
- **Numba JIT compilation** for 10-100x speedup
- **Multi-core parallelization** support
- **Split-operator method** for accurate time evolution
- **6 idealized scenarios** with increasing complexity
- **Parsimonious NetCDF output** (essential variables only)
- **Professional GIF animations**

## Physics

Solves the 1D time-dependent Schrödinger equation (idealized):

```
iℏ ∂ψ/∂t = -ℏ²/2m ∂²ψ/∂x² + V(x)ψ
```

Using the split-operator method with adaptive time stepping for numerical stability.

**Note**: These are idealized models. Real quantum systems involve additional complexities 
(many-body effects, decoherence, measurement, etc.) not captured here.

## Installation

### Install from PyPI (Recommended)

```bash
pip install 1d-qt-ideal-solver
```

### Install from Source

```bash
git clone https://github.com/yourusername/1d-qt-ideal-solver.git
cd 1d-qt-ideal-solver
pip install .
```

### Development Installation

```bash
git clone https://github.com/yourusername/1d-qt-ideal-solver.git
cd 1d-qt-ideal-solver
pip install -e .
```

## Quick Start

```bash
# Run rectangular barrier simulation
qt1d-simulate rect-barrier

# Run with 8 CPU cores
qt1d-simulate rect-barrier --cores 8

# Run all 6 scenarios
qt1d-simulate --all

# Use custom configuration
qt1d-simulate --config my_config.txt
```

### Python API

```python
import numpy as np
from qt1d_ideal import QuantumTunneling1D, GaussianWavePacket

# Create solver with adaptive time stepping
solver = QuantumTunneling1D(
    nx=2048,
    x_min=-10.0,
    x_max=10.0,
    adaptive_dt=True,  # Enable adaptive time stepping
    n_cores=8
)

# Initialize Gaussian wave packet
psi0 = GaussianWavePacket(x0=-5.0, k0=5.0, sigma=0.5)(solver.x)

# Create rectangular barrier
V = solver.rectangular_barrier(height=2.0, width=2.0)

# Run simulation
result = solver.solve(
    psi0=psi0,
    V=V,
    t_final=5.0,
    n_snapshots=200,
    particle_mass=1.0
)

# Access results
print(f"Transmission: {result['transmission_coefficient']:.2%}")
print(f"Reflection: {result['reflection_coefficient']:.2%}")
```

## Idealized Test Cases

All scenarios use **idealized models** capturing essential physics:

1. **rect-barrier** - Classic rectangular potential barrier
2. **gaussian-barrier** - Smooth Gaussian barrier  
3. **step-potential** - Half-infinite step (reflection/transmission)
4. **double-barrier** - Resonant tunneling through double barrier
5. **triple-barrier** - Complex multi-barrier resonances
6. **periodic-potential** - Band structure in periodic potential (solid-state analog)

## Configuration Example

```text
# Rectangular Barrier (Idealized)
scenario_name = Rectangular Barrier
nx = 2048
x_min = -10.0
x_max = 10.0
t_final = 5.0
n_snapshots = 200
barrier_type = rectangular
barrier_height = 2.0
barrier_width = 2.0
particle_mass = 1.0
x0 = -5.0
k0 = 5.0
sigma = 0.5
adaptive_dt = true
dt_min = 0.0001
dt_max = 0.01
n_cores = 8
```

## Adaptive Time Stepping

The solver automatically adjusts time step based on:
- Wavefunction evolution rate
- Numerical stability criteria  
- User-specified min/max bounds

This ensures accuracy while optimizing computational efficiency.

## Performance

```bash
# Use all CPU cores (default)
qt1d-simulate rect-barrier

# Specify cores
qt1d-simulate rect-barrier --cores 4

# Single core
qt1d-simulate rect-barrier --cores 1
```

## Output

- **NetCDF files**: Parsimonious format with essential variables
  - Wavefunction (real & imaginary parts)
  - Probability density |ψ|²
  - Potential V(x)
  - Transmission/reflection coefficients
  
- **Animated GIFs**: Wavefunction evolution visualization

- **Log files**: Simulation parameters and performance

## Requirements

- Python 3.8+
- NumPy
- SciPy  
- Matplotlib
- netCDF4
- tqdm
- Numba

## Authors

- **Siti N. Kaban**
- **Sandy H. S. Herho** (sandy.herho@email.ucr.edu)
- **Sonny Prayogo**
- **Iwan P. Anwar**

## License

MIT License - see [LICENSE](LICENSE) file.

## Citation

```bibtex
@software{qt1d_ideal_2025,
  title = {Idealized 1D Quantum Tunneling Solver},
  author = {Kaban, Siti N. and Herho, Sandy H. S. and Prayogo, Sonny},
  year = {2025},
  version = {0.1.0},
  note = {Educational tool for idealized quantum tunneling simulations},
  url = {https://github.com/yourusername/1d-qt-ideal-solver}
}
```

## Disclaimer

This software provides **idealized simulations** for educational and research purposes. 
The models are simplified to capture essential quantum tunneling physics but do not 
represent complete quantum mechanical calculations. Use appropriately for your application.
