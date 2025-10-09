# `1d-qt-ideal-solver`: Idealized 1D Quantum Tunneling Solver

[![DOI](https://zenodo.org/badge/1072081371.svg)](https://doi.org/10.5281/zenodo.17299767)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Numba](https://img.shields.io/badge/accelerated-numba-orange.svg)](https://numba.pydata.org/)
[![PyPI](https://img.shields.io/pypi/v/1d-qt-ideal-solver.svg)](https://pypi.org/project/1d-qt-ideal-solver/)

High-performance 1D quantum tunneling solver implementing the split-operator Fourier method with adaptive time stepping, Numba-accelerated parallel computation, and optional stochastic noise for idealized simulations of coherent tunneling dynamics in nanoscale barrier systems.

## Physics

Solves the time-dependent Schrödinger equation in natural units ($\hbar = m_e = 1$):

$$i\frac{\partial \psi}{\partial t} = \hat{H}\psi = \left[-\frac{1}{2}\nabla^2 + V(x,t)\right]\psi$$

The wavefunction evolves via the split-operator method:

$$\psi(x, t+\delta t) = e^{-iV\delta t/2} \cdot \mathcal{F}^{-1}\left[e^{-ik^2\delta t/2}\mathcal{F}[\psi]\right] \cdot e^{-iV\delta t/2}$$

**Key Observables:**
- Transmission coefficient: $T = \int_{x>x_b} |\psi(x,t_f)|^2 dx$
- Reflection coefficient: $R = \int_{x<x_a} |\psi(x,t_f)|^2 dx$
- Unitarity: $T + R \approx 1$

## Features

- **Adaptive Time Stepping**: Automatically adjusts $\delta t$ based on wavefunction dynamics
- **Numba JIT Compilation**: 10-100× speedup via parallel CPU execution
- **Stochastic Environments**: Ornstein-Uhlenbeck noise ($\tau_{\text{corr}}$) and decoherence ($\gamma$)
- **Conservation Monitoring**: Real-time validation of $\|\psi\|^2 = 1$ and energy conservation
- **Professional Visualization**: Parallel-rendered GIF animations with publication-quality aesthetics
- **NetCDF4 Output**: Self-describing, compressed data format for reproducible research

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
qt1d-simulate case1                    # Rectangular barrier
qt1d-simulate case2 --cores 8          # Double barrier (8 cores)

# Run all 4 cases sequentially
qt1d-simulate --all

# Custom configuration
qt1d-simulate --config myconfig.txt
```

**Python API:**
```python
from qt1d_ideal import QuantumTunneling1D, GaussianWavePacket

# Initialize solver
solver = QuantumTunneling1D(nx=2048, x_min=-10, x_max=10)

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
```

## Output Files

**Simulation data** (in `outputs/`):
- `*.nc` — NetCDF4 format (wavefunction evolution, parameters, metadata)
- `*.gif` — High-quality animations with statistics overlay

**Diagnostics** (in `logs/`):
- `*.log` — Complete simulation records with conservation diagnostics

## Physical Applications

Relevant for idealized studies of quantum tunneling in:
- **Nuclear Physics**: α-decay (fm scale)
- **Surface Science**: STM imaging, field emission (Å scale)  
- **Chemical Dynamics**: Proton transfer reactions (nm scale)
- **Nanoelectronics**: Resonant tunneling diodes, Josephson junctions (μm scale)

## Citation

If you use this solver in your research, please cite:

```bibtex
@software{qt1d_solver_2025,
  author = {Kaban, Siti N. and Herho, Sandy H. S. and Prayogo, Sonny and Anwar, Iwan P.},
  title = {1D Quantum Tunneling Solver: Idealized Split-Operator Method},
  year = {2025},
  version = {0.0.4},
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
