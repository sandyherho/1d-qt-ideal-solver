# `1d-qt-ideal-solver`: Idealized 1D Quantum Tunneling Solver

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Numba](https://img.shields.io/badge/accelerated-numba-orange.svg)](https://numba.pydata.org/)

High-performance educational solver for 1D quantum tunneling phenomena using the split-operator Fourier method with adaptive time stepping.

## Physics

Solves the time-dependent Schrödinger equation in natural units ($\hbar = m_e = 1$):

$$i\frac{\partial \psi}{\partial t} = \hat{H}\psi = \left[-\frac{1}{2}\nabla^2 + V(x,t)\right]\psi$$

The wavefunction evolves via the split-operator method:

$$\psi(x, t+\delta t) = e^{-iV\delta t/2} \cdot \mathcal{F}^{-1}\left[e^{-ik^2\delta t/2}\mathcal{F}[\psi]\right] \cdot e^{-iV\delta t/2}$$

**Key Observables:**
- Transmission coefficient: $T = \int_{x>x_b} |\psi(x,t_f)|^2 dx$
- Reflection coefficient: $R = \int_{x<x_a} |\psi(x,t_f)|^2 dx$
- Probability conservation: $T + R \approx 1$

## Features

- **Adaptive Time Stepping**: Automatically adjusts $\delta t$ based on wavefunction dynamics
- **Numba JIT Compilation**: 10-100× speedup via parallel CPU execution
- **Stochastic Environments**: Ornstein-Uhlenbeck noise ($\tau_{\text{corr}}$) and decoherence ($\gamma$)
- **Conservation Monitoring**: Real-time validation of $\|\psi\|^2 = 1$ and energy conservation
- **Professional Visualization**: Parallel-rendered GIF animations with publication-quality aesthetics
- **NetCDF4 Output**: Self-describing, compressed data format for reproducible research

## Installation

```bash
pip install -e .
```

**Dependencies**: NumPy, SciPy, matplotlib, netCDF4, tqdm, numba

## Quick Start

```bash
# Run individual test case
qt1d-simulate case1                    # Rectangular barrier
qt1d-simulate case2 --cores 8          # Double barrier (8 cores)

# Run all 4 cases sequentially
qt1d-simulate --all

# Custom configuration
qt1d-simulate --config myconfig.txt
```

## Test Cases

| Case | System | Physics | Expected $T$ |
|------|--------|---------|-------------|
| **1** | Rectangular barrier | Classic tunneling: $T \propto e^{-2\kappa w}$ | 15-25% |
| **2** | Double barrier | Resonant tunneling: $T(E_n) \approx 1$ | 30-60% |
| **3** | Triple barrier | Coupled quantum wells, beating | 15-40% |
| **4** | Gaussian barrier | Smooth WKB regime | 25-35% |

**Complexity Progression**: Case 1 (fundamental) → Case 4 (multi-scale dynamics)

## Python API

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

## Output

**Generated Files** (in `outputs/`):
- `case1_rectangular.nc` — NetCDF4 data (wavefunction evolution, parameters, metadata)
- `case1_rectangular.gif` — Professional animation with statistics overlay

**Log Files** (in `logs/`):
- `case1_rectangular.log` — Timestamped simulation record with diagnostics

## Configuration

Example (`configs/case1_rectangular.txt`):

```ini
nx = 2048                  # Grid points (FFT-optimized)
x_min = -10.0              # [nm]
x_max = 10.0               # [nm]
t_final = 5.0              # [fs]

barrier_type = rectangular
barrier_height = 2.0       # [eV]
barrier_width = 2.0        # [nm]

x0 = -5.0                  # Initial position [nm]
k0 = 5.0                   # Wave vector [nm⁻¹] → E_k ≈ 1.52 eV
sigma = 0.5                # Gaussian width [nm]

noise_amplitude = 0.0      # [eV] (disabled for ideal case)
decoherence_rate = 0.0     # [fs⁻¹]
```

## Performance

- **Typical Runtime**: 10-30 seconds per case (2048 grid points, 200 frames)
- **Parallel Rendering**: ~10× faster animation generation (multiprocessing)
- **Conservation Accuracy**: Norm error < 0.1%, energy error < 1% (fixed FFT normalization)

## Physical Context

Demonstrates universal quantum tunneling across scales:
- **Nuclear**: α-decay (fm scale)
- **Atomic**: STM imaging, field emission (Å scale)  
- **Molecular**: Proton transfer, chemical reactions (nm scale)
- **Mesoscopic**: Josephson junctions, RTDs (μm scale)

## Authors

- **Siti N. Kaban**
- **Sandy H. S. Herho** (sandy.herho@email.ucr.edu)
- **Sonny Prayogo**
- **Iwan P. Anwar**

## License

MIT License — Free for educational and research use.

## Citation

If you use this solver in your study, please cite:

```bibtex
@software{qt1d_solver_2025,
  author = {Kaban, Siti N. and Herho, Sandy H. S. and Prayogo, Sonny and Anwar, Iwan P.},
  title = {1D Quantum Tunneling Solver with Stochastic Noise},
  year = {2025},
  version = {0.0.1},
  license = {MIT}
}
```

---

**Note**: This is an educational tool. For production quantum simulations, consider specialized packages (QuTiP, GPAW, VASP).
