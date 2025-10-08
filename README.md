# 1D Quantum Tunneling Solver (Simplified)

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Numba](https://img.shields.io/badge/accelerated-numba-orange.svg)](https://numba.pydata.org/)

Educational tool for simulating 1D quantum tunneling with **4 classic test cases** of increasing complexity.

## Features

- **Adaptive time stepping** for optimal accuracy
- **Numba JIT compilation** for 10-100x speedup
- **Multi-core parallelization**
- **4 classic scenarios** with increasing complexity
- **NetCDF output** with essential variables
- **GIF animations** of wavefunction evolution

## Quick Start

```bash
# Install
pip install -e .

# Run simulations
qt1d-simulate case1    # Rectangular barrier
qt1d-simulate case2    # Double barrier
qt1d-simulate case3    # Triple barrier
qt1d-simulate case4    # Gaussian barrier

# Run all cases
qt1d-simulate --all

# Use multiple cores
qt1d-simulate case1 --cores 8
```

## Test Cases (Increasing Complexity)

1. **Case 1: Rectangular Barrier** - Classic single barrier tunneling
2. **Case 2: Double Barrier** - Resonant tunneling structure
3. **Case 3: Triple Barrier** - Multiple resonances and quantum beats
4. **Case 4: Gaussian Barrier** - Smooth potential profile

## Python API

```python
from qt1d_ideal import QuantumTunneling1D, GaussianWavePacket

solver = QuantumTunneling1D(nx=2048, x_min=-10, x_max=10)
psi0 = GaussianWavePacket(x0=-5.0, k0=5.0, sigma=0.5)(solver.x)
V = solver.rectangular_barrier(height=2.0, width=2.0)

result = solver.solve(psi0=psi0, V=V, t_final=5.0)
print(f"Transmission: {result['transmission_coefficient']:.2%}")
```

## Authors

- Siti N. Kaban
- Sandy H. S. Herho (sandy.herho@email.ucr.edu)
- Sonny Prayogo
- Iwan P. Anwar

## License

MIT License
