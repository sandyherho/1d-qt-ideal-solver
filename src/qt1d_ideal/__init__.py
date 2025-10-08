"""
Idealized 1D Quantum Tunneling Solver
======================================

Educational and research tool for simulating IDEALIZED 1D quantum tunneling.

WARNING: These are simplified models for educational purposes, capturing
essential physics but not representing complete quantum mechanical calculations.

The solver implements the time-dependent Schrödinger equation:
    iℏ ∂ψ/∂t = -ℏ²/2m ∂²ψ/∂x² + V(x)ψ

using split-operator method with adaptive time stepping.
"""

__version__ = "0.1.0"
__author__ = "Siti N. Kaban, Sandy H. S. Herho, Sonny Prayogo"
__email__ = "sandy.herho@email.ucr.edu"
__license__ = "MIT"

from .core.solver import QuantumTunneling1D
from .core.initial_conditions import GaussianWavePacket
from .io.config_manager import ConfigManager
from .io.data_handler import DataHandler

__all__ = [
    "QuantumTunneling1D",
    "GaussianWavePacket",
    "ConfigManager",
    "DataHandler",
]
