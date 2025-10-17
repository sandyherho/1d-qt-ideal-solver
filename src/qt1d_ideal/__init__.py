"""1D Quantum Tunneling Solver - Idealized Version"""

__version__ = "0.0.9"
__author__ = "Sandy H. S. Herho, Siti N. Kaban, Iwan P. Anwar, Nurjanna J. Trilaksono, Rusmawan Suwarman"

from .core.solver import QuantumTunneling1D
from .core.initial_conditions import GaussianWavePacket
from .io.config_manager import ConfigManager
from .io.data_handler import DataHandler

__all__ = ["QuantumTunneling1D", "GaussianWavePacket", "ConfigManager", "DataHandler"]
