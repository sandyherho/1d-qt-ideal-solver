"""1D Quantum Tunneling Solver - Idealized Version"""

__version__ = "0.0.6"
__author__ = "Siti N. Kaban, Sandy H. S. Herho, Sonny Prayogo, Iwan P. Anwar"

from .core.solver import QuantumTunneling1D
from .core.initial_conditions import GaussianWavePacket
from .io.config_manager import ConfigManager
from .io.data_handler import DataHandler

__all__ = ["QuantumTunneling1D", "GaussianWavePacket", "ConfigManager", "DataHandler"]
