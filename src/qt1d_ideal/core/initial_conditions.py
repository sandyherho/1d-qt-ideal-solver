"""Gaussian wave packet initial conditions."""

import numpy as np


class GaussianWavePacket:
    """Gaussian wave packet: ψ(x,0) = N * exp(-(x-x0)²/(4σ²)) * exp(ik0*x)"""
    
    def __init__(self, x0: float = -5.0, k0: float = 5.0, sigma: float = 0.5):
        self.x0 = x0
        self.k0 = k0
        self.sigma = sigma
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        norm = (1.0 / (2.0 * np.pi * self.sigma**2))**0.25
        envelope = np.exp(-((x - self.x0)**2) / (4.0 * self.sigma**2))
        phase = np.exp(1j * self.k0 * x)
        return norm * envelope * phase
