"""
Initial Conditions for Idealized Quantum Tunneling Simulations

Provides Gaussian wave packet initial conditions commonly used
in idealized quantum mechanics calculations.
"""

import numpy as np


class GaussianWavePacket:
    """
    Gaussian wave packet for idealized quantum simulations.
    
    Creates a localized wave packet with definite position and momentum:
        ψ(x,0) = N * exp(-(x-x0)²/(4σ²)) * exp(ik0*x)
    
    where N is normalization constant.
    
    This represents an idealized free particle before interaction
    with the potential barrier.
    
    Parameters
    ----------
    x0 : float
        Initial center position (nm)
    k0 : float  
        Wave number: k = 2π/λ = p/ℏ (1/nm)
        Related to particle momentum and energy: E = ℏ²k²/(2m)
    sigma : float
        Spatial width (nm)
        Determines position uncertainty: Δx ≈ σ
        Momentum uncertainty: Δp ≈ ℏ/σ (Heisenberg uncertainty)
    
    Notes
    -----
    This is an IDEALIZED representation. Real quantum systems may have
    more complex initial states.
    """
    
    def __init__(self, x0: float = -5.0, k0: float = 5.0, sigma: float = 0.5):
        self.x0 = x0
        self.k0 = k0
        self.sigma = sigma
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Generate Gaussian wave packet on spatial grid.
        
        Parameters
        ----------
        x : ndarray
            Spatial grid points (nm)
        
        Returns
        -------
        psi : ndarray (complex128)
            Normalized wavefunction
        """
        # Normalization constant (ensures ∫|ψ|²dx = 1)
        norm = (1.0 / (2.0 * np.pi * self.sigma**2))**0.25
        
        # Gaussian envelope (position localization)
        envelope = np.exp(-((x - self.x0)**2) / (4.0 * self.sigma**2))
        
        # Plane wave (momentum/energy)
        phase = np.exp(1j * self.k0 * x)
        
        return norm * envelope * phase
