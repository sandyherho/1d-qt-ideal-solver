"""
Idealized 1D Quantum Tunneling Solver with Adaptive Time Stepping

Implements split-operator method for the time-dependent Schrödinger equation
with automatic adaptive time step control for optimal accuracy and efficiency.

IMPORTANT: This solver uses IDEALIZED models for educational purposes.
Real quantum systems involve additional complexities not captured here.
"""

import numpy as np
from scipy.fftpack import fft, ifft, fftfreq
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm
import numba
from numba import jit, prange
import os


@jit(nopython=True, parallel=True, cache=True)
def apply_kinetic_operator(psi_k: np.ndarray, k: np.ndarray, 
                          dt: float, m: float) -> np.ndarray:
    """
    Apply kinetic energy operator in momentum space (Numba-optimized).
    
    Kinetic operator is diagonal in momentum space:
        T̂ = (ℏk)²/(2m) = k²/(2m) in atomic units
    
    Time evolution: ψ(k,t+dt) = exp(-i*T̂*dt) * ψ(k,t)
    
    Parameters
    ----------
    psi_k : complex array
        Wavefunction in momentum space
    k : float array
        Wave vector grid (1/nm)
    dt : float
        Time step (fs)
    m : float
        Particle mass (electron masses)
    
    Returns
    -------
    psi_k_evolved : complex array
        Time-evolved wavefunction in momentum space
    """
    n = len(psi_k)
    psi_k_new = np.zeros(n, dtype=np.complex128)
    
    for i in prange(n):
        # Kinetic energy: E_k = k²/(2m)
        kinetic_energy = k[i]**2 / (2.0 * m)
        # Apply time evolution: exp(-i*E*dt)
        phase = np.exp(-1j * kinetic_energy * dt)
        psi_k_new[i] = psi_k[i] * phase
    
    return psi_k_new


@jit(nopython=True, parallel=True, cache=True)
def apply_potential_operator(psi: np.ndarray, V: np.ndarray, 
                             dt: float) -> np.ndarray:
    """
    Apply potential energy operator in position space (Numba-optimized).
    
    Potential operator is diagonal in position space:
        V̂ = V(x)
    
    Time evolution: ψ(x,t+dt) = exp(-i*V̂*dt) * ψ(x,t)
    
    Parameters
    ----------
    psi : complex array
        Wavefunction in position space
    V : float array
        Potential energy at each position (eV)
    dt : float
        Time step (fs)
    
    Returns
    -------
    psi_evolved : complex array
        Time-evolved wavefunction in position space
    """
    n = len(psi)
    psi_new = np.zeros(n, dtype=np.complex128)
    
    for i in prange(n):
        # Apply time evolution: exp(-i*V*dt)
        phase = np.exp(-1j * V[i] * dt)
        psi_new[i] = psi[i] * phase
    
    return psi_new


@jit(nopython=True, cache=True)
def estimate_wavefunction_change(psi_curr: np.ndarray, 
                                 psi_prev: np.ndarray,
                                 dx: float) -> float:
    """
    Estimate rate of wavefunction change for adaptive time stepping.
    
    Computes normalized L2 difference between consecutive steps:
        ε = ||ψ_n - ψ_{n-1}|| / ||ψ_n||
    
    Used to adaptively adjust time step for optimal accuracy.
    
    Parameters
    ----------
    psi_curr : complex array
        Current wavefunction
    psi_prev : complex array
        Previous wavefunction
    dx : float
        Spatial grid spacing
    
    Returns
    -------
    change_rate : float
        Normalized wavefunction change rate
    """
    # Compute L2 norms
    diff = psi_curr - psi_prev
    diff_norm = np.sqrt(np.sum(np.abs(diff)**2) * dx)
    curr_norm = np.sqrt(np.sum(np.abs(psi_curr)**2) * dx)
    
    if curr_norm > 1e-12:
        return diff_norm / curr_norm
    else:
        return 0.0


class QuantumTunneling1D:
    """
    Idealized 1D quantum tunneling solver with adaptive time stepping.
    
    Uses split-operator (Trotter-Suzuki) method:
        exp(-i*Ĥ*dt) ≈ exp(-i*V̂*dt/2) * exp(-i*T̂*dt) * exp(-i*V̂*dt/2)
    
    Adaptive time stepping automatically adjusts dt based on:
    - Wavefunction evolution rate
    - Stability criteria
    - User-specified bounds
    
    This provides optimal balance between accuracy and efficiency.
    
    Attributes
    ----------
    nx : int
        Number of spatial grid points
    x : ndarray
        Position grid (nm)
    k : ndarray
        Momentum grid (1/nm)
    adaptive_dt : bool
        Enable adaptive time stepping
    """
    
    def __init__(
        self,
        nx: int = 2048,
        x_min: float = -10.0,
        x_max: float = 10.0,
        adaptive_dt: bool = True,
        verbose: bool = True,
        logger: Optional[Any] = None,
        n_cores: Optional[int] = None
    ):
        """
        Initialize idealized 1D quantum tunneling solver.
        
        Parameters
        ----------
        nx : int
            Grid points (power of 2 recommended for FFT)
        x_min, x_max : float
            Domain boundaries (nm)
        adaptive_dt : bool
            Enable adaptive time stepping
        verbose : bool
            Print status messages
        logger : optional
            Logger instance
        n_cores : int, optional
            CPU cores for parallelization (default: all)
        """
        self.nx = nx
        self.x_min = x_min
        self.x_max = x_max
        self.dx = (x_max - x_min) / (nx - 1)
        self.adaptive_dt = adaptive_dt
        self.verbose = verbose
        self.logger = logger
        
        # Configure Numba parallelization
        if n_cores is None:
            n_cores = os.cpu_count()
        numba.set_num_threads(n_cores)
        
        # Create spatial grid
        self.x = np.linspace(x_min, x_max, nx)
        
        # Create momentum grid (for FFT-based kinetic operator)
        self.k = 2.0 * np.pi * fftfreq(nx, d=self.dx)
        
        if verbose:
            print(f"  Solver initialized (IDEALIZED model)")
            print(f"  Grid: {nx} points, domain: [{x_min:.2f}, {x_max:.2f}] nm")
            print(f"  Spacing: dx = {self.dx:.4f} nm")
            print(f"  Adaptive time stepping: {adaptive_dt}")
            print(f"  Using {n_cores} CPU cores")
    
    def rectangular_barrier(self, height: float, width: float,
                           center: float = 0.0) -> np.ndarray:
        """
        Create idealized rectangular potential barrier.
        
        V(x) = { height, if |x - center| < width/2
               { 0,      otherwise
        """
        V = np.zeros(self.nx)
        mask = np.abs(self.x - center) < width / 2.0
        V[mask] = height
        return V
    
    def gaussian_barrier(self, height: float, width: float,
                        center: float = 0.0) -> np.ndarray:
        """
        Create idealized smooth Gaussian barrier.
        
        V(x) = height * exp(-(x-center)²/(2*width²))
        """
        return height * np.exp(-((self.x - center)**2) / (2.0 * width**2))
    
    def step_potential(self, step_height: float, 
                      step_position: float = 0.0) -> np.ndarray:
        """
        Create idealized step potential (half-infinite barrier).
        
        V(x) = { 0,           if x < step_position
               { step_height, if x >= step_position
        """
        V = np.zeros(self.nx)
        V[self.x >= step_position] = step_height
        return V
    
    def double_barrier(self, height: float, width: float,
                      separation: float) -> np.ndarray:
        """
        Create idealized double barrier (resonant tunneling structure).
        
        Two rectangular barriers separated by a quantum well region.
        """
        V = np.zeros(self.nx)
        # Left barrier
        left_center = -separation / 2.0 - width / 2.0
        left_mask = np.abs(self.x - left_center) < width / 2.0
        V[left_mask] = height
        # Right barrier
        right_center = separation / 2.0 + width / 2.0
        right_mask = np.abs(self.x - right_center) < width / 2.0
        V[right_mask] = height
        return V
    
    def triple_barrier(self, height: float, width: float,
                      separation: float) -> np.ndarray:
        """
        Create idealized triple barrier system.
        
        Three barriers with two quantum wells - exhibits complex resonances.
        """
        V = np.zeros(self.nx)
        positions = [-separation, 0.0, separation]
        for center in positions:
            mask = np.abs(self.x - center) < width / 2.0
            V[mask] = height
        return V
    
    def periodic_potential(self, height: float, n_periods: int,
                          period_length: float) -> np.ndarray:
        """
        Create idealized periodic potential (band structure analog).
        
        Represents simplified solid-state crystal potential.
        
        V(x) = height * cos²(2π*x/period_length)
        """
        period = period_length
        V = height * np.cos(2.0 * np.pi * self.x / period)**2
        return V
    
    def solve(
        self,
        psi0: np.ndarray,
        V: np.ndarray,
        t_final: float = 5.0,
        dt_initial: Optional[float] = None,
        dt_min: float = 1e-4,
        dt_max: float = 1e-2,
        n_snapshots: int = 200,
        particle_mass: float = 1.0,
        show_progress: bool = True,
        tolerance: float = 1e-3
    ) -> Dict[str, Any]:
        """
        Solve idealized 1D quantum tunneling with adaptive time stepping.
        
        Uses split-operator method with automatic dt adjustment based on
        wavefunction evolution rate.
        
        Parameters
        ----------
        psi0 : complex array
            Initial wavefunction
        V : float array
            Potential energy (eV)
        t_final : float
            Final time (fs)
        dt_initial : float, optional
            Initial time step (auto if None)
        dt_min, dt_max : float
            Time step bounds for adaptive control
        n_snapshots : int
            Number of snapshots to save
        particle_mass : float
            Particle mass (electron masses)
        show_progress : bool
            Show progress bar
        tolerance : float
            Target accuracy for adaptive stepping
        
        Returns
        -------
        result : dict
            Simulation results including wavefunction, probability,
            transmission/reflection coefficients
        """
        # Normalize initial wavefunction
        psi = psi0.copy().astype(np.complex128)
        norm = np.sqrt(np.sum(np.abs(psi)**2) * self.dx)
        psi = psi / norm
        
        # Initialize time step
        if dt_initial is None:
            # Conservative initial estimate based on grid and potential
            k_max = np.max(np.abs(self.k))
            E_k_max = k_max**2 / (2.0 * particle_mass)
            V_max = np.max(np.abs(V))
            dt = 0.5 / (E_k_max + V_max + 1e-10)
            dt = np.clip(dt, dt_min, dt_max)
        else:
            dt = np.clip(dt_initial, dt_min, dt_max)
        
        if self.verbose:
            print(f"  Initial time step: dt = {dt:.6f} fs")
            if self.adaptive_dt:
                print(f"  Adaptive stepping: dt ∈ [{dt_min:.6f}, {dt_max:.6f}] fs")
        
        # Time evolution parameters
        t = 0.0
        t_save = np.linspace(0, t_final, n_snapshots)
        save_idx = 0
        
        # Storage (parsimonious - only essentials)
        psi_hist = np.zeros((n_snapshots, self.nx), dtype=np.complex128)
        prob_hist = np.zeros((n_snapshots, self.nx))
        dt_hist = []  # Track adaptive time steps
        
        # Save initial state
        psi_hist[0] = psi
        prob_hist[0] = np.abs(psi)**2
        save_idx = 1
        
        # For adaptive time stepping
        psi_prev = psi.copy()
        
        # Prepare progress bar
        if show_progress:
            pbar = tqdm(total=t_final, desc="Time integration", unit="fs")
        
        # Main time evolution loop
        n_steps = 0
        while t < t_final and save_idx < n_snapshots:
            # === Split-Operator Time Step ===
            
            # Step 1: Apply V̂/2
            psi = apply_potential_operator(psi, V, dt / 2.0)
            
            # Step 2: FFT to momentum space
            psi_k = fft(psi)
            
            # Step 3: Apply T̂ in momentum space
            psi_k = apply_kinetic_operator(psi_k, self.k, dt, particle_mass)
            
            # Step 4: Inverse FFT back to position space
            psi = ifft(psi_k)
            
            # Step 5: Apply V̂/2
            psi = apply_potential_operator(psi, V, dt / 2.0)
            
            # === Adaptive Time Step Control ===
            if self.adaptive_dt and n_steps > 0:
                # Estimate wavefunction change rate
                change_rate = estimate_wavefunction_change(psi, psi_prev, self.dx)
                
                # Adjust time step based on change rate
                # If change is large, decrease dt; if small, increase dt
                if change_rate > tolerance:
                    dt = dt * 0.9  # Decrease
                elif change_rate < tolerance * 0.5:
                    dt = dt * 1.1  # Increase
                
                # Enforce bounds
                dt = np.clip(dt, dt_min, dt_max)
                
                # Also ensure we don't overshoot final time
                if t + dt > t_final:
                    dt = t_final - t
            
            # Store previous for next adaptive step
            psi_prev = psi.copy()
            
            # Check numerical stability
            if np.any(np.isnan(psi)) or np.any(np.isinf(psi)):
                if self.verbose:
                    print(f"\nWARNING: Numerical instability at t={t:.3f} fs")
                break
            
            # Update time
            t += dt
            n_steps += 1
            dt_hist.append(dt)
            
            # Update progress
            if show_progress:
                pbar.update(dt)
            
            # Save snapshot if we've reached next save time
            if save_idx < n_snapshots and t >= t_save[save_idx]:
                psi_hist[save_idx] = psi
                prob_hist[save_idx] = np.abs(psi)**2
                save_idx += 1
        
        if show_progress:
            pbar.close()
        
        if self.verbose:
            print(f"  Completed {n_steps} time steps")
            if self.adaptive_dt:
                print(f"  Time step range: [{np.min(dt_hist):.6f}, {np.max(dt_hist):.6f}] fs")
                print(f"  Mean time step: {np.mean(dt_hist):.6f} fs")
        
        # === Calculate transmission and reflection coefficients ===
        # Find barrier region
        barrier_mask = V > 0.1 * np.max(V)
        if np.any(barrier_mask):
            barrier_indices = np.where(barrier_mask)[0]
            barrier_start = barrier_indices[0]
            barrier_end = barrier_indices[-1]
        else:
            barrier_start = self.nx // 2
            barrier_end = self.nx // 2
        
        # Final probability distribution
        prob_final = prob_hist[save_idx - 1]
        
        # Transmission: probability past barrier
        trans_prob = np.sum(prob_final[barrier_end:]) * self.dx
        
        # Reflection: probability before barrier
        refl_prob = np.sum(prob_final[:barrier_start]) * self.dx
        
        # Normalize (should sum to ~1)
        total_prob = trans_prob + refl_prob
        if total_prob > 0:
            transmission = trans_prob / total_prob
            reflection = refl_prob / total_prob
        else:
            transmission = 0.0
            reflection = 0.0
        
        return {
            'x': self.x,
            't': t_save[:save_idx],
            'psi': psi_hist[:save_idx],
            'probability': prob_hist[:save_idx],
            'potential': V,
            'transmission_coefficient': transmission,
            'reflection_coefficient': reflection,
            'params': {
                'particle_mass': particle_mass,
                'dt_initial': dt_hist[0] if dt_hist else dt,
                'dt_final': dt_hist[-1] if dt_hist else dt,
                'dt_mean': np.mean(dt_hist) if dt_hist else dt,
                'n_steps': n_steps,
                'adaptive': self.adaptive_dt,
                'nx': self.nx,
                'dx': self.dx
            }
        }
