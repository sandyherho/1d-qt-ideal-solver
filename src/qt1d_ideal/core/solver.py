"""
1D Quantum Tunneling Solver with Adaptive Time Stepping and Stochastic Noise
"""

import numpy as np
from scipy.fftpack import fft, ifft, fftfreq
from typing import Dict, Any, Optional
from tqdm import tqdm
import numba
from numba import jit, prange
import os


@jit(nopython=True, parallel=True, cache=True)
def apply_kinetic_operator(psi_k: np.ndarray, k: np.ndarray, 
                          dt: float, m: float) -> np.ndarray:
    """Apply kinetic energy operator in momentum space."""
    n = len(psi_k)
    psi_k_new = np.zeros(n, dtype=np.complex128)
    for i in prange(n):
        kinetic_energy = k[i]**2 / (2.0 * m)
        phase = np.exp(-1j * kinetic_energy * dt)
        psi_k_new[i] = psi_k[i] * phase
    return psi_k_new


@jit(nopython=True, parallel=True, cache=True)
def apply_potential_operator(psi: np.ndarray, V: np.ndarray, 
                             dt: float) -> np.ndarray:
    """Apply potential energy operator in position space."""
    n = len(psi)
    psi_new = np.zeros(n, dtype=np.complex128)
    for i in prange(n):
        phase = np.exp(-1j * V[i] * dt)
        psi_new[i] = psi[i] * phase
    return psi_new


@jit(nopython=True, cache=True)
def estimate_wavefunction_change(psi_curr: np.ndarray, psi_prev: np.ndarray,
                                 dx: float) -> float:
    """Estimate rate of wavefunction change for adaptive time stepping."""
    diff = psi_curr - psi_prev
    diff_norm = np.sqrt(np.sum(np.abs(diff)**2) * dx)
    curr_norm = np.sqrt(np.sum(np.abs(psi_curr)**2) * dx)
    return diff_norm / curr_norm if curr_norm > 1e-12 else 0.0


@jit(nopython=True, cache=True)
def compute_energy(psi: np.ndarray, psi_k: np.ndarray, k: np.ndarray, 
                   V: np.ndarray, dx: float, m: float) -> float:
    """
    Compute total energy with CORRECT FFT normalization.
    
    For scipy.fftpack FFT normalization:
    - Position space: ∫|ψ(x)|²dx = ∑|ψ_n|² * dx
    - Momentum space: ψ̃(k) = ψ_k[fft] * dx (physical Fourier transform)
    - Parseval: ∑|ψ_n|² * dx = ∑|ψ_k|² * (dx/N)
    - Kinetic: T = ∑(k²/2m)|ψ_k|² * (dx/N)
    
    FIXED: Changed dx/(2π) to dx/N for correct energy conservation.
    """
    N = len(psi_k)
    
    # Kinetic energy: T = ∑(k²/2m)|ψ_k|² * (dx/N)
    # This is the CORRECT normalization for scipy FFT
    kinetic = 0.0
    for i in range(N):
        kinetic += (k[i]**2 / (2.0 * m)) * np.abs(psi_k[i])**2
    kinetic = kinetic * dx / N  # FIXED: was dx/(2π), now dx/N
    
    # Potential energy: V = ∫V(x)|ψ(x)|²dx = ∑V_n|ψ_n|² * dx
    potential = 0.0
    for i in range(len(psi)):
        potential += V[i] * np.abs(psi[i])**2
    potential = potential * dx
    
    return kinetic + potential


class QuantumTunneling1D:
    """1D Quantum Tunneling Solver with Split-Operator Method."""
    
    def __init__(self, nx: int = 2048, x_min: float = -10.0, x_max: float = 10.0,
                 adaptive_dt: bool = True, verbose: bool = True,
                 logger: Optional[Any] = None, n_cores: Optional[int] = None):
        self.nx = nx
        self.x_min = x_min
        self.x_max = x_max
        self.dx = (x_max - x_min) / (nx - 1)
        self.adaptive_dt = adaptive_dt
        self.verbose = verbose
        self.logger = logger
        
        if n_cores is None:
            n_cores = os.cpu_count()
        numba.set_num_threads(n_cores)
        
        self.x = np.linspace(x_min, x_max, nx)
        self.k = 2.0 * np.pi * fftfreq(nx, d=self.dx)
        
        if verbose:
            print(f"  Grid: {nx} points, dx = {self.dx:.4f} nm")
            print(f"  Using {n_cores} CPU cores")
    
    def rectangular_barrier(self, height: float, width: float,
                           center: float = 0.0) -> np.ndarray:
        V = np.zeros(self.nx)
        mask = np.abs(self.x - center) < width / 2.0
        V[mask] = height
        return V
    
    def gaussian_barrier(self, height: float, width: float,
                        center: float = 0.0) -> np.ndarray:
        return height * np.exp(-((self.x - center)**2) / (2.0 * width**2))
    
    def double_barrier(self, height: float, width: float,
                      separation: float) -> np.ndarray:
        V = np.zeros(self.nx)
        left_center = -separation / 2.0 - width / 2.0
        left_mask = np.abs(self.x - left_center) < width / 2.0
        V[left_mask] = height
        right_center = separation / 2.0 + width / 2.0
        right_mask = np.abs(self.x - right_center) < width / 2.0
        V[right_mask] = height
        return V
    
    def triple_barrier(self, height: float, width: float,
                      separation: float) -> np.ndarray:
        V = np.zeros(self.nx)
        positions = [-separation, 0.0, separation]
        for center in positions:
            mask = np.abs(self.x - center) < width / 2.0
            V[mask] = height
        return V
    
    def solve(self, psi0: np.ndarray, V: np.ndarray, t_final: float = 5.0,
              dt_initial: Optional[float] = None, dt_min: float = 1e-3,
              dt_max: float = 1e-2, n_snapshots: int = 200,
              particle_mass: float = 1.0, show_progress: bool = True,
              tolerance: float = 1e-3, 
              noise_amplitude: float = 0.0, noise_correlation_time: float = 0.1,
              decoherence_rate: float = 0.0) -> Dict[str, Any]:
        """Solve quantum tunneling with adaptive time stepping and stochastic noise."""
        
        # Initialize
        psi = psi0.copy().astype(np.complex128)
        norm = np.sqrt(np.sum(np.abs(psi)**2) * self.dx)
        psi = psi / norm
        
        noise_potential = np.zeros(self.nx)
        noise_enabled = noise_amplitude > 0
        
        if dt_initial is None:
            k_max = np.max(np.abs(self.k))
            E_k_max = k_max**2 / (2.0 * particle_mass)
            V_max = np.max(np.abs(V))
            dt = 0.5 / (E_k_max + V_max + 1e-10)
            dt = np.clip(dt, dt_min, dt_max)
        else:
            dt = np.clip(dt_initial, dt_min, dt_max)
        
        if self.verbose:
            print(f"  Initial dt = {dt:.6f} fs")
            if noise_enabled:
                print(f"  Stochastic noise: amplitude = {noise_amplitude:.4f} eV, "
                      f"τ_corr = {noise_correlation_time:.3f} fs")
            if decoherence_rate > 0:
                print(f"  Decoherence: γ = {decoherence_rate:.4f} fs⁻¹ "
                      f"(T₂ ~ {1.0/decoherence_rate:.1f} fs)")
        
        t = 0.0
        t_save = np.linspace(0, t_final, n_snapshots)
        save_idx = 0
        
        psi_hist = np.zeros((n_snapshots, self.nx), dtype=np.complex128)
        prob_hist = np.zeros((n_snapshots, self.nx))
        dt_hist = []
        
        psi_hist[0] = psi
        prob_hist[0] = np.abs(psi)**2
        save_idx = 1
        
        psi_prev = psi.copy()
        
        # Initial energy
        psi_k = fft(psi)
        E_initial = compute_energy(psi, psi_k, self.k, V, self.dx, particle_mass)
        
        norm_violations = []
        energy_violations = []
        max_norm_error = 0.0
        max_energy_error = 0.0
        
        if show_progress:
            pbar = tqdm(
                total=t_final, 
                desc="  Evolving", 
                unit="fs",
                bar_format='{l_bar}{bar}| {n:.2f}/{total:.2f} fs [{elapsed}<{remaining}]'
            )
        
        n_steps = 0
        while t < t_final and save_idx < n_snapshots:
            # Stochastic noise
            if noise_enabled:
                decay = np.exp(-dt / noise_correlation_time)
                noise_potential = decay * noise_potential + \
                    noise_amplitude * np.sqrt((1 - decay**2)) * np.random.randn(self.nx)
                V_total = V + noise_potential
            else:
                V_total = V
            
            # Split-operator step
            psi = apply_potential_operator(psi, V_total, dt / 2.0)
            psi_k = fft(psi)
            psi_k = apply_kinetic_operator(psi_k, self.k, dt, particle_mass)
            psi = ifft(psi_k)
            psi = apply_potential_operator(psi, V_total, dt / 2.0)
            
            # Decoherence
            if decoherence_rate > 0:
                damping = np.exp(-decoherence_rate * dt)
                psi = psi * damping
            
            # Stability check
            if np.any(np.isnan(psi)) or np.any(np.isinf(psi)):
                error_msg = f"Numerical instability at t={t:.3f} fs"
                if self.logger:
                    self.logger.error(error_msg)
                if self.verbose:
                    print(f"\n  ERROR: {error_msg}")
                break
            
            # Norm check - FIXED: Don't flag violations when decoherence is enabled
            current_norm = np.sqrt(np.sum(np.abs(psi)**2) * self.dx)
            norm_error = abs(current_norm - 1.0)
            
            # Only flag as violation if no decoherence (decoherence causes expected norm decay)
            if norm_error > 0.01 and decoherence_rate == 0.0:
                norm_violations.append((t, norm_error))
                if norm_error > max_norm_error:
                    max_norm_error = norm_error
            
            # Track max error even with decoherence (for diagnostic purposes)
            if norm_error > max_norm_error:
                max_norm_error = norm_error
            
            # Energy check (only if no noise/decoherence)
            if not noise_enabled and decoherence_rate == 0.0 and n_steps % 100 == 0 and n_steps > 0:
                psi_k_check = fft(psi)
                E_current = compute_energy(psi, psi_k_check, self.k, V, 
                                          self.dx, particle_mass)
                energy_error = abs((E_current - E_initial) / E_initial)
                if energy_error > 0.01:  # 1% threshold
                    energy_violations.append((t, energy_error))
                    if energy_error > max_energy_error:
                        max_energy_error = energy_error
            
            # Adaptive time stepping
            if self.adaptive_dt and n_steps > 0:
                change_rate = estimate_wavefunction_change(psi, psi_prev, self.dx)
                
                if change_rate > tolerance * 5.0:
                    dt = dt * 0.95
                elif change_rate < tolerance * 0.05:
                    dt = dt * 1.5
                
                dt = np.clip(dt, dt_min, dt_max)
                if t + dt > t_final:
                    dt = t_final - t
            
            psi_prev = psi.copy()
            t += dt
            n_steps += 1
            dt_hist.append(dt)
            
            if show_progress:
                pbar.update(dt)
            
            if save_idx < n_snapshots and t >= t_save[save_idx]:
                psi_hist[save_idx] = psi
                prob_hist[save_idx] = np.abs(psi)**2
                save_idx += 1
        
        if show_progress:
            pbar.close()
        
        # Log violations (only for truly unexpected violations)
        if norm_violations:
            warning = f"Norm violated {len(norm_violations)} times (max: {max_norm_error:.4f})"
            if self.logger:
                self.logger.warning(warning)
            if self.verbose:
                print(f"  WARNING: {warning}")
        
        if energy_violations:
            warning = f"Energy violated {len(energy_violations)} times (max: {max_energy_error:.4%})"
            if self.logger:
                self.logger.warning(warning)
            if self.verbose:
                print(f"  WARNING: {warning}")
        
        if self.verbose:
            print(f"  Completed {n_steps} time steps")
        
        # Transmission/Reflection
        barrier_mask = V > 0.1 * np.max(V)
        if np.any(barrier_mask):
            barrier_indices = np.where(barrier_mask)[0]
            barrier_start = barrier_indices[0]
            barrier_end = barrier_indices[-1]
        else:
            barrier_start = self.nx // 2
            barrier_end = self.nx // 2
        
        prob_final = prob_hist[save_idx - 1]
        trans_prob = np.sum(prob_final[barrier_end:]) * self.dx
        refl_prob = np.sum(prob_final[:barrier_start]) * self.dx
        total_prob = trans_prob + refl_prob
        
        if total_prob > 0:
            transmission = trans_prob / total_prob
            reflection = refl_prob / total_prob
        else:
            transmission = 0.0
            reflection = 0.0
        
        tr_sum = transmission + reflection
        if abs(tr_sum - 1.0) > 0.05:
            warning = f"T + R = {tr_sum:.4f} ≠ 1"
            if self.logger:
                self.logger.warning(warning)
            if self.verbose:
                print(f"  WARNING: {warning}")
        
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
                'dx': self.dx,
                'noise_amplitude': noise_amplitude,
                'noise_correlation_time': noise_correlation_time,
                'decoherence_rate': decoherence_rate,
                'max_norm_error': max_norm_error,
                'max_energy_error': max_energy_error,
                'n_norm_violations': len(norm_violations),
                'n_energy_violations': len(energy_violations)
            }
        }
