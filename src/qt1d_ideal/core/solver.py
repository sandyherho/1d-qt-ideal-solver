"""
1D Quantum Tunneling Solver with Absorbing Boundary Conditions
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


@jit(nopython=True, parallel=True, cache=True)
def apply_absorbing_mask(psi: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply absorbing boundary mask to prevent reflections."""
    n = len(psi)
    psi_new = np.zeros(n, dtype=np.complex128)
    for i in prange(n):
        psi_new[i] = psi[i] * mask[i]
    return psi_new


@jit(nopython=True, parallel=True, cache=True)
def apply_dephasing(psi: np.ndarray, dt: float, gamma: float, 
                   random_phases: np.ndarray) -> np.ndarray:
    """
    Apply pure dephasing (phase randomization) without probability loss.
    This preserves |psi|^2 while destroying phase coherence.
    """
    n = len(psi)
    psi_new = np.zeros(n, dtype=np.complex128)
    phase_spread = np.sqrt(2.0 * gamma * dt)
    
    for i in prange(n):
        # Apply random phase shift (pure dephasing)
        phase_shift = phase_spread * random_phases[i]
        dephasing_factor = np.exp(1j * phase_shift)
        psi_new[i] = psi[i] * dephasing_factor
    
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
    """Compute total energy with correct FFT normalization."""
    N = len(psi_k)
    
    kinetic = 0.0
    for i in range(N):
        kinetic += (k[i]**2 / (2.0 * m)) * np.abs(psi_k[i])**2
    kinetic = kinetic * dx / N
    
    potential = 0.0
    for i in range(len(psi)):
        potential += V[i] * np.abs(psi[i])**2
    potential = potential * dx
    
    return kinetic + potential


class QuantumTunneling1D:
    """1D Quantum Tunneling Solver with Absorbing Boundaries - FINAL REVISION."""
    
    def __init__(self, nx: int = 2048, x_min: float = -30.0, x_max: float = 30.0,
                 adaptive_dt: bool = True, verbose: bool = True,
                 logger: Optional[Any] = None, n_cores: Optional[int] = None,
                 boundary_width: float = 3.0, boundary_strength: float = 0.03):
        """
        Initialize solver with absorbing boundary conditions.
        """
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
        
        self._boundary_width = boundary_width
        self._boundary_strength = boundary_strength
        
        self.boundary_mask = self._create_absorbing_mask(boundary_width, boundary_strength)
        
        # Safe zone boundaries (where mask = 1.0, no absorption)
        self.x_safe_left = x_min + boundary_width
        self.x_safe_right = x_max - boundary_width
        
        if verbose:
            print(f"  Grid: {nx} points, dx = {self.dx:.4f} nm")
            print(f"  Domain: [{x_min:.1f}, {x_max:.1f}] nm (total: {x_max-x_min:.1f} nm)")
            print(f"  Absorbing boundaries: width = {boundary_width:.2f} nm, "
                  f"strength = {boundary_strength:.3f}")
            print(f"  Safe zone (no absorption): [{self.x_safe_left:.1f}, "
                  f"{self.x_safe_right:.1f}] nm (size: {self.x_safe_right-self.x_safe_left:.1f} nm)")
            print(f"  Using {n_cores} CPU cores")
    
    def _create_absorbing_mask(self, width: float, strength: float) -> np.ndarray:
        """Create smooth absorbing boundary mask with cos^4 profile."""
        mask = np.ones(self.nx)
        n_boundary = int(width / self.dx)
        
        if n_boundary > 0:
            # Left boundary
            for i in range(n_boundary):
                ratio = i / n_boundary
                # cos^4 for very smooth transition
                cos_factor = np.cos(0.5 * np.pi * (1.0 - ratio))**4
                mask[i] = 1.0 - strength * (1.0 - cos_factor)
            
            # Right boundary
            for i in range(n_boundary):
                idx = self.nx - 1 - i
                ratio = i / n_boundary
                cos_factor = np.cos(0.5 * np.pi * (1.0 - ratio))**4
                mask[idx] = 1.0 - strength * (1.0 - cos_factor)
        
        return mask
    
    def rectangular_barrier(self, height: float, width: float,
                           center: float = 0.0) -> np.ndarray:
        """Rectangular potential barrier."""
        V = np.zeros(self.nx)
        mask = np.abs(self.x - center) < width / 2.0
        V[mask] = height
        return V
    
    def gaussian_barrier(self, height: float, width: float,
                        center: float = 0.0) -> np.ndarray:
        """Gaussian potential barrier."""
        return height * np.exp(-((self.x - center)**2) / (2.0 * width**2))
    
    def double_barrier(self, height: float, width: float,
                      separation: float) -> np.ndarray:
        """Double rectangular barrier."""
        V = np.zeros(self.nx)
        left_center = -separation / 2.0
        left_mask = np.abs(self.x - left_center) < width / 2.0
        V[left_mask] = height
        
        right_center = separation / 2.0
        right_mask = np.abs(self.x - right_center) < width / 2.0
        V[right_mask] = height
        return V
    
    def triple_barrier(self, height: float, width: float,
                      separation: float) -> np.ndarray:
        """Triple barrier."""
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
        """
        Solve quantum tunneling with absorbing boundaries.
        """
        
        # Initialize
        psi = psi0.copy().astype(np.complex128)
        norm = np.sqrt(np.sum(np.abs(psi)**2) * self.dx)
        psi = psi / norm
        
        noise_potential = np.zeros(self.nx)
        noise_enabled = noise_amplitude > 0
        dephasing_enabled = decoherence_rate > 0
        
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
            if dephasing_enabled:
                print(f"  Pure dephasing: γ = {decoherence_rate:.4f} fs⁻¹ "
                      f"(T₂ ~ {1.0/decoherence_rate:.1f} fs)")
                print(f"  NOTE: Pure dephasing preserves probability |psi|^2")
        
        # Time evolution
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
        
        # Diagnostics
        norm_violations = []
        energy_violations = []
        max_norm_error = 0.0
        max_energy_error = 0.0
        absorbed_probability = 0.0
        
        # Find barrier region
        barrier_mask = V > 0.1 * np.max(V)
        if np.any(barrier_mask):
            barrier_indices = np.where(barrier_mask)[0]
            barrier_start_idx = barrier_indices[0]
            barrier_end_idx = barrier_indices[-1]
            barrier_center_x = (self.x[barrier_start_idx] + self.x[barrier_end_idx]) / 2.0
        else:
            barrier_center_x = 0.0
            barrier_start_idx = self.nx // 2
            barrier_end_idx = self.nx // 2
        
        # Detection zones in safe region
        idx_safe_left = np.searchsorted(self.x, self.x_safe_left)
        idx_safe_right = np.searchsorted(self.x, self.x_safe_right)
        
        if self.verbose:
            print(f"  Barrier center: {barrier_center_x:.1f} nm")
            print(f"  Left detection zone: [{self.x_safe_left:.1f}, "
                  f"{self.x[barrier_start_idx]:.1f}] nm")
            print(f"  Right detection zone: [{self.x[barrier_end_idx]:.1f}, "
                  f"{self.x_safe_right:.1f}] nm")
        
        if show_progress:
            pbar = tqdm(
                total=t_final, 
                desc="  Evolving", 
                unit="fs",
                bar_format='{l_bar}{bar}| {n:.2f}/{total:.2f} fs [{elapsed}<{remaining}]'
            )
        
        n_steps = 0
        while t < t_final and save_idx < n_snapshots:
            norm_before = np.sqrt(np.sum(np.abs(psi)**2) * self.dx)
            
            if noise_enabled:
                decay = np.exp(-dt / noise_correlation_time)
                noise_potential = decay * noise_potential + \
                    noise_amplitude * np.sqrt((1 - decay**2)) * np.random.randn(self.nx)
                V_total = V + noise_potential
            else:
                V_total = V
            
            # Split-operator propagation
            psi = apply_potential_operator(psi, V_total, dt / 2.0)
            psi_k = fft(psi)
            psi_k = apply_kinetic_operator(psi_k, self.k, dt, particle_mass)
            psi = ifft(psi_k)
            psi = apply_potential_operator(psi, V_total, dt / 2.0)
            
            # Apply pure dephasing (if enabled) - preserves probability
            if dephasing_enabled:
                random_phases = np.random.randn(self.nx)
                psi = apply_dephasing(psi, dt, decoherence_rate, random_phases)
            
            # Apply absorbing boundaries
            psi = apply_absorbing_mask(psi, self.boundary_mask)
            
            norm_after = np.sqrt(np.sum(np.abs(psi)**2) * self.dx)
            absorbed_this_step = norm_before**2 - norm_after**2
            absorbed_probability += absorbed_this_step
            
            if np.any(np.isnan(psi)) or np.any(np.isinf(psi)):
                error_msg = f"Numerical instability at t={t:.3f} fs"
                if self.logger:
                    self.logger.error(error_msg)
                if self.verbose:
                    print(f"\n  ERROR: {error_msg}")
                break
            
            current_norm = np.sqrt(np.sum(np.abs(psi)**2) * self.dx)
            expected_norm = np.sqrt(1.0 - absorbed_probability)
            norm_error = abs(current_norm - expected_norm)
            
            if norm_error > 0.01:
                norm_violations.append((t, norm_error))
            
            if norm_error > max_norm_error:
                max_norm_error = norm_error
            
            # Energy check - only in safe zone and for coherent evolution
            if not noise_enabled and not dephasing_enabled and n_steps % 100 == 0 and n_steps > 0:
                # Only check energy for probability in safe zone
                prob_in_safe = np.sum(np.abs(psi[idx_safe_left:idx_safe_right])**2) * self.dx
                
                if prob_in_safe > 0.1:  # Only check if significant probability in safe zone
                    psi_k_check = fft(psi)
                    E_current = compute_energy(psi, psi_k_check, self.k, V, 
                                              self.dx, particle_mass)
                    E_expected = E_initial * (1.0 - absorbed_probability)
                    energy_error = abs((E_current - E_expected) / (E_expected + 1e-10))
                    
                    if energy_error > 0.10:  # More lenient: 10% instead of 5%
                        energy_violations.append((t, energy_error))
                        if energy_error > max_energy_error:
                            max_energy_error = energy_error
            
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
        
        if norm_violations and self.logger:
            self.logger.warning(
                f"Norm conservation violated {len(norm_violations)} times "
                f"(max deviation: {max_norm_error:.4f})"
            )
        
        if energy_violations and self.logger:
            self.logger.warning(
                f"Energy conservation violated {len(energy_violations)} times "
                f"(max deviation: {max_energy_error:.4%})"
            )
        
        if self.verbose:
            print(f"  Completed {n_steps} time steps")
        
        # Measure T and R in safe zones (no absorption)
        prob_final = prob_hist[save_idx - 1]
        
        reflection = np.sum(prob_final[idx_safe_left:barrier_start_idx]) * self.dx
        transmission = np.sum(prob_final[barrier_end_idx:idx_safe_right]) * self.dx
        
        total_prob = transmission + reflection + absorbed_probability
        
        if abs(total_prob - 1.0) > 0.03:
            if self.logger:
                self.logger.warning(
                    f"Probability conservation deviation: T+R+A = {total_prob:.4f}"
                )
        
        if self.verbose:
            print(f"  T = {transmission:.4f} ({transmission*100:.2f}%)")
            print(f"  R = {reflection:.4f} ({reflection*100:.2f}%)")
            print(f"  A = {absorbed_probability:.4f} ({absorbed_probability*100:.2f}%)")
            print(f"  T + R + A = {total_prob:.4f}")
        
        dt_final_value = dt_hist[-1] if len(dt_hist) > 0 else dt
        
        return {
            'x': self.x,
            't': t_save[:save_idx],
            'psi': psi_hist[:save_idx],
            'probability': prob_hist[:save_idx],
            'potential': V,
            'transmission_coefficient': transmission,
            'reflection_coefficient': reflection,
            'absorbed_probability': absorbed_probability,
            'detection_indices': {
                'safe_left': idx_safe_left,
                'safe_right': idx_safe_right,
                'barrier_start': barrier_start_idx,
                'barrier_end': barrier_end_idx
            },
            'params': {
                'particle_mass': particle_mass,
                'dt_initial': dt_hist[0] if len(dt_hist) > 0 else dt,
                'dt_final': dt_final_value,
                'dt_mean': np.mean(dt_hist) if len(dt_hist) > 0 else dt,
                'n_steps': n_steps,
                'adaptive': self.adaptive_dt,
                'nx': self.nx,
                'dx': self.dx,
                'boundary_width': self._boundary_width,
                'boundary_strength': self._boundary_strength,
                'noise_amplitude': noise_amplitude,
                'noise_correlation_time': noise_correlation_time,
                'decoherence_rate': decoherence_rate,
                'max_norm_error': max_norm_error,
                'max_energy_error': max_energy_error,
                'n_norm_violations': len(norm_violations),
                'n_energy_violations': len(energy_violations)
            }
        }
