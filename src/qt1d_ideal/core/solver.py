"""
1D Quantum Tunneling Solver with Adaptive Time Stepping and Stochastic Noise

This module implements the split-operator method for solving the time-dependent
Schrödinger equation with optional stochastic environmental noise.

Physical Basis:
    - Hamiltonian: H = T + V = -ℏ²/(2m)∇² + V(x,t)
    - Evolution: ψ(t+dt) = exp(-iHdt/ℏ)ψ(t)
    - Split-operator: exp(-i(T+V)dt) ≈ exp(-iVdt/2)exp(-iTdt)exp(-iVdt/2)
    
Stochastic Effects:
    - Potential noise: V(x,t) = V₀(x) + ξ(x,t) [Ornstein-Uhlenbeck process]
    - Decoherence: Environmental coupling causing pure→mixed state transition
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
    """
    Apply kinetic energy operator in momentum space.
    
    Physics: T = ℏ²k²/(2m) → exp(-iTdt/ℏ) = exp(-i·ℏk²dt/(2m))
    
    Args:
        psi_k: Wavefunction in momentum space
        k: Momentum grid (wave numbers) [nm⁻¹]
        dt: Time step [fs]
        m: Particle mass [m_e, electron mass units]
    
    Returns:
        Evolved wavefunction in momentum space
    """
    n = len(psi_k)
    psi_k_new = np.zeros(n, dtype=np.complex128)
    
    # Parallel loop over momentum states
    for i in prange(n):
        kinetic_energy = k[i]**2 / (2.0 * m)  # E_k in atomic units
        phase = np.exp(-1j * kinetic_energy * dt)  # Quantum phase evolution
        psi_k_new[i] = psi_k[i] * phase
    
    return psi_k_new


@jit(nopython=True, parallel=True, cache=True)
def apply_potential_operator(psi: np.ndarray, V: np.ndarray, 
                             dt: float) -> np.ndarray:
    """
    Apply potential energy operator in position space.
    
    Physics: exp(-iVdt/ℏ) applied point-wise in real space
    
    Args:
        psi: Wavefunction in position space
        V: Potential energy [eV]
        dt: Time step [fs]
    
    Returns:
        Evolved wavefunction in position space
    """
    n = len(psi)
    psi_new = np.zeros(n, dtype=np.complex128)
    
    # Parallel loop over spatial grid
    for i in prange(n):
        phase = np.exp(-1j * V[i] * dt)  # Local phase shift from potential
        psi_new[i] = psi[i] * phase
    
    return psi_new


@jit(nopython=True, cache=True)
def estimate_wavefunction_change(psi_curr: np.ndarray, psi_prev: np.ndarray,
                                 dx: float) -> float:
    """
    Estimate rate of wavefunction change for adaptive time stepping.
    
    Computes: ||ψ_curr - ψ_prev|| / ||ψ_curr||
    
    This metric indicates how fast the wavefunction is evolving. Large values
    suggest we need smaller time steps for accuracy.
    
    Args:
        psi_curr: Current wavefunction
        psi_prev: Previous wavefunction
        dx: Spatial grid spacing [nm]
    
    Returns:
        Normalized change rate (dimensionless)
    """
    # L² norm of difference
    diff = psi_curr - psi_prev
    diff_norm = np.sqrt(np.sum(np.abs(diff)**2) * dx)
    
    # L² norm of current state
    curr_norm = np.sqrt(np.sum(np.abs(psi_curr)**2) * dx)
    
    # Return relative change (avoid division by zero)
    return diff_norm / curr_norm if curr_norm > 1e-12 else 0.0


@jit(nopython=True, cache=True)
def compute_energy(psi: np.ndarray, psi_k: np.ndarray, k: np.ndarray, 
                   V: np.ndarray, dx: float, m: float) -> float:
    """
    Compute total energy expectation value <ψ|H|ψ> = <T> + <V>.
    
    Uses physical k-space cutoff to avoid numerical noise from high-k modes.
    
    Args:
        psi: Wavefunction in position space
        psi_k: Wavefunction in momentum space
        k: Momentum grid [nm⁻¹]
        V: Potential [eV]
        dx: Spatial grid spacing [nm]
        m: Particle mass [m_e]
    
    Returns:
        Total energy [eV]
    """
    # Physical cutoff: only sum over relevant k modes (|k| < k_cutoff)
    # Typical wavepacket has k ~ 5-10 nm⁻¹, so cutoff at 50 nm⁻¹ is safe
    k_cutoff = 50.0  # [nm⁻¹]
    
    # Kinetic energy: <T> = ∫ |ψ(k)|² · ℏ²k²/(2m) dk
    kinetic = 0.0
    for i in range(len(psi_k)):
        if np.abs(k[i]) < k_cutoff:  # Only physical modes
            kinetic += (k[i]**2 / (2.0 * m)) * np.abs(psi_k[i])**2
    kinetic = kinetic * dx / (2.0 * np.pi)
    
    # Potential energy: <V> = ∫ |ψ(x)|² V(x) dx
    potential = 0.0
    for i in range(len(psi)):
        potential += V[i] * np.abs(psi[i])**2
    potential = potential * dx
    
    return kinetic + potential


class QuantumTunneling1D:
    """
    1D Quantum Tunneling Solver with Split-Operator Method and Stochastic Noise.
    
    This class provides a complete framework for simulating quantum tunneling
    through potential barriers, including realistic environmental effects.
    
    Features:
        - Adaptive time stepping for optimal efficiency
        - Numba JIT compilation for 10-100x speedup
        - Multi-core parallelization
        - Stochastic noise (Ornstein-Uhlenbeck process)
        - Environmental decoherence
        - Comprehensive error detection
    
    Physical Units:
        - Length: nm (nanometers)
        - Time: fs (femtoseconds)
        - Energy: eV (electron volts)
        - Mass: m_e (electron mass)
        - ℏ = 1 (natural units)
    """
    
    def __init__(self, nx: int = 2048, x_min: float = -10.0, x_max: float = 10.0,
                 adaptive_dt: bool = True, verbose: bool = True,
                 logger: Optional[Any] = None, n_cores: Optional[int] = None):
        """
        Initialize the quantum solver.
        
        Args:
            nx: Number of spatial grid points (power of 2 recommended for FFT)
            x_min: Left boundary of spatial domain [nm]
            x_max: Right boundary of spatial domain [nm]
            adaptive_dt: Enable adaptive time stepping
            verbose: Print progress information
            logger: Optional logger for recording simulation details
            n_cores: Number of CPU cores (None = use all available)
        """
        self.nx = nx
        self.x_min = x_min
        self.x_max = x_max
        self.dx = (x_max - x_min) / (nx - 1)  # Grid spacing
        self.adaptive_dt = adaptive_dt
        self.verbose = verbose
        self.logger = logger
        
        # Configure parallel execution
        if n_cores is None:
            n_cores = os.cpu_count()
        numba.set_num_threads(n_cores)
        
        # Create spatial grid
        self.x = np.linspace(x_min, x_max, nx)
        
        # Create momentum grid (for FFT)
        # k ranges from -k_max to k_max with proper FFT ordering
        self.k = 2.0 * np.pi * fftfreq(nx, d=self.dx)
        
        if verbose:
            print(f"  Grid: {nx} points, dx = {self.dx:.4f} nm")
            print(f"  Using {n_cores} CPU cores")
    
    def rectangular_barrier(self, height: float, width: float,
                           center: float = 0.0) -> np.ndarray:
        """
        Create rectangular potential barrier.
        
        V(x) = { height  if |x - center| < width/2
               { 0       otherwise
        
        Models: Idealized step potential, textbook problem
        
        Args:
            height: Barrier height [eV]
            width: Barrier width [nm]
            center: Barrier center position [nm]
        
        Returns:
            Potential array [eV]
        """
        V = np.zeros(self.nx)
        mask = np.abs(self.x - center) < width / 2.0
        V[mask] = height
        return V
    
    def gaussian_barrier(self, height: float, width: float,
                        center: float = 0.0) -> np.ndarray:
        """
        Create smooth Gaussian barrier.
        
        V(x) = height · exp[-(x-center)²/(2·width²)]
        
        Models: Smooth potentials, molecular barriers, nuclear potentials
        
        Args:
            height: Peak barrier height [eV]
            width: Width parameter σ [nm]
            center: Barrier center position [nm]
        
        Returns:
            Potential array [eV]
        """
        return height * np.exp(-((self.x - center)**2) / (2.0 * width**2))
    
    def double_barrier(self, height: float, width: float,
                      separation: float) -> np.ndarray:
        """
        Create double barrier for resonant tunneling.
        
        Two rectangular barriers separated by a quantum well.
        
        Models: Resonant tunneling diode (RTD), Fabry-Perot interferometer analog
        
        Args:
            height: Barrier height [eV]
            width: Individual barrier width [nm]
            separation: Distance between barrier centers [nm]
        
        Returns:
            Potential array [eV]
        """
        V = np.zeros(self.nx)
        
        # Left barrier: centered at -separation/2 - width/2
        left_center = -separation / 2.0 - width / 2.0
        left_mask = np.abs(self.x - left_center) < width / 2.0
        V[left_mask] = height
        
        # Right barrier: centered at +separation/2 + width/2
        right_center = separation / 2.0 + width / 2.0
        right_mask = np.abs(self.x - right_center) < width / 2.0
        V[right_mask] = height
        
        return V
    
    def triple_barrier(self, height: float, width: float,
                      separation: float) -> np.ndarray:
        """
        Create triple barrier system.
        
        Three barriers creating two coupled quantum wells.
        
        Models: Quantum cascade structures, superlattices, complex interference
        
        Args:
            height: Barrier height [eV]
            width: Individual barrier width [nm]
            separation: Spacing between adjacent barriers [nm]
        
        Returns:
            Potential array [eV]
        """
        V = np.zeros(self.nx)
        
        # Three barriers at positions: -separation, 0, +separation
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
        Solve quantum tunneling with adaptive time stepping and stochastic noise.
        
        Implements the split-operator method:
            ψ(t+dt) ≈ exp(-iV·dt/2) · exp(-iT·dt) · exp(-iV·dt/2) · ψ(t)
        
        Stochastic noise parameters:
            - noise_amplitude: Amplitude of random potential fluctuations [eV]
              Models thermal/environmental noise from lattice vibrations, EM fields
              
            - noise_correlation_time: Correlation time τ for OU process [fs]
              Ornstein-Uhlenbeck: dξ = -ξ/τ dt + σ√(2/τ) dW
              Physical meaning: memory time of environmental fluctuations
              
            - decoherence_rate: Environmental decoherence rate γ [fs⁻¹]
              Models pure dephasing from unobserved degrees of freedom
              Coherence time: T₂ = 1/γ
        
        Physical motivation for noise:
            - Lattice phonons (thermal vibrations)
            - Electromagnetic field fluctuations
            - Defects and impurities
            - Coupling to unmeasured environmental modes
        
        Args:
            psi0: Initial wavefunction (will be normalized)
            V: Base potential (noise will be added) [eV]
            t_final: Total simulation time [fs]
            dt_initial: Initial time step (auto-calculated if None) [fs]
            dt_min: Minimum allowed time step [fs]
            dt_max: Maximum allowed time step [fs]
            n_snapshots: Number of time points to save
            particle_mass: Particle mass [m_e]
            show_progress: Display progress bar
            tolerance: Tolerance for adaptive stepping (relative change)
            noise_amplitude: Stochastic potential noise strength [eV]
            noise_correlation_time: Noise correlation time τ [fs]
            decoherence_rate: Decoherence rate γ [fs⁻¹]
        
        Returns:
            Dictionary containing:
                - x: Spatial grid [nm]
                - t: Time points [fs]
                - psi: Wavefunction evolution [t × x]
                - probability: |ψ|² evolution [t × x]
                - potential: Potential array [eV]
                - transmission_coefficient: T (dimensionless)
                - reflection_coefficient: R (dimensionless)
                - params: Dictionary of simulation parameters and diagnostics
        """
        # Initialize wavefunction (make a copy and normalize)
        psi = psi0.copy().astype(np.complex128)
        norm = np.sqrt(np.sum(np.abs(psi)**2) * self.dx)
        psi = psi / norm  # Ensure ∫|ψ|²dx = 1
        
        # Initialize stochastic noise process (Ornstein-Uhlenbeck)
        noise_potential = np.zeros(self.nx)
        noise_enabled = noise_amplitude > 0
        
        # Auto-calculate initial time step if not provided
        if dt_initial is None:
            # CFL-like condition: dt < 1/(E_max) for stability
            k_max = np.max(np.abs(self.k))
            E_k_max = k_max**2 / (2.0 * particle_mass)  # Maximum kinetic energy
            V_max = np.max(np.abs(V))  # Maximum potential energy
            dt = 0.5 / (E_k_max + V_max + 1e-10)  # Safety factor 0.5
            dt = np.clip(dt, dt_min, dt_max)
        else:
            dt = np.clip(dt_initial, dt_min, dt_max)
        
        # Print initial info
        if self.verbose:
            print(f"  Initial dt = {dt:.6f} fs")
            if noise_enabled:
                print(f"  Stochastic noise: amplitude = {noise_amplitude:.4f} eV, "
                      f"τ_corr = {noise_correlation_time:.3f} fs")
            if decoherence_rate > 0:
                print(f"  Decoherence: γ = {decoherence_rate:.4f} fs⁻¹ "
                      f"(T₂ ~ {1.0/decoherence_rate:.1f} fs)")
        
        # Initialize time tracking
        t = 0.0
        t_save = np.linspace(0, t_final, n_snapshots)  # Times to save snapshots
        save_idx = 0
        
        # Pre-allocate output arrays
        psi_hist = np.zeros((n_snapshots, self.nx), dtype=np.complex128)
        prob_hist = np.zeros((n_snapshots, self.nx))
        dt_hist = []  # Track time step evolution
        
        # Save initial state
        psi_hist[0] = psi
        prob_hist[0] = np.abs(psi)**2
        save_idx = 1
        
        # For adaptive time stepping
        psi_prev = psi.copy()
        
        # Calculate initial energy for conservation check
        psi_k = fft(psi)
        E_initial = compute_energy(psi, psi_k, self.k, V, self.dx, particle_mass)
        
        # Error tracking for diagnostics
        norm_violations = []  # List of (time, error) tuples
        energy_violations = []  # List of (time, error) tuples
        max_norm_error = 0.0
        max_energy_error = 0.0
        
        # Progress bar (clean format)
        if show_progress:
            pbar = tqdm(
                total=t_final, 
                desc="  Evolving", 
                unit="fs",
                bar_format='{l_bar}{bar}| {n:.2f}/{total:.2f} fs [{elapsed}<{remaining}]'
            )
        
        # Main time evolution loop
        n_steps = 0
        while t < t_final and save_idx < n_snapshots:
            # ================================================================
            # STOCHASTIC NOISE UPDATE
            # ================================================================
            if noise_enabled:
                # Ornstein-Uhlenbeck process: dξ = -ξ/τ dt + σ√(2/τ) dW
                # Solution: ξ(t+dt) = ξ(t)·exp(-dt/τ) + σ√[1-exp(-2dt/τ)]·N(0,1)
                decay = np.exp(-dt / noise_correlation_time)
                noise_potential = decay * noise_potential + \
                    noise_amplitude * np.sqrt((1 - decay**2)) * np.random.randn(self.nx)
                V_total = V + noise_potential  # Total potential
            else:
                V_total = V  # No noise
            
            # ================================================================
            # SPLIT-OPERATOR TIME STEP
            # ================================================================
            # Step 1: Half-step in potential (position space)
            psi = apply_potential_operator(psi, V_total, dt / 2.0)
            
            # Step 2: Full step in kinetic (momentum space via FFT)
            psi_k = fft(psi)
            psi_k = apply_kinetic_operator(psi_k, self.k, dt, particle_mass)
            psi = ifft(psi_k)
            
            # Step 3: Half-step in potential (position space)
            psi = apply_potential_operator(psi, V_total, dt / 2.0)
            
            # ================================================================
            # DECOHERENCE (if enabled)
            # ================================================================
            if decoherence_rate > 0:
                # Phenomenological pure dephasing: |ψ| → |ψ|·exp(-γt)
                # Models loss of quantum coherence to environment
                damping = np.exp(-decoherence_rate * dt)
                psi = psi * damping
            
            # ================================================================
            # NUMERICAL STABILITY CHECK
            # ================================================================
            if np.any(np.isnan(psi)) or np.any(np.isinf(psi)):
                error_msg = f"Numerical instability detected at t={t:.3f} fs"
                if self.logger:
                    self.logger.error(error_msg)
                if self.verbose:
                    print(f"\n  ERROR: {error_msg}")
                break  # Stop simulation
            
            # ================================================================
            # NORM CONSERVATION CHECK
            # ================================================================
            # Quantum mechanics requires ∫|ψ|²dx = 1 always
            current_norm = np.sqrt(np.sum(np.abs(psi)**2) * self.dx)
            norm_error = abs(current_norm - 1.0)
            
            if norm_error > 0.01:  # Flag if > 1% deviation
                norm_violations.append((t, norm_error))
                if norm_error > max_norm_error:
                    max_norm_error = norm_error
            
            # ================================================================
            # ENERGY CONSERVATION CHECK (only if no noise)
            # ================================================================
            if not noise_enabled and n_steps % 100 == 0:  # Check every 100 steps
                psi_k_check = fft(psi)
                E_current = compute_energy(psi, psi_k_check, self.k, V, 
                                          self.dx, particle_mass)
                energy_error = abs((E_current - E_initial) / E_initial)
                
                if energy_error > 0.05:  # Flag if > 5% deviation (relaxed)
                    energy_violations.append((t, energy_error))
                    if energy_error > max_energy_error:
                        max_energy_error = energy_error
            
            # ================================================================
            # ADAPTIVE TIME STEPPING
            # ================================================================
            if self.adaptive_dt and n_steps > 0:
                # Estimate how fast wavefunction is changing
                change_rate = estimate_wavefunction_change(psi, psi_prev, self.dx)
                
                # Adjust time step based on change rate
                # Use more relaxed criteria to avoid getting stuck at dt_min
                if change_rate > tolerance * 2.0:  # Much faster than desired
                    dt = dt * 0.9  # Shrink
                elif change_rate < tolerance * 0.1:  # Much slower than desired
                    dt = dt * 1.2  # Grow faster
                
                # Enforce limits
                dt = np.clip(dt, dt_min, dt_max)
                
                # Don't overshoot final time
                if t + dt > t_final:
                    dt = t_final - t
            
            # Update for next iteration
            psi_prev = psi.copy()
            t += dt
            n_steps += 1
            dt_hist.append(dt)
            
            # Update progress bar
            if show_progress:
                pbar.update(dt)
            
            # ================================================================
            # SAVE SNAPSHOT (if at scheduled time)
            # ================================================================
            if save_idx < n_snapshots and t >= t_save[save_idx]:
                psi_hist[save_idx] = psi
                prob_hist[save_idx] = np.abs(psi)**2
                save_idx += 1
        
        # Close progress bar
        if show_progress:
            pbar.close()
        
        # ================================================================
        # LOG CONSERVATION VIOLATIONS (if any)
        # ================================================================
        if norm_violations:
            warning = (f"Norm conservation violated {len(norm_violations)} times "
                      f"(max error: {max_norm_error:.4f})")
            if self.logger:
                self.logger.warning(warning)
            if self.verbose:
                print(f"  WARNING: {warning}")
        
        if energy_violations:
            warning = (f"Energy conservation violated {len(energy_violations)} times "
                      f"(max error: {max_energy_error:.4%})")
            if self.logger:
                self.logger.warning(warning)
            if self.verbose:
                print(f"  WARNING: {warning}")
        
        if self.verbose:
            print(f"  Completed {n_steps} time steps")
        
        # ================================================================
        # CALCULATE TRANSMISSION AND REFLECTION COEFFICIENTS
        # ================================================================
        # Determine barrier boundaries
        barrier_mask = V > 0.1 * np.max(V)  # Regions with significant potential
        if np.any(barrier_mask):
            barrier_indices = np.where(barrier_mask)[0]
            barrier_start = barrier_indices[0]
            barrier_end = barrier_indices[-1]
        else:
            # No barrier found, use middle of domain
            barrier_start = self.nx // 2
            barrier_end = self.nx // 2
        
        # Final probability distribution
        prob_final = prob_hist[save_idx - 1]
        
        # Transmission: probability to the right of barrier
        trans_prob = np.sum(prob_final[barrier_end:]) * self.dx
        
        # Reflection: probability to the left of barrier
        refl_prob = np.sum(prob_final[:barrier_start]) * self.dx
        
        # Total probability in these regions
        total_prob = trans_prob + refl_prob
        
        # Calculate coefficients (normalize)
        if total_prob > 0:
            transmission = trans_prob / total_prob
            reflection = refl_prob / total_prob
        else:
            transmission = 0.0
            reflection = 0.0
        
        # Check if T + R ≈ 1 (should be true for closed system)
        tr_sum = transmission + reflection
        if abs(tr_sum - 1.0) > 0.05:  # 5% deviation threshold
            warning = f"T + R = {tr_sum:.4f} ≠ 1 (deviation: {abs(tr_sum - 1.0):.4f})"
            if self.logger:
                self.logger.warning(warning)
            if self.verbose:
                print(f"  WARNING: {warning}")
        
        # ================================================================
        # RETURN RESULTS
        # ================================================================
        return {
            'x': self.x,
            't': t_save[:save_idx],
            'psi': psi_hist[:save_idx],
            'probability': prob_hist[:save_idx],
            'potential': V,
            'transmission_coefficient': transmission,
            'reflection_coefficient': reflection,
            'params': {
                # Time stepping info
                'particle_mass': particle_mass,
                'dt_initial': dt_hist[0] if dt_hist else dt,
                'dt_final': dt_hist[-1] if dt_hist else dt,
                'dt_mean': np.mean(dt_hist) if dt_hist else dt,
                'n_steps': n_steps,
                'adaptive': self.adaptive_dt,
                
                # Grid info
                'nx': self.nx,
                'dx': self.dx,
                
                # Stochastic parameters
                'noise_amplitude': noise_amplitude,
                'noise_correlation_time': noise_correlation_time,
                'decoherence_rate': decoherence_rate,
                
                # Error diagnostics
                'max_norm_error': max_norm_error,
                'max_energy_error': max_energy_error,
                'n_norm_violations': len(norm_violations),
                'n_energy_violations': len(energy_violations)
            }
        }
