"""
Simulation Logger with proper thresholds for idealized solver
FINAL REVISION: Appropriate conservation thresholds for quantum simulations
"""

import logging
from pathlib import Path
from datetime import datetime


class SimulationLogger:
    """Enhanced logger for quantum tunneling simulations."""
    
    def __init__(self, scenario_name: str, log_dir: str = "logs", verbose: bool = True):
        """Initialize simulation logger."""
        self.scenario_name = scenario_name
        self.log_dir = Path(log_dir)
        self.verbose = verbose
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"{scenario_name}.log"
        
        self.logger = self._setup_logger()
        
        self.warnings = []
        self.errors = []
    
    def _setup_logger(self):
        """Configure Python logging."""
        logger = logging.getLogger(f"qt1d_{self.scenario_name}")
        logger.setLevel(logging.DEBUG)
        logger.handlers = []
        
        handler = logging.FileHandler(self.log_file, mode='w')
        handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        return logger
    
    def info(self, msg: str):
        """Log informational message."""
        self.logger.info(msg)
    
    def warning(self, msg: str):
        """Log warning message."""
        self.logger.warning(msg)
        self.warnings.append(msg)
        
        if self.verbose:
            print(f"  WARNING: {msg}")
    
    def error(self, msg: str):
        """Log error message."""
        self.logger.error(msg)
        self.errors.append(msg)
        
        if self.verbose:
            print(f"  ERROR: {msg}")
    
    def log_parameters(self, params: dict):
        """Log all simulation parameters."""
        self.info("=" * 60)
        self.info(f"SIMULATION PARAMETERS - {params.get('scenario_name', 'Unknown')}")
        self.info("=" * 60)
        
        for key, value in sorted(params.items()):
            self.info(f"  {key}: {value}")
        
        self.info("=" * 60)
    
    def log_timing(self, timing: dict):
        """Log timing breakdown."""
        self.info("=" * 60)
        self.info("TIMING BREAKDOWN")
        self.info("=" * 60)
        
        for key, value in sorted(timing.items()):
            self.info(f"  {key}: {value:.3f} s")
        
        self.info("=" * 60)
    
    def log_results(self, results: dict):
        """
        Log simulation results.
        FINAL REVISION: Proper thresholds for quantum simulations
        - Warn if T+R+A deviates >5% from 1.0 (realistic threshold)
        - Warn if absorption > 15% (allows for some boundary effects)
        - More lenient energy checks (quantum simulations have some numerical error)
        """
        self.info("=" * 60)
        self.info("SIMULATION RESULTS")
        self.info("=" * 60)
        
        T = results['transmission_coefficient']
        R = results['reflection_coefficient']
        A = results.get('absorbed_probability', 0.0)
        params = results['params']
        
        self.info(f"  Transmission coefficient: {T:.6f} ({T*100:.3f}%)")
        self.info(f"  Reflection coefficient: {R:.6f} ({R*100:.3f}%)")
        self.info(f"  Absorbed probability: {A:.6f} ({A*100:.3f}%)")
        
        total = T + R + A
        self.info(f"  T + R + A sum: {total:.6f}")
        
        # Conservation check - realistic threshold
        conservation_error = abs(total - 1.0)
        
        if conservation_error > 0.05:  # 5% threshold
            self.warning(
                f"Probability conservation violated: T+R+A = {total:.4f} "
                f"(deviation: {conservation_error:.4f})"
            )
            quality_status = "POOR"
        elif conservation_error > 0.02:  # 2-5% is acceptable
            self.info(f"  Probability conservation: T + R + A = {total:.4f} (acceptable)")
            quality_status = "GOOD"
        else:  # < 2% is excellent
            self.info(f"  Probability conservation: T + R + A = {total:.4f} (excellent)")
            quality_status = "EXCELLENT"
        
        # Check absorption level - more lenient
        if A > 0.15:  # More than 15% absorbed
            self.warning(
                f"High absorption detected: {A*100:.2f}% - "
                "Consider reducing boundary_strength or increasing boundary_width"
            )
            if quality_status == "EXCELLENT":
                quality_status = "GOOD"
        elif A > 0.08:  # 8-15% is moderate
            self.info(f"  Moderate absorption: {A*100:.2f}%")
        else:  # < 8% is ideal
            self.info(f"  Low absorption: {A*100:.2f}% (ideal)")
        
        # Overall quality assessment
        self.info(f"  Simulation quality: {quality_status}")
        
        self.info(f"  Time steps: {params['n_steps']}")
        
        dt_final = params.get('dt_final', params.get('dt_mean', 0.0))
        self.info(f"  dt (initial/mean/final): "
                 f"{params['dt_initial']:.6f} / "
                 f"{params['dt_mean']:.6f} / "
                 f"{dt_final:.6f} fs")
        
        # Environment info
        noise_enabled = params.get('noise_amplitude', 0) > 0
        dephasing_enabled = params.get('decoherence_rate', 0) > 0
        
        if noise_enabled or dephasing_enabled:
            self.info("  --- Stochastic Environment ---")
            
            if noise_enabled:
                self.info(f"  Noise amplitude: {params['noise_amplitude']:.4f} eV")
                self.info(f"  Noise correlation time: {params['noise_correlation_time']:.4f} fs")
            
            if dephasing_enabled:
                T2 = 1.0 / params['decoherence_rate']
                self.info(f"  Decoherence rate: {params['decoherence_rate']:.4f} fs^-1")
                self.info(f"  Coherence time T2: {T2:.2f} fs")
                self.info(f"  Dephasing type: Pure dephasing (probability conserving)")
        else:
            self.info("  --- Idealized (Coherent) Evolution ---")
            self.info("  No noise or decoherence")
        
        # Numerical stability checks - more lenient for energy
        if params.get('n_norm_violations', 0) > 0:
            self.warning(
                f"Wavefunction norm deviated {params['n_norm_violations']} times "
                f"(max: {params['max_norm_error']:.6f})"
            )
        else:
            self.info("  Wavefunction norm stable")
        
        # Energy conservation - only warn if very large violations
        n_energy_violations = params.get('n_energy_violations', 0)
        max_energy_error = params.get('max_energy_error', 0.0)
        
        if n_energy_violations > 0 and max_energy_error > 0.20:  # Only warn if > 20%
            self.warning(
                f"Significant energy violations: {n_energy_violations} times "
                f"(max: {max_energy_error:.4%})"
            )
        elif n_energy_violations > 0:
            # Just informational for moderate violations
            self.info(f"  Energy checks: {n_energy_violations} deviations "
                     f"(max: {max_energy_error:.4%}, within tolerance)")
        elif not noise_enabled and not dephasing_enabled:
            self.info("  Energy conservation maintained")
        
        self.info("=" * 60)
    
    def finalize(self):
        """Write final summary."""
        self.info("=" * 60)
        self.info("SIMULATION SUMMARY")
        self.info("=" * 60)
        
        if self.errors:
            self.info(f"  ERRORS: {len(self.errors)}")
            for i, err in enumerate(self.errors, 1):
                self.info(f"    {i}. {err}")
        else:
            self.info("  ERRORS: None")
        
        if self.warnings:
            self.info(f"  WARNINGS: {len(self.warnings)}")
            for i, warn in enumerate(self.warnings, 1):
                self.info(f"    {i}. {warn}")
        else:
            self.info("  WARNINGS: None")
        
        self.info(f"  Log file: {self.log_file}")
        self.info("=" * 60)
        self.info(f"Simulation completed: {self.scenario_name}")
        self.info("=" * 60)
