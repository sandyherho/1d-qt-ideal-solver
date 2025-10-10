"""
Simulation Logger with correct probability conservation checks
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
        """Log simulation results with correct conservation check."""
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
        
        conservation_error = abs(total - 1.0)
        
        if conservation_error > 0.05:
            self.warning(
                f"Probability conservation violated: T+R+A = {total:.4f} "
                f"deviates from 1.0 by {conservation_error:.4f}"
            )
        elif conservation_error > 0.01:
            self.warning(
                f"Minor conservation deviation: T+R+A = {total:.4f} "
                f"(error: {conservation_error:.4f})"
            )
        else:
            self.info("  ✓ Probability conservation: T + R + A ≈ 1.0")
        
        self.info(f"  Time steps: {params['n_steps']}")
        
        dt_final = params.get('dt_final', params.get('dt_mean', 0.0))
        self.info(f"  dt (initial/mean/final): "
                 f"{params['dt_initial']:.6f} / "
                 f"{params['dt_mean']:.6f} / "
                 f"{dt_final:.6f} fs")
        
        if params.get('noise_amplitude', 0) > 0 or params.get('decoherence_rate', 0) > 0:
            self.info("  --- Stochastic Environment ---")
            
            if params.get('noise_amplitude', 0) > 0:
                self.info(f"  Noise amplitude: {params['noise_amplitude']:.4f} eV")
                self.info(f"  Noise correlation time: {params['noise_correlation_time']:.4f} fs")
            
            if params.get('decoherence_rate', 0) > 0:
                T2 = 1.0 / params['decoherence_rate']
                self.info(f"  Decoherence rate: {params['decoherence_rate']:.4f} fs⁻¹")
                self.info(f"  Coherence time T₂: {T2:.2f} fs")
        
        if params.get('n_norm_violations', 0) > 0:
            self.warning(
                f"Wavefunction norm deviated {params['n_norm_violations']} times "
                f"(max: {params['max_norm_error']:.6f})"
            )
        else:
            self.info("  ✓ Wavefunction norm stable")
        
        if params.get('n_energy_violations', 0) > 0:
            self.warning(
                f"Energy conservation violated {params['n_energy_violations']} times "
                f"(max: {params['max_energy_error']:.4%})"
            )
        elif params.get('noise_amplitude', 0) == 0 and params.get('decoherence_rate', 0) == 0:
            self.info("  ✓ Energy conservation maintained")
        
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
            self.info("  ERRORS: None ✓")
        
        if self.warnings:
            self.info(f"  WARNINGS: {len(self.warnings)}")
            for i, warn in enumerate(self.warnings, 1):
                self.info(f"    {i}. {warn}")
        else:
            self.info("  WARNINGS: None ✓")
        
        self.info(f"  Log file: {self.log_file}")
        self.info("=" * 60)
        self.info(f"Simulation completed: {self.scenario_name}")
        self.info("=" * 60)
