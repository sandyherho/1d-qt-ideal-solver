"""
Simulation Logger with Enhanced Error Reporting
"""

import logging
from pathlib import Path
from datetime import datetime


class SimulationLogger:
    """Enhanced logger for quantum tunneling simulations."""
    
    def __init__(self, scenario_name: str, log_dir: str = "logs", verbose: bool = True):
        """
        Initialize simulation logger.
        
        Args:
            scenario_name: Name of scenario (used in log filename)
            log_dir: Directory for log files
            verbose: If True, also print warnings/errors to console
        """
        self.scenario_name = scenario_name
        self.log_dir = Path(log_dir)
        self.verbose = verbose
        
        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate simple filename WITHOUT timestamp (cleaner)
        # e.g., "case1_rectangular.log" instead of "case_1___rectangular_barrier_20251008_203215.log"
        self.log_file = self.log_dir / f"{scenario_name}.log"
        
        # Set up Python logging
        self.logger = self._setup_logger()
        
        # Track all warnings and errors for summary
        self.warnings = []
        self.errors = []
    
    def _setup_logger(self):
        """Configure Python logging with file handler and formatter."""
        # Create logger with unique name
        logger = logging.getLogger(f"qt1d_{self.scenario_name}")
        logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers (important for multiple runs)
        logger.handlers = []
        
        # Create file handler
        handler = logging.FileHandler(self.log_file, mode='w')  # 'w' = overwrite
        handler.setLevel(logging.DEBUG)
        
        # Create formatter: timestamp - level - message
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        # Attach handler to logger
        logger.addHandler(handler)
        
        return logger
    
    def info(self, msg: str):
        """Log informational message."""
        self.logger.info(msg)
    
    def warning(self, msg: str):
        """Log warning message and track for summary."""
        self.logger.warning(msg)
        self.warnings.append(msg)
        
        if self.verbose:
            print(f"  WARNING: {msg}")
    
    def error(self, msg: str):
        """Log error message and track for summary."""
        self.logger.error(msg)
        self.errors.append(msg)
        
        if self.verbose:
            print(f"  ERROR: {msg}")
    
    def log_parameters(self, params: dict):
        """Log all simulation parameters in organized format."""
        self.info("=" * 60)
        self.info(f"SIMULATION PARAMETERS - {params.get('scenario_name', 'Unknown')}")
        self.info("=" * 60)
        
        for key, value in sorted(params.items()):
            self.info(f"  {key}: {value}")
        
        self.info("=" * 60)
    
    def log_timing(self, timing: dict):
        """Log timing breakdown for different simulation phases."""
        self.info("=" * 60)
        self.info("TIMING BREAKDOWN")
        self.info("=" * 60)
        
        for key, value in sorted(timing.items()):
            self.info(f"  {key}: {value:.3f} s")
        
        self.info("=" * 60)
    
    def log_results(self, results: dict):
        """Log simulation results with automatic validation."""
        self.info("=" * 60)
        self.info("SIMULATION RESULTS")
        self.info("=" * 60)
        
        T = results['transmission_coefficient']
        R = results['reflection_coefficient']
        params = results['params']
        
        self.info(f"  Transmission coefficient: {T:.6f} ({T*100:.3f}%)")
        self.info(f"  Reflection coefficient: {R:.6f} ({R*100:.3f}%)")
        self.info(f"  T + R sum: {T+R:.6f}")
        
        if abs(T + R - 1.0) > 0.05:
            self.warning(f"T + R = {T+R:.4f} deviates significantly from 1.0")
        
        self.info(f"  Time steps: {params['n_steps']}")
        self.info(f"  dt (initial/mean/final): "
                 f"{params['dt_initial']:.6f} / "
                 f"{params['dt_mean']:.6f} / "
                 f"{params['dt_final']:.6f} fs")
        
        if params.get('noise_amplitude', 0) > 0:
            self.info("  --- Stochastic Noise ---")
            self.info(f"  Noise amplitude: {params['noise_amplitude']:.4f} eV")
            self.info(f"  Noise correlation time: {params['noise_correlation_time']:.4f} fs")
        
        if params.get('decoherence_rate', 0) > 0:
            decoherence_time = 1.0 / params['decoherence_rate']
            self.info(f"  Decoherence rate: {params['decoherence_rate']:.4f} fs⁻¹")
            self.info(f"  Coherence time T₂: {decoherence_time:.2f} fs")
        
        if params.get('n_norm_violations', 0) > 0:
            self.warning(
                f"Norm violated {params['n_norm_violations']} times "
                f"(max: {params['max_norm_error']:.6f})"
            )
        else:
            self.info("  ✓ Norm conservation maintained")
        
        if params.get('n_energy_violations', 0) > 0:
            self.warning(
                f"Energy violated {params['n_energy_violations']} times "
                f"(max: {params['max_energy_error']:.6%})"
            )
        elif params.get('noise_amplitude', 0) == 0:
            self.info("  ✓ Energy conservation maintained")
        
        self.info("=" * 60)
    
    def finalize(self):
        """Write final summary and close logger."""
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
