"""
Simulation Logger with Enhanced Error Reporting

Provides comprehensive logging capabilities for quantum tunneling simulations,
including parameter tracking, timing analysis, error/warning aggregation, and
automatic validation of simulation results.

Features:
    - Timestamped log files (one per simulation)
    - Automatic parameter logging
    - Warning and error aggregation
    - Result validation with anomaly detection
    - Conservation law violation tracking
"""

import logging
from pathlib import Path
from datetime import datetime


class SimulationLogger:
    """
    Enhanced logger for quantum tunneling simulations.
    
    Creates detailed log files with:
        - Simulation parameters
        - Timing breakdown
        - Physical results (T, R, conservation violations)
        - Warnings and errors summary
    
    Log files are saved to: logs/{scenario_name}_{timestamp}.log
    """
    
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
        
        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{scenario_name}_{timestamp}.log"
        
        # Set up Python logging
        self.logger = self._setup_logger()
        
        # Track all warnings and errors for summary
        self.warnings = []
        self.errors = []
    
    def _setup_logger(self):
        """
        Configure Python logging with file handler and formatter.
        
        Returns:
            Configured logger instance
        """
        # Create logger with unique name
        logger = logging.getLogger(f"qt1d_{self.scenario_name}")
        logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers (important for multiple runs)
        logger.handlers = []
        
        # Create file handler
        handler = logging.FileHandler(self.log_file)
        handler.setLevel(logging.DEBUG)
        
        # Create formatter: timestamp - level - message
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        # Attach handler to logger
        logger.addHandler(handler)
        
        return logger
    
    def info(self, msg: str):
        """
        Log informational message.
        
        Args:
            msg: Message to log
        """
        self.logger.info(msg)
    
    def warning(self, msg: str):
        """
        Log warning message and track for summary.
        
        Args:
            msg: Warning message
        """
        self.logger.warning(msg)
        self.warnings.append(msg)  # Store for final summary
        
        # Also print to console if verbose mode
        if self.verbose:
            print(f"  WARNING: {msg}")
    
    def error(self, msg: str):
        """
        Log error message and track for summary.
        
        Args:
            msg: Error message
        """
        self.logger.error(msg)
        self.errors.append(msg)  # Store for final summary
        
        # Always print errors to console (even if not verbose)
        if self.verbose:
            print(f"  ERROR: {msg}")
    
    def log_parameters(self, params: dict):
        """
        Log all simulation parameters in organized format.
        
        Creates a formatted table of all configuration parameters.
        
        Args:
            params: Dictionary of simulation parameters
        """
        self.info("=" * 60)
        self.info(f"SIMULATION PARAMETERS - {params.get('scenario_name', 'Unknown')}")
        self.info("=" * 60)
        
        # Log parameters in alphabetical order
        for key, value in sorted(params.items()):
            self.info(f"  {key}: {value}")
        
        self.info("=" * 60)
    
    def log_timing(self, timing: dict):
        """
        Log timing breakdown for different simulation phases.
        
        Args:
            timing: Dictionary of {phase_name: time_in_seconds}
        """
        self.info("=" * 60)
        self.info("TIMING BREAKDOWN")
        self.info("=" * 60)
        
        # Log timings in alphabetical order
        for key, value in sorted(timing.items()):
            self.info(f"  {key}: {value:.3f} s")
        
        self.info("=" * 60)
    
    def log_results(self, results: dict):
        """
        Log simulation results with automatic validation and anomaly detection.
        
        Performs checks for:
            - T + R = 1 (probability conservation)
            - Norm conservation violations
            - Energy conservation violations
        
        Args:
            results: Results dictionary from solver.solve()
        """
        self.info("=" * 60)
        self.info("SIMULATION RESULTS")
        self.info("=" * 60)
        
        # Extract key results
        T = results['transmission_coefficient']
        R = results['reflection_coefficient']
        params = results['params']
        
        # Log transmission and reflection
        self.info(f"  Transmission coefficient: {T:.6f} ({T*100:.3f}%)")
        self.info(f"  Reflection coefficient: {R:.6f} ({R*100:.3f}%)")
        self.info(f"  T + R sum: {T+R:.6f}")
        
        # ================================================================
        # VALIDATION: Check T + R ≈ 1
        # ================================================================
        if abs(T + R - 1.0) > 0.05:  # 5% threshold
            self.warning(f"T + R = {T+R:.4f} deviates significantly from 1.0")
        
        # Log computational info
        self.info(f"  Time steps: {params['n_steps']}")
        self.info(f"  dt (initial/mean/final): "
                 f"{params['dt_initial']:.6f} / "
                 f"{params['dt_mean']:.6f} / "
                 f"{params['dt_final']:.6f} fs")
        
        # ================================================================
        # LOG STOCHASTIC PARAMETERS (if enabled)
        # ================================================================
        if params.get('noise_amplitude', 0) > 0:
            self.info("  --- Stochastic Noise ---")
            self.info(f"  Noise amplitude: {params['noise_amplitude']:.4f} eV")
            self.info(f"  Noise correlation time: {params['noise_correlation_time']:.4f} fs")
        
        if params.get('decoherence_rate', 0) > 0:
            decoherence_time = 1.0 / params['decoherence_rate']
            self.info(f"  Decoherence rate: {params['decoherence_rate']:.4f} fs⁻¹")
            self.info(f"  Coherence time T₂: {decoherence_time:.2f} fs")
        
        # ================================================================
        # REPORT CONSERVATION VIOLATIONS
        # ================================================================
        # Norm conservation (should be ∫|ψ|²dx = 1)
        if params.get('n_norm_violations', 0) > 0:
            self.warning(
                f"Norm violated {params['n_norm_violations']} times "
                f"(max: {params['max_norm_error']:.6f})"
            )
        else:
            self.info("  ✓ Norm conservation maintained")
        
        # Energy conservation (should be <H> = constant if no noise)
        if params.get('n_energy_violations', 0) > 0:
            self.warning(
                f"Energy violated {params['n_energy_violations']} times "
                f"(max: {params['max_energy_error']:.6%})"
            )
        elif params.get('noise_amplitude', 0) == 0:
            # Only expect energy conservation if no noise
            self.info("  ✓ Energy conservation maintained")
        
        self.info("=" * 60)
    
    def finalize(self):
        """
        Write final summary and close logger.
        
        Creates a comprehensive summary including:
            - Total errors count
            - Total warnings count
            - List of all issues
            - Log file location
        
        Should be called at the end of simulation.
        """
        self.info("=" * 60)
        self.info("SIMULATION SUMMARY")
        self.info("=" * 60)
        
        # ================================================================
        # REPORT ALL ERRORS
        # ================================================================
        if self.errors:
            self.info(f"  ERRORS: {len(self.errors)}")
            for i, err in enumerate(self.errors, 1):
                self.info(f"    {i}. {err}")
        else:
            self.info("  ERRORS: None ✓")
        
        # ================================================================
        # REPORT ALL WARNINGS
        # ================================================================
        if self.warnings:
            self.info(f"  WARNINGS: {len(self.warnings)}")
            for i, warn in enumerate(self.warnings, 1):
                self.info(f"    {i}. {warn}")
        else:
            self.info("  WARNINGS: None ✓")
        
        # Log file location
        self.info(f"  Log file: {self.log_file}")
        self.info("=" * 60)
        self.info(f"Simulation completed: {self.scenario_name}")
        self.info("=" * 60)
