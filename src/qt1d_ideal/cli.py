#!/usr/bin/env python
"""
Command Line Interface for 1D Quantum Tunneling Solver

Provides user-friendly command-line interface for running quantum tunneling
simulations with predefined test cases or custom configurations.

Usage Examples:
    qt1d-simulate case1              # Run rectangular barrier case
    qt1d-simulate case2 --cores 8    # Use 8 CPU cores
    qt1d-simulate --all              # Run all 4 test cases
    qt1d-simulate -c custom.txt      # Use custom config file
    qt1d-simulate case1 -q           # Quiet mode (minimal output)

Features:
    - Pre-configured test cases (4 scenarios)
    - Custom configuration file support
    - Multi-core processing
    - Verbose/quiet output modes
    - Comprehensive error handling
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

from .core.solver import QuantumTunneling1D
from .core.initial_conditions import GaussianWavePacket
from .io.config_manager import ConfigManager
from .io.data_handler import DataHandler
from .visualization.animator import Animator
from .utils.logger import SimulationLogger
from .utils.timer import Timer


def print_header():
    """
    Print ASCII art header with version information.
    """
    print("\n" + "=" * 60)
    print(" " * 10 + "1D QUANTUM TUNNELING SOLVER")
    print(" " * 15 + "WITH STOCHASTIC NOISE")
    print(" " * 20 + "Version 0.0.1")
    print("=" * 60 + "\n")


def create_potential(config: dict, solver: QuantumTunneling1D) -> np.ndarray:
    """
    Create potential barrier based on configuration.
    
    Supports:
        - rectangular: Sharp-edged barrier
        - gaussian: Smooth barrier
        - double_barrier: Two barriers (resonant tunneling)
        - triple_barrier: Three barriers (complex interference)
    
    Args:
        config: Configuration dictionary
        solver: QuantumTunneling1D instance (provides spatial grid)
    
    Returns:
        Potential array V(x) [eV]
    """
    barrier_type = config.get('barrier_type', 'rectangular')
    
    if barrier_type == 'rectangular':
        return solver.rectangular_barrier(
            height=config.get('barrier_height', 2.0),
            width=config.get('barrier_width', 2.0),
            center=config.get('barrier_center', 0.0))
    
    elif barrier_type == 'gaussian':
        return solver.gaussian_barrier(
            height=config.get('barrier_height', 2.0),
            width=config.get('barrier_width', 2.0),
            center=config.get('barrier_center', 0.0))
    
    elif barrier_type == 'double_barrier':
        return solver.double_barrier(
            height=config.get('barrier_height', 2.0),
            width=config.get('barrier_width', 1.0),
            separation=config.get('barrier_separation', 2.0))
    
    elif barrier_type == 'triple_barrier':
        return solver.triple_barrier(
            height=config.get('barrier_height', 2.0),
            width=config.get('barrier_width', 1.0),
            separation=config.get('barrier_separation', 2.5))
    
    else:
        # Default to rectangular if unknown type
        print(f"  WARNING: Unknown barrier type '{barrier_type}', using rectangular")
        return solver.rectangular_barrier(2.0, 2.0)


def run_scenario(config: dict, output_dir: str = "outputs", 
                verbose: bool = True, n_cores: int = None):
    """
    Run a complete quantum tunneling simulation scenario.
    
    Workflow:
        1. Initialize solver and logger
        2. Set up initial wavefunction and potential
        3. Run time evolution
        4. Save results to NetCDF
        5. Create animation (optional)
        6. Log timing and results
    
    Args:
        config: Configuration dictionary with all parameters
        output_dir: Directory for saving outputs
        verbose: Print progress information
        n_cores: Number of CPU cores (None = use all)
    
    Raises:
        Exception: If simulation fails (logged and re-raised)
    """
    scenario_name = config.get('scenario_name', 'simulation')
    
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"SCENARIO: {scenario_name}")
        print(f"{'=' * 60}")
    
    # ====================================================================
    # INITIALIZE LOGGER AND TIMER
    # ====================================================================
    logger = SimulationLogger(
        scenario_name.lower().replace(' ', '_').replace('-', '_'),
        "logs", 
        verbose
    )
    timer = Timer()
    timer.start("total")
    
    try:
        # Log all configuration parameters
        logger.log_parameters(config)
        
        # ================================================================
        # PHASE 1: INITIALIZE SOLVER
        # ================================================================
        with timer.time_section("solver_init"):
            if verbose:
                print("\n[1/5] Initializing solver...")
            
            solver = QuantumTunneling1D(
                nx=config.get('nx', 2048),
                x_min=config.get('x_min', -10.0),
                x_max=config.get('x_max', 10.0),
                adaptive_dt=config.get('adaptive_dt', True),
                verbose=verbose,
                logger=logger,
                n_cores=n_cores
            )
        
        # ================================================================
        # PHASE 2: SET UP INITIAL CONDITIONS
        # ================================================================
        with timer.time_section("initial_condition"):
            if verbose:
                print("\n[2/5] Setting up initial state...")
            
            # Create Gaussian wave packet
            wf = GaussianWavePacket(
                x0=config.get('x0', -5.0),     # Initial position
                k0=config.get('k0', 5.0),       # Initial momentum
                sigma=config.get('sigma', 0.5)  # Wavepacket width
            )
            psi0 = wf(solver.x)
            
            # Create potential barrier
            V = create_potential(config, solver)
        
        # ================================================================
        # PHASE 3: RUN QUANTUM DYNAMICS SIMULATION
        # ================================================================
        with timer.time_section("simulation"):
            if verbose:
                print("\n[3/5] Running quantum dynamics...")
            
            result = solver.solve(
                psi0=psi0,
                V=V,
                t_final=config.get('t_final', 5.0),
                dt_min=config.get('dt_min', 1e-4),
                dt_max=config.get('dt_max', 1e-2),
                n_snapshots=config.get('n_snapshots', 200),
                particle_mass=config.get('particle_mass', 1.0),
                show_progress=verbose,
                # Stochastic noise parameters
                noise_amplitude=config.get('noise_amplitude', 0.0),
                noise_correlation_time=config.get('noise_correlation_time', 0.1),
                decoherence_rate=config.get('decoherence_rate', 0.0)
            )
            
            # Log results with validation
            logger.log_results(result)
            
            # Print key results to console
            if verbose:
                T = result['transmission_coefficient']
                R = result['reflection_coefficient']
                print(f"      T = {T:.4f} ({T*100:.2f}%)")
                print(f"      R = {R:.4f} ({R*100:.2f}%)")
                print(f"      T + R = {T+R:.4f}")
        
        # ================================================================
        # PHASE 4: SAVE DATA TO NETCDF (optional)
        # ================================================================
        if config.get('save_netcdf', True):
            with timer.time_section("save_netcdf"):
                if verbose:
                    print("\n[4/5] Saving NetCDF data...")
                
                # Create filename from scenario name
                filename = f"{scenario_name.lower().replace(' ', '_').replace('-', '_')}.nc"
                
                # Save to NetCDF format
                DataHandler.save_netcdf(filename, result, config, output_dir)
                
                if verbose:
                    print(f"      Saved: {output_dir}/{filename}")
        
        # ================================================================
        # PHASE 5: CREATE ANIMATION (optional)
        # ================================================================
        if config.get('save_animation', True):
            with timer.time_section("animation"):
                if verbose:
                    print("\n[5/5] Creating animation...")
                
                # Create filename from scenario name
                filename = f"{scenario_name.lower().replace(' ', '_').replace('-', '_')}.gif"
                
                # Generate animated GIF
                Animator.create_gif(
                    result, 
                    filename, 
                    output_dir, 
                    scenario_name,
                    config.get('fps', 30),
                    config.get('dpi', 100)
                )
                
                if verbose:
                    print(f"      Saved: {output_dir}/{filename}")
        
        # ================================================================
        # FINALIZE: LOG TIMING AND COMPLETE
        # ================================================================
        timer.stop("total")
        logger.log_timing(timer.get_times())
        
        if verbose:
            print(f"\n{'=' * 60}")
            print("✓ SIMULATION COMPLETED SUCCESSFULLY")
            print(f"  Total time: {timer.get_times()['total']:.2f} s")
            
            # Show number of warnings/errors
            if logger.warnings:
                print(f"  Warnings: {len(logger.warnings)}")
            if logger.errors:
                print(f"  Errors: {len(logger.errors)}")
            
            print(f"{'=' * 60}\n")
    
    except Exception as e:
        # ================================================================
        # ERROR HANDLING
        # ================================================================
        # Log error
        logger.error(f"Simulation failed: {str(e)}")
        
        # Print error box
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"✗ SIMULATION FAILED")
            print(f"  Error: {str(e)}")
            print(f"{'=' * 60}\n")
        
        # Re-raise exception for debugging
        raise
    
    finally:
        # Always finalize logger (writes summary)
        logger.finalize()


def main():
    """
    Main entry point for command-line interface.
    
    Parses arguments and dispatches to appropriate scenario runner.
    """
    # ====================================================================
    # ARGUMENT PARSER SETUP
    # ====================================================================
    parser = argparse.ArgumentParser(
        description='1D Quantum Tunneling Solver with Stochastic Noise',
        epilog='Example: qt1d-simulate case1 --cores 8'
    )
    
    # Positional argument: which case to run
    parser.add_argument(
        'case', 
        nargs='?',
        choices=['case1', 'case2', 'case3', 'case4'],
        help='Test case to run (1=rectangular, 2=double, 3=triple, 4=gaussian)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--config', '-c', 
        type=str, 
        help='Path to custom configuration file'
    )
    
    parser.add_argument(
        '--all', '-a', 
        action='store_true',
        help='Run all 4 test cases sequentially'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='outputs',
        help='Output directory for results (default: outputs)'
    )
    
    parser.add_argument(
        '--cores',
        type=int,
        default=None,
        help='Number of CPU cores to use (default: all available)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Quiet mode (minimal output)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    verbose = not args.quiet
    
    # ====================================================================
    # PRINT HEADER (unless quiet mode)
    # ====================================================================
    if verbose:
        print_header()
    
    # ====================================================================
    # DISPATCH TO APPROPRIATE MODE
    # ====================================================================
    
    # MODE 1: Custom configuration file
    if args.config:
        config = ConfigManager.load(args.config)
        run_scenario(config, args.output_dir, verbose, args.cores)
    
    # MODE 2: Run all test cases
    elif args.all:
        configs_dir = Path(__file__).parent.parent.parent / 'configs'
        
        # Find all case*.txt files
        config_files = sorted(configs_dir.glob('case*.txt'))
        
        if not config_files:
            print("ERROR: No configuration files found in configs/")
            sys.exit(1)
        
        # Run each case sequentially
        for i, cfg_file in enumerate(config_files, 1):
            if verbose:
                print(f"\n[Case {i}/{len(config_files)}] Running {cfg_file.stem}...")
            
            config = ConfigManager.load(str(cfg_file))
            run_scenario(config, args.output_dir, verbose, args.cores)
    
    # MODE 3: Run specific case
    elif args.case:
        # Map case shorthand to full config filename
        case_map = {
            'case1': 'case1_rectangular',
            'case2': 'case2_double',
            'case3': 'case3_triple',
            'case4': 'case4_gaussian'
        }
        
        cfg_name = case_map[args.case]
        configs_dir = Path(__file__).parent.parent.parent / 'configs'
        cfg_file = configs_dir / f'{cfg_name}.txt'
        
        # Check if config file exists
        if cfg_file.exists():
            config = ConfigManager.load(str(cfg_file))
            run_scenario(config, args.output_dir, verbose, args.cores)
        else:
            print(f"ERROR: Configuration file not found: {cfg_file}")
            sys.exit(1)
    
    # MODE 4: No arguments - show help
    else:
        parser.print_help()
        sys.exit(0)


# ====================================================================
# ENTRY POINT
# ====================================================================
if __name__ == '__main__':
    main()
