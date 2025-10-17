#!/usr/bin/env python
"""
Command Line Interface for 1D Quantum Tunneling Solver
2 Cases: Rectangular and Gaussian barriers
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
    """Print ASCII art header with version and authors."""
    print("\n" + "=" * 70)
    print(" " * 15 + "1D QUANTUM TUNNELING SOLVER")
    print(" " * 18 + "WITH ABSORBING BOUNDARIES")
    print(" " * 25 + "Version 0.0.9")
    print("=" * 70)
    print("\n  Authors: Sandy H. S. Herho, Siti N. Kaban, Iwan P. Anwar, ")
    print("           Nurjanna J. Trilaksono, and Rusmawan Suwarman")
    print("\n  License: MIT License")
    print("=" * 70 + "\n")


def normalize_scenario_name(scenario_name: str) -> str:
    """
    Convert scenario name to clean filename format.
    
    Examples:
        "Case 1 - Rectangular Barrier" -> "case1_rectangular_barrier"
        "Case 2 - Gaussian Barrier" -> "case2_gaussian_barrier"
    """
    clean = scenario_name.lower()
    clean = clean.replace(' - ', '_')
    clean = clean.replace('-', '_')
    clean = clean.replace(' ', '_')
    
    while '__' in clean:
        clean = clean.replace('__', '_')
    
    if clean.startswith('case_'):
        parts = clean.split('_')
        if len(parts) >= 2 and parts[1].isdigit():
            case_num = parts[1]
            rest = '_'.join(parts[2:])
            clean = f"case{case_num}_{rest}"
    
    clean = clean.rstrip('_')
    
    return clean


def create_potential(config: dict, solver: QuantumTunneling1D) -> np.ndarray:
    """Create potential barrier based on configuration."""
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
    
    else:
        print(f"  WARNING: Unknown barrier type '{barrier_type}', using rectangular")
        return solver.rectangular_barrier(2.0, 2.0)


def run_scenario(config: dict, output_dir: str = "outputs", 
                verbose: bool = True, n_cores: int = None):
    """Run a complete quantum tunneling simulation scenario."""
    scenario_name = config.get('scenario_name', 'simulation')
    clean_name = normalize_scenario_name(scenario_name)
    
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"SCENARIO: {scenario_name}")
        print(f"{'=' * 60}")
    
    logger = SimulationLogger(clean_name, "logs", verbose)
    timer = Timer()
    timer.start("total")
    
    try:
        logger.log_parameters(config)
        
        with timer.time_section("solver_init"):
            if verbose:
                print("\n[1/5] Initializing solver with absorbing boundaries...")
            
            boundary_width = config.get('boundary_width', 2.0)
            boundary_strength = config.get('boundary_strength', 0.1)
            
            solver = QuantumTunneling1D(
                nx=config.get('nx', 2048),
                x_min=config.get('x_min', -10.0),
                x_max=config.get('x_max', 10.0),
                adaptive_dt=config.get('adaptive_dt', True),
                verbose=verbose,
                logger=logger,
                n_cores=n_cores,
                boundary_width=boundary_width,
                boundary_strength=boundary_strength
            )
        
        with timer.time_section("initial_condition"):
            if verbose:
                print("\n[2/5] Setting up initial state...")
            
            wf = GaussianWavePacket(
                x0=config.get('x0', -5.0),
                k0=config.get('k0', 5.0),
                sigma=config.get('sigma', 0.5)
            )
            psi0 = wf(solver.x)
            V = create_potential(config, solver)
        
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
                noise_amplitude=config.get('noise_amplitude', 0.0),
                noise_correlation_time=config.get('noise_correlation_time', 0.1),
                decoherence_rate=config.get('decoherence_rate', 0.0)
            )
            
            logger.log_results(result)
            
            if verbose:
                T = result['transmission_coefficient']
                R = result['reflection_coefficient']
                A = result.get('absorbed_probability', 0.0)
                print(f"      T = {T:.4f} ({T*100:.2f}%)")
                print(f"      R = {R:.4f} ({R*100:.2f}%)")
                if A > 0.001:
                    print(f"      Absorbed = {A:.4f} ({A*100:.2f}%)")
                print(f"      T + R + A = {T+R+A:.4f}")
        
        if config.get('save_netcdf', True):
            with timer.time_section("save_netcdf"):
                if verbose:
                    print("\n[4/5] Saving NetCDF data...")
                
                filename = f"{clean_name}.nc"
                DataHandler.save_netcdf(filename, result, config, output_dir)
                
                if verbose:
                    print(f"      Saved: {output_dir}/{filename}")
        
        if config.get('save_animation', True):
            with timer.time_section("animation"):
                if verbose:
                    print("\n[5/5] Creating animation...")
                
                filename = f"{clean_name}.gif"
                
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
        
        timer.stop("total")
        logger.log_timing(timer.get_times())
        
        if verbose:
            print(f"\n{'=' * 60}")
            print("SIMULATION COMPLETED SUCCESSFULLY")
            print(f"  Total time: {timer.get_times()['total']:.2f} s")
            
            if logger.warnings:
                print(f"  Warnings: {len(logger.warnings)}")
            if logger.errors:
                print(f"  Errors: {len(logger.errors)}")
            
            print(f"{'=' * 60}\n")
    
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"SIMULATION FAILED")
            print(f"  Error: {str(e)}")
            print(f"{'=' * 60}\n")
        
        raise
    
    finally:
        logger.finalize()


def main():
    """Main entry point for command-line interface."""
    parser = argparse.ArgumentParser(
        description='1D Quantum Tunneling Solver with Absorbing Boundaries',
        epilog='Example: qt1d-simulate case1 --cores 8'
    )
    
    parser.add_argument(
        'case', 
        nargs='?',
        choices=['case1', 'case2'],
        help='Test case to run (case1=rectangular, case2=gaussian)'
    )
    
    parser.add_argument(
        '--config', '-c', 
        type=str, 
        help='Path to custom configuration file'
    )
    
    parser.add_argument(
        '--all', '-a', 
        action='store_true',
        help='Run both test cases sequentially'
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
    
    parser.add_argument(
        '--boundary-width',
        type=float,
        default=None,
        help='Override absorbing boundary width in nm (default: 2.0)'
    )
    
    parser.add_argument(
        '--boundary-strength',
        type=float,
        default=None,
        help='Override absorbing boundary strength (default: 0.1)'
    )
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    if verbose:
        print_header()
    
    if args.config:
        config = ConfigManager.load(args.config)
        
        if args.boundary_width is not None:
            config['boundary_width'] = args.boundary_width
        if args.boundary_strength is not None:
            config['boundary_strength'] = args.boundary_strength
        
        run_scenario(config, args.output_dir, verbose, args.cores)
    
    elif args.all:
        configs_dir = Path(__file__).parent.parent.parent / 'configs'
        config_files = sorted(configs_dir.glob('case*.txt'))
        
        if not config_files:
            print("ERROR: No configuration files found in configs/")
            sys.exit(1)
        
        for i, cfg_file in enumerate(config_files, 1):
            if verbose:
                print(f"\n[Case {i}/{len(config_files)}] Running {cfg_file.stem}...")
            
            config = ConfigManager.load(str(cfg_file))
            
            if args.boundary_width is not None:
                config['boundary_width'] = args.boundary_width
            if args.boundary_strength is not None:
                config['boundary_strength'] = args.boundary_strength
            
            run_scenario(config, args.output_dir, verbose, args.cores)
    
    elif args.case:
        case_map = {
            'case1': 'case1_rectangular',
            'case2': 'case2_gaussian'
        }
        
        cfg_name = case_map[args.case]
        configs_dir = Path(__file__).parent.parent.parent / 'configs'
        cfg_file = configs_dir / f'{cfg_name}.txt'
        
        if cfg_file.exists():
            config = ConfigManager.load(str(cfg_file))
            
            if args.boundary_width is not None:
                config['boundary_width'] = args.boundary_width
            if args.boundary_strength is not None:
                config['boundary_strength'] = args.boundary_strength
            
            run_scenario(config, args.output_dir, verbose, args.cores)
        else:
            print(f"ERROR: Configuration file not found: {cfg_file}")
            sys.exit(1)
    
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == '__main__':
    main()
