#!/usr/bin/env python3
"""
Command Line Interface for Idealized 1D Quantum Tunneling Solver

Provides easy access to predefined test cases and custom simulations.
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from .core.solver import QuantumTunneling1D
from .core.initial_conditions import GaussianWavePacket
from .io.config_manager import ConfigManager
from .io.data_handler import DataHandler
from .visualization.animator import Animator
from .utils.logger import SimulationLogger
from .utils.timer import Timer


def print_header():
    """Print header with disclaimer about idealized nature."""
    print("\n" + "="*70)
    print(" "*10 + "IDEALIZED 1D QUANTUM TUNNELING SOLVER")
    print(" "*25 + "Version 0.0.1")
    print("="*70)
    print("\n⚠️  IMPORTANT: These are IDEALIZED simulations")
    print("   Models are simplified for educational purposes")
    print("="*70)
    print("\nAuthors:")
    print("  • Siti N. Kaban")
    print("  • Sandy H. S. Herho")
    print("  • Sonny Prayogo")
    print("  • Iwan P. Anwar")
    print("\nLicense: MIT License")
    print("Repository: https://github.com/sandyherho/1d-qt-ideal-solver")
    print("="*70 + "\n")


def create_potential(config: Dict[str, Any], 
                    solver: QuantumTunneling1D) -> np.ndarray:
    """Create potential based on configuration."""
    barrier_type = config.get('barrier_type', 'rectangular')
    
    if barrier_type == 'rectangular':
        V = solver.rectangular_barrier(
            height=config.get('barrier_height', 2.0),
            width=config.get('barrier_width', 2.0),
            center=config.get('barrier_center', 0.0)
        )
    elif barrier_type == 'gaussian':
        V = solver.gaussian_barrier(
            height=config.get('barrier_height', 2.0),
            width=config.get('barrier_width', 2.0),
            center=config.get('barrier_center', 0.0)
        )
    elif barrier_type == 'step':
        V = solver.step_potential(
            step_height=config.get('step_height', 1.5),
            step_position=config.get('step_position', 0.0)
        )
    elif barrier_type == 'double_barrier':
        V = solver.double_barrier(
            height=config.get('barrier_height', 2.0),
            width=config.get('barrier_width', 1.0),
            separation=config.get('barrier_separation', 2.0)
        )
    elif barrier_type == 'triple_barrier':
        V = solver.triple_barrier(
            height=config.get('barrier_height', 2.0),
            width=config.get('barrier_width', 1.0),
            separation=config.get('barrier_separation', 2.5)
        )
    elif barrier_type == 'periodic':
        V = solver.periodic_potential(
            height=config.get('barrier_height', 1.5),
            n_periods=config.get('n_periods', 8),
            period_length=config.get('period_length', 4.0)
        )
    else:
        V = solver.rectangular_barrier(
            height=config.get('barrier_height', 2.0),
            width=config.get('barrier_width', 2.0)
        )
    
    return V


def run_scenario(config: Dict[str, Any], output_dir: str = "outputs",
                verbose: bool = True, n_cores: int = None) -> None:
    """Run single simulation scenario."""
    scenario_name = config.get('scenario_name', 'simulation')
    
    if n_cores is None:
        n_cores = config.get('n_cores', None)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"SCENARIO: {scenario_name} (IDEALIZED)")
        print(f"{'='*70}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    logger = SimulationLogger(
        scenario_name=scenario_name.lower().replace(' ', '_'),
        log_dir="logs",
        verbose=verbose
    )
    
    timer = Timer()
    timer.start("total")
    
    try:
        logger.log_parameters(config)
        
        if verbose:
            print("\nConfiguration:")
            print("-" * 40)
            for key, value in config.items():
                if key != 'scenario_name':
                    if isinstance(value, float):
                        print(f"  {key:20s}: {value:.3f}")
                    else:
                        print(f"  {key:20s}: {value}")
            print("-" * 40)
        
        # Initialize solver
        with timer.time_section("solver_init"):
            if verbose:
                print("\n[1/5] Initializing Solver...")
            solver = QuantumTunneling1D(
                nx=config.get('nx', 2048),
                x_min=config.get('x_min', -10.0),
                x_max=config.get('x_max', 10.0),
                adaptive_dt=config.get('adaptive_dt', True),
                verbose=verbose,
                logger=logger,
                n_cores=n_cores
            )
        
        # Initial wavefunction
        with timer.time_section("initial_condition"):
            if verbose:
                print("\n[2/5] Setting Initial State...")
            wf = GaussianWavePacket(
                x0=config.get('x0', -5.0),
                k0=config.get('k0', 5.0),
                sigma=config.get('sigma', 0.5)
            )
            psi0 = wf(solver.x)
            V = create_potential(config, solver)
            
            if verbose:
                print(f"      ✓ Wave packet: x0={config.get('x0', -5.0):.2f} nm, "
                      f"k0={config.get('k0', 5.0):.2f} 1/nm")
                print(f"      ✓ Barrier: {config.get('barrier_type', 'rectangular')}")
        
        # Run simulation
        with timer.time_section("simulation"):
            if verbose:
                print("\n[3/5] Running Simulation...")
                print(f"      Target time: {config.get('t_final', 5.0):.3f} fs")
            
            result = solver.solve(
                psi0=psi0,
                V=V,
                t_final=config.get('t_final', 5.0),
                dt_min=config.get('dt_min', 1e-4),
                dt_max=config.get('dt_max', 1e-2),
                n_snapshots=config.get('n_snapshots', 200),
                particle_mass=config.get('particle_mass', 1.0),
                show_progress=verbose
            )
            
            if verbose:
                print(f"      ✓ Transmission: {result['transmission_coefficient']:.2%}")
                print(f"      ✓ Reflection: {result['reflection_coefficient']:.2%}")
        
        # Save NetCDF
        if config.get('save_netcdf', True):
            with timer.time_section("save_netcdf"):
                if verbose:
                    print("\n[4/5] Saving Data...")
                filename = f"{scenario_name.lower().replace(' ', '_')}.nc"
                DataHandler.save_netcdf(
                    filename=filename,
                    result=result,
                    metadata=config,
                    output_dir=output_dir
                )
                if verbose:
                    filepath = Path(output_dir) / filename
                    size_mb = filepath.stat().st_size / (1024*1024)
                    print(f"      ✓ NetCDF: {filename} ({size_mb:.2f} MB)")
        
        # Create animation
        if config.get('save_animation', True):
            with timer.time_section("animation"):
                if verbose:
                    print("\n[5/5] Creating Animation...")
                filename = f"{scenario_name.lower().replace(' ', '_')}.gif"
                Animator.create_gif(
                    result=result,
                    filename=filename,
                    output_dir=output_dir,
                    title=scenario_name,
                    fps=config.get('fps', 30),
                    dpi=config.get('dpi', 100)
                )
                if verbose:
                    filepath = Path(output_dir) / filename
                    size_mb = filepath.stat().st_size / (1024*1024)
                    print(f"      ✓ Animation: {filename} ({size_mb:.2f} MB)")
        
        timer.stop("total")
        logger.log_timing(timer.get_times())
        
        if verbose:
            print("\n" + "="*70)
            print("SIMULATION COMPLETED")
            print("-"*40)
            times = timer.get_times()
            for section, time_val in times.items():
                if section != 'total':
                    print(f"  {section:20s}: {time_val:6.2f} s")
            print("-"*40)
            print(f"  {'TOTAL':20s}: {times['total']:6.2f} s")
            print("="*70)
            print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        if verbose:
            print(f"\n{'='*70}")
            print("ERROR")
            print(f"{'-'*40}\n  {str(e)}\n{'='*70}\n")
        raise
    finally:
        logger.finalize()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Idealized 1D Quantum Tunneling Solver',
        epilog='https://github.com/yourusername/1d-qt-ideal-solver'
    )
    
    parser.add_argument('scenario', nargs='?',
                       choices=['rect-barrier', 'gaussian-barrier', 
                               'step-potential', 'double-barrier',
                               'triple-barrier', 'periodic-potential'],
                       help='Predefined scenario')
    
    parser.add_argument('--config', '-c', type=str,
                       help='Configuration file path')
    
    parser.add_argument('--all', '-a', action='store_true',
                       help='Run all 6 scenarios')
    
    parser.add_argument('--output-dir', '-o', type=str, default='outputs',
                       help='Output directory')
    
    parser.add_argument('--cores', type=int, default=None,
                       help='CPU cores (default: all)')
    
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Quiet mode')
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    if verbose:
        print_header()
    
    if args.config:
        config = ConfigManager.load(args.config)
        run_scenario(config, args.output_dir, verbose, args.cores)
    elif args.all:
        configs_dir = Path(__file__).parent.parent.parent / 'configs'
        scenarios = sorted(configs_dir.glob('*.txt'))
        if verbose:
            print(f"Running all {len(scenarios)} scenarios\n")
        for i, cfg_file in enumerate(scenarios, 1):
            if verbose:
                print(f"\n[{i}/{len(scenarios)}] {cfg_file.name}")
            config = ConfigManager.load(str(cfg_file))
            run_scenario(config, args.output_dir, verbose, args.cores)
    elif args.scenario:
        # Map CLI names to config files
        scenario_map = {
            'rect-barrier': 'rect_barrier',
            'gaussian-barrier': 'gaussian_barrier',
            'step-potential': 'step_potential',
            'double-barrier': 'double_barrier',
            'triple-barrier': 'triple_barrier',
            'periodic-potential': 'periodic_potential'
        }
        cfg_name = scenario_map.get(args.scenario, args.scenario)
        configs_dir = Path(__file__).parent.parent.parent / 'configs'
        cfg_file = configs_dir / f'{cfg_name}.txt'
        if cfg_file.exists():
            config = ConfigManager.load(str(cfg_file))
            run_scenario(config, args.output_dir, verbose, args.cores)
        else:
            print(f"Error: Config not found: {cfg_file}")
            sys.exit(1)
    else:
        parser.print_help()
        if verbose:
            print("\nExamples:")
            print("  qt1d-simulate rect-barrier")
            print("  qt1d-simulate rect-barrier --cores 8")
            print("  qt1d-simulate --all")
        sys.exit(0)
    
    if verbose:
        print("\nThank you for using 1D QT Ideal Solver!")


if __name__ == '__main__':
    main()
