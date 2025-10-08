#!/usr/bin/env python3
"""Command Line Interface for 1D Quantum Tunneling Solver"""

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
    print("\n" + "="*60)
    print(" "*10 + "1D QUANTUM TUNNELING SOLVER")
    print(" "*20 + "Version 0.0.1")
    print("="*60 + "\n")


def create_potential(config, solver):
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
        return solver.rectangular_barrier(2.0, 2.0)


def run_scenario(config, output_dir="outputs", verbose=True, n_cores=None):
    scenario_name = config.get('scenario_name', 'simulation')
    if verbose:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario_name}")
        print(f"{'='*60}")
    
    logger = SimulationLogger(
        scenario_name.lower().replace(' ', '_').replace('-', '_'),
        "logs", verbose)
    timer = Timer()
    timer.start("total")
    
    try:
        logger.log_parameters(config)
        
        with timer.time_section("solver_init"):
            if verbose:
                print("\n[1/5] Initializing...")
            solver = QuantumTunneling1D(
                nx=config.get('nx', 2048),
                x_min=config.get('x_min', -10.0),
                x_max=config.get('x_max', 10.0),
                adaptive_dt=config.get('adaptive_dt', True),
                verbose=verbose, logger=logger, n_cores=n_cores)
        
        with timer.time_section("initial_condition"):
            if verbose:
                print("\n[2/5] Setting initial state...")
            wf = GaussianWavePacket(
                x0=config.get('x0', -5.0),
                k0=config.get('k0', 5.0),
                sigma=config.get('sigma', 0.5))
            psi0 = wf(solver.x)
            V = create_potential(config, solver)
        
        with timer.time_section("simulation"):
            if verbose:
                print("\n[3/5] Running simulation...")
            result = solver.solve(
                psi0=psi0, V=V,
                t_final=config.get('t_final', 5.0),
                dt_min=config.get('dt_min', 1e-4),
                dt_max=config.get('dt_max', 1e-2),
                n_snapshots=config.get('n_snapshots', 200),
                particle_mass=config.get('particle_mass', 1.0),
                show_progress=verbose)
            if verbose:
                print(f"      T: {result['transmission_coefficient']:.2%}, "
                      f"R: {result['reflection_coefficient']:.2%}")
        
        if config.get('save_netcdf', True):
            with timer.time_section("save_netcdf"):
                if verbose:
                    print("\n[4/5] Saving data...")
                filename = f"{scenario_name.lower().replace(' ', '_').replace('-', '_')}.nc"
                DataHandler.save_netcdf(filename, result, config, output_dir)
        
        if config.get('save_animation', True):
            with timer.time_section("animation"):
                if verbose:
                    print("\n[5/5] Creating animation...")
                filename = f"{scenario_name.lower().replace(' ', '_').replace('-', '_')}.gif"
                Animator.create_gif(result, filename, output_dir, scenario_name,
                                   config.get('fps', 30), config.get('dpi', 100))
        
        timer.stop("total")
        logger.log_timing(timer.get_times())
        
        if verbose:
            print(f"\n{'='*60}")
            print("COMPLETED")
            print(f"Total time: {timer.get_times()['total']:.2f} s")
            print(f"{'='*60}\n")
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        if verbose:
            print(f"\nERROR: {str(e)}\n")
        raise
    finally:
        logger.finalize()


def main():
    parser = argparse.ArgumentParser(description='1D Quantum Tunneling Solver')
    parser.add_argument('case', nargs='?',
                       choices=['case1', 'case2', 'case3', 'case4'],
                       help='Test case (1-4)')
    parser.add_argument('--config', '-c', type=str, help='Config file')
    parser.add_argument('--all', '-a', action='store_true', help='Run all cases')
    parser.add_argument('--output-dir', '-o', type=str, default='outputs')
    parser.add_argument('--cores', type=int, default=None)
    parser.add_argument('--quiet', '-q', action='store_true')
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    if verbose:
        print_header()
    
    if args.config:
        config = ConfigManager.load(args.config)
        run_scenario(config, args.output_dir, verbose, args.cores)
    elif args.all:
        configs_dir = Path(__file__).parent.parent.parent / 'configs'
        for i, cfg_file in enumerate(sorted(configs_dir.glob('case*.txt')), 1):
            if verbose:
                print(f"\n[{i}/4] Running {cfg_file.stem}...")
            config = ConfigManager.load(str(cfg_file))
            run_scenario(config, args.output_dir, verbose, args.cores)
    elif args.case:
        case_map = {'case1': 'case1_rectangular', 'case2': 'case2_double',
                    'case3': 'case3_triple', 'case4': 'case4_gaussian'}
        cfg_name = case_map[args.case]
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
        sys.exit(0)


if __name__ == '__main__':
    main()
