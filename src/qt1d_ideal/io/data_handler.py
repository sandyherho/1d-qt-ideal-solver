"""
NetCDF Data Handler for Quantum Tunneling Results

Saves essential variables for efficient storage and analysis:
- Coordinates: x, t
- Wavefunction: psi_real, psi_imag (full quantum state)
- Probability density: |ψ|²
- Potential: V(x)
- Key results: T, R, absorbed probability
- Critical parameters

File size: ~3-4 MB per simulation (with compression)
"""

import numpy as np
from netCDF4 import Dataset
from pathlib import Path
from datetime import datetime


class DataHandler:
    """
    NetCDF output handler for quantum tunneling simulations.
    
    Saves wavefunction evolution, potential, and simulation metadata
    in a compact, self-describing NetCDF4 format.
    """
    
    @staticmethod
    def save_netcdf(filename: str, result: dict, metadata: dict, 
                   output_dir: str = "outputs"):
        """
        Save quantum tunneling simulation results to NetCDF file.
        
        Creates a compressed NetCDF4 file with:
            - Full complex wavefunction ψ(x,t) = Re(ψ) + i·Im(ψ)
            - Probability density |ψ(x,t)|²
            - Potential barrier V(x)
            - Transmission/reflection/absorption coefficients
            - Complete simulation parameters
        
        File Structure:
            dimensions:
                x: 2048-4096 (spatial grid)
                t: 200-300 (time snapshots)
            
            variables:
                x(x): Position coordinate [nm]
                t(t): Time coordinate [fs]
                psi_real(t,x): Real part of wavefunction [nm^-0.5]
                psi_imag(t,x): Imaginary part of wavefunction [nm^-0.5]
                probability(t,x): Probability density [nm^-1]
                potential(x): Potential energy [eV]
            
            attributes:
                transmission, reflection, absorbed
                particle_mass, dx, dt_mean
                noise_amplitude, decoherence_rate
                scenario, barrier_type
                created, software, method
        
        Args:
            filename: Output filename (e.g., 'case1_field_emission.nc')
            result: Results dictionary from solver.solve()
            metadata: Configuration dictionary
            output_dir: Output directory (default: 'outputs')
        
        Example:
            >>> result = solver.solve(psi0, V, t_final=10.0)
            >>> config = {'scenario_name': 'Case 1', 'barrier_type': 'rectangular'}
            >>> DataHandler.save_netcdf('case1.nc', result, config)
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / filename
        
        # Open NetCDF file for writing
        with Dataset(filepath, 'w', format='NETCDF4') as nc:
            
            # ============================================================
            # DEFINE DIMENSIONS
            # ============================================================
            nx = len(result['x'])
            nt = len(result['t'])
            
            nc.createDimension('x', nx)  # Spatial dimension
            nc.createDimension('t', nt)  # Temporal dimension
            
            # ============================================================
            # CREATE COORDINATE VARIABLES
            # ============================================================
            # Position coordinate
            nc_x = nc.createVariable('x', 'f4', ('x',), 
                                    zlib=True,      # Enable compression
                                    complevel=4)    # Compression level
            nc_x[:] = result['x']
            nc_x.units = "nm"
            nc_x.long_name = "position"
            nc_x.axis = "X"
            
            # Time coordinate
            nc_t = nc.createVariable('t', 'f4', ('t',),
                                    zlib=True, 
                                    complevel=4)
            nc_t[:] = result['t']
            nc_t.units = "fs"
            nc_t.long_name = "time"
            nc_t.axis = "T"
            
            # ============================================================
            # WAVEFUNCTION COMPONENTS (Real and Imaginary)
            # ============================================================
            # Real part: Re(ψ)
            nc_psi_r = nc.createVariable('psi_real', 'f4', ('t', 'x'),
                                        zlib=True, 
                                        complevel=5)  # Higher compression
            nc_psi_r[:] = result['psi'].real
            nc_psi_r.units = "nm^-0.5"
            nc_psi_r.long_name = "wavefunction_real_part"
            nc_psi_r.description = "Real part of quantum wavefunction"
            nc_psi_r.standard_name = "real_part_of_wavefunction"
            
            # Imaginary part: Im(ψ)
            nc_psi_i = nc.createVariable('psi_imag', 'f4', ('t', 'x'),
                                        zlib=True, 
                                        complevel=5)
            nc_psi_i[:] = result['psi'].imag
            nc_psi_i.units = "nm^-0.5"
            nc_psi_i.long_name = "wavefunction_imaginary_part"
            nc_psi_i.description = "Imaginary part of quantum wavefunction"
            nc_psi_i.standard_name = "imaginary_part_of_wavefunction"
            
            # ============================================================
            # PROBABILITY DENSITY
            # ============================================================
            nc_prob = nc.createVariable('probability', 'f4', ('t', 'x'),
                                       zlib=True, 
                                       complevel=6)  # Highest compression
            nc_prob[:] = result['probability']
            nc_prob.units = "nm^-1"
            nc_prob.long_name = "probability_density"
            nc_prob.description = "Probability density |psi|^2"
            nc_prob.standard_name = "probability_density"
            nc_prob.normalization = "Integral over x equals 1"
            
            # ============================================================
            # POTENTIAL ENERGY
            # ============================================================
            nc_V = nc.createVariable('potential', 'f4', ('x',),
                                    zlib=True, 
                                    complevel=4)
            nc_V[:] = result['potential']
            nc_V.units = "eV"
            nc_V.long_name = "potential_energy"
            nc_V.description = "Potential barrier profile"
            nc_V.standard_name = "potential_energy"
            
            # ============================================================
            # GLOBAL ATTRIBUTES: RESULTS
            # ============================================================
            nc.transmission = float(result['transmission_coefficient'])
            nc.transmission_units = "dimensionless"
            nc.transmission_description = "Transmission coefficient (0-1)"
            
            nc.reflection = float(result['reflection_coefficient'])
            nc.reflection_units = "dimensionless"
            nc.reflection_description = "Reflection coefficient (0-1)"
            
            nc.absorbed = float(result.get('absorbed_probability', 0.0))
            nc.absorbed_units = "dimensionless"
            nc.absorbed_description = "Probability absorbed at boundaries (0-1)"
            
            # ============================================================
            # GLOBAL ATTRIBUTES: SCENARIO INFORMATION
            # ============================================================
            nc.scenario = metadata.get('scenario_name', 'unknown')
            nc.barrier_type = metadata.get('barrier_type', 'unknown')
            
            # ============================================================
            # GLOBAL ATTRIBUTES: SIMULATION PARAMETERS
            # ============================================================
            params = result['params']
            
            # Particle properties
            nc.particle_mass = float(params['particle_mass'])
            nc.particle_mass_units = "electron_mass"
            nc.particle_mass_description = "Particle mass in units of m_e"
            
            # Grid parameters
            nc.nx = int(params['nx'])
            nc.dx = float(params['dx'])
            nc.dx_units = "nm"
            
            # Time stepping
            nc.n_steps = int(params['n_steps'])
            nc.dt_mean = float(params['dt_mean'])
            nc.dt_mean_units = "fs"
            nc.adaptive_dt = int(params.get('adaptive', False))
            
            # ============================================================
            # GLOBAL ATTRIBUTES: ENVIRONMENT PARAMETERS
            # ============================================================
            nc.noise_amplitude = float(params.get('noise_amplitude', 0.0))
            nc.noise_amplitude_units = "eV"
            nc.noise_amplitude_description = "Stochastic potential noise strength"
            
            nc.noise_correlation_time = float(params.get('noise_correlation_time', 0.0))
            nc.noise_correlation_time_units = "fs"
            
            nc.decoherence_rate = float(params.get('decoherence_rate', 0.0))
            nc.decoherence_rate_units = "fs^-1"
            nc.decoherence_rate_description = "Environmental decoherence rate gamma"
            
            # Coherence time (if decoherence present)
            if params.get('decoherence_rate', 0.0) > 0:
                T2 = 1.0 / params['decoherence_rate']
                nc.coherence_time_T2 = float(T2)
                nc.coherence_time_T2_units = "fs"
                nc.coherence_time_T2_description = "Decoherence time T2 = 1/gamma"
            
            # ============================================================
            # GLOBAL ATTRIBUTES: DIAGNOSTICS
            # ============================================================
            nc.max_norm_error = float(params.get('max_norm_error', 0.0))
            nc.max_norm_error_description = "Maximum deviation from |psi|^2 = 1"
            
            nc.max_energy_error = float(params.get('max_energy_error', 0.0))
            nc.max_energy_error_description = "Maximum relative energy deviation"
            
            nc.n_norm_violations = int(params.get('n_norm_violations', 0))
            nc.n_energy_violations = int(params.get('n_energy_violations', 0))
            
            # ============================================================
            # GLOBAL ATTRIBUTES: PROVENANCE
            # ============================================================
            nc.created = datetime.now().isoformat()
            nc.software = "1d-qt-ideal-solver"
            nc.version = "0.0.7"
            nc.method = "split_operator_with_absorbing_boundaries"
            nc.method_description = "Split-operator Fourier method with absorbing boundary conditions"
            
            # Physical constants (for reference)
            nc.hbar = 1.0
            nc.hbar_description = "Natural units: hbar = 1"
            
            # Conventions
            nc.Conventions = "CF-1.8"
            nc.title = f"Quantum Tunneling Simulation: {metadata.get('scenario_name', 'unknown')}"
            nc.institution = "Samudera Sains Teknologi (SST) Ltd."
            nc.history = f"Created {datetime.now().isoformat()}."
            
        # File automatically closed when exiting 'with' block
        # All data flushed to disk with compression
