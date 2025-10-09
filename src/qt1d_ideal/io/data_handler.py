"""
NetCDF Data Handler for Quantum Tunneling Results

Handles saving simulation results to NetCDF4 format for long-term storage,
analysis, and data sharing. NetCDF is a self-describing, portable, scalable
binary format widely used in scientific computing.

NetCDF Structure:
    Dimensions:
        - x: Spatial grid points
        - t: Time snapshots
    
    Variables:
        - x(x): Position array [nm]
        - t(t): Time array [fs]
        - psi_real(t,x): Real part of wavefunction [nm^-0.5]
        - psi_imag(t,x): Imaginary part of wavefunction [nm^-0.5]
        - probability(t,x): |ψ|² probability density [nm^-1]
        - potential(x): Potential energy [eV]
    
    Global Attributes:
        - Physical results (T, R)
        - Simulation parameters
        - Metadata (timestamp, software version)
        - Stochastic noise parameters
        - Error diagnostics

Advantages of NetCDF:
    - Self-describing (contains metadata)
    - Efficient compression
    - Random access
    - Platform independent
    - Compatible with Python (xarray), MATLAB, Julia, R
"""

import numpy as np
from netCDF4 import Dataset
from pathlib import Path
from datetime import datetime


class DataHandler:
    """
    Static methods for saving quantum tunneling results to NetCDF format.
    """
    
    @staticmethod
    def save_netcdf(filename: str, result: dict, metadata: dict, 
                   output_dir: str = "outputs"):
        """
        Save quantum tunneling simulation results to NetCDF file.
        
        Creates a CF-compliant NetCDF4 file with:
            - All wavefunction evolution data
            - Potential barrier information
            - Physical observables (T, R)
            - Complete parameter set
            - Error diagnostics
            - Timestamp and provenance
        
        Args:
            filename: Output filename (e.g., 'case1.nc')
            result: Results dictionary from solver.solve()
            metadata: Configuration dictionary (scenario_name, barrier_type, etc.)
            output_dir: Directory for saving file (created if needed)
        
        File Structure:
            case1.nc
            ├── dimensions
            │   ├── x: 2048
            │   └── t: 200
            ├── variables
            │   ├── x(x) [nm]
            │   ├── t(t) [fs]
            │   ├── psi_real(t,x)
            │   ├── psi_imag(t,x)
            │   ├── probability(t,x) [nm^-1]
            │   └── potential(x) [eV]
            └── attributes (global)
                ├── transmission
                ├── reflection
                ├── noise_amplitude
                ├── max_norm_error
                └── ... (many more)
        """
        # ================================================================
        # CREATE OUTPUT DIRECTORY
        # ================================================================
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / filename
        
        # ================================================================
        # CREATE NETCDF FILE
        # ================================================================
        # 'w' = write mode (overwrites if exists)
        # format='NETCDF4' enables compression and groups
        with Dataset(filepath, 'w', format='NETCDF4') as nc:
            
            # ============================================================
            # DEFINE DIMENSIONS
            # ============================================================
            # Dimensions define the size of coordinate axes
            nx = len(result['x'])
            nt = len(result['t'])
            
            nc.createDimension('x', nx)  # Spatial dimension
            nc.createDimension('t', nt)  # Temporal dimension
            
            # ============================================================
            # CREATE COORDINATE VARIABLES
            # ============================================================
            # These define the actual coordinate values
            
            # Spatial coordinate [nm]
            nc_x = nc.createVariable('x', 'f4', ('x',))
            nc_x[:] = np.round(result['x'], 4)  # Round to save space
            nc_x.units = "nm"
            nc_x.long_name = "position"
            nc_x.axis = "X"
            
            # Temporal coordinate [fs]
            nc_t = nc.createVariable('t', 'f4', ('t',))
            nc_t[:] = np.round(result['t'], 4)
            nc_t.units = "fs"
            nc_t.long_name = "time"
            nc_t.axis = "T"
            
            # ============================================================
            # CREATE DATA VARIABLES
            # ============================================================
            # Store wavefunction as real and imaginary parts
            # (NetCDF4 doesn't support complex numbers directly)
            
            # Real part of ψ(t,x)
            nc_psi_r = nc.createVariable('psi_real', 'f4', ('t', 'x'))
            nc_psi_r[:] = np.round(result['psi'].real, 6)
            nc_psi_r.units = "nm^-0.5"
            nc_psi_r.long_name = "wavefunction_real_part"
            nc_psi_r.description = "Real part of quantum wavefunction"
            
            # Imaginary part of ψ(t,x)
            nc_psi_i = nc.createVariable('psi_imag', 'f4', ('t', 'x'))
            nc_psi_i[:] = np.round(result['psi'].imag, 6)
            nc_psi_i.units = "nm^-0.5"
            nc_psi_i.long_name = "wavefunction_imaginary_part"
            nc_psi_i.description = "Imaginary part of quantum wavefunction"
            
            # Probability density |ψ(t,x)|²
            nc_prob = nc.createVariable('probability', 'f4', ('t', 'x'))
            nc_prob[:] = np.round(result['probability'], 6)
            nc_prob.units = "nm^-1"
            nc_prob.long_name = "probability_density"
            nc_prob.description = "Probability density |psi|^2"
            nc_prob.normalization = "Integral over x equals 1"
            
            # Potential energy V(x)
            nc_V = nc.createVariable('potential', 'f4', ('x',))
            nc_V[:] = np.round(result['potential'], 4)
            nc_V.units = "eV"
            nc_V.long_name = "potential_energy"
            nc_V.description = "Potential barrier profile"
            
            # ============================================================
            # GLOBAL ATTRIBUTES: METADATA
            # ============================================================
            # General information
            nc.description = "1D Quantum Tunneling Simulation with Stochastic Noise"
            nc.created = datetime.now().isoformat()
            nc.software = "1d-qt-ideal-solver v0.0.1"
            nc.author = "Siti N. Kaban, Sandy H. S. Herho, Sonny Prayogo, Iwan P. Anwar"
            
            # Physical results (key observables)
            nc.transmission = float(result['transmission_coefficient'])
            nc.reflection = float(result['reflection_coefficient'])
            nc.transmission_description = "Transmission coefficient (dimensionless)"
            nc.reflection_description = "Reflection coefficient (dimensionless)"
            
            # Scenario information
            if 'scenario_name' in metadata:
                nc.scenario = metadata['scenario_name']
            if 'barrier_type' in metadata:
                nc.barrier_type = metadata['barrier_type']
            
            # ============================================================
            # GLOBAL ATTRIBUTES: SIMULATION PARAMETERS
            # ============================================================
            params = result['params']
            
            # Particle properties
            nc.particle_mass = float(params['particle_mass'])
            nc.particle_mass_units = "electron_mass"
            
            # Time stepping
            nc.adaptive_dt = int(params.get('adaptive', False))
            if params.get('adaptive'):
                nc.dt_initial = float(params['dt_initial'])
                nc.dt_final = float(params['dt_final'])
                nc.dt_mean = float(params['dt_mean'])
                nc.dt_units = "fs"
            nc.n_steps = int(params['n_steps'])
            
            # Grid parameters
            nc.nx = int(params['nx'])
            nc.dx = float(params['dx'])
            nc.dx_units = "nm"
            
            # ============================================================
            # GLOBAL ATTRIBUTES: STOCHASTIC NOISE PARAMETERS
            # ============================================================
            # Save noise parameters (even if zero, for reproducibility)
            nc.noise_amplitude = float(params.get('noise_amplitude', 0.0))
            nc.noise_amplitude_units = "eV"
            nc.noise_amplitude_description = "Ornstein-Uhlenbeck potential noise strength"
            
            nc.noise_correlation_time = float(params.get('noise_correlation_time', 0.0))
            nc.noise_correlation_time_units = "fs"
            nc.noise_correlation_time_description = "Noise autocorrelation time (tau)"
            
            nc.decoherence_rate = float(params.get('decoherence_rate', 0.0))
            nc.decoherence_rate_units = "fs^-1"
            nc.decoherence_rate_description = "Environmental decoherence rate (gamma)"
            
            # Calculate coherence time if decoherence present
            if params.get('decoherence_rate', 0.0) > 0:
                T2 = 1.0 / params['decoherence_rate']
                nc.coherence_time_T2 = float(T2)
                nc.coherence_time_T2_units = "fs"
            
            # ============================================================
            # GLOBAL ATTRIBUTES: ERROR DIAGNOSTICS
            # ============================================================
            # Conservation law violations
            nc.max_norm_error = float(params.get('max_norm_error', 0.0))
            nc.max_norm_error_description = "Maximum deviation from norm=1"
            
            nc.max_energy_error = float(params.get('max_energy_error', 0.0))
            nc.max_energy_error_description = "Maximum relative energy deviation"
            
            nc.n_norm_violations = int(params.get('n_norm_violations', 0))
            nc.n_norm_violations_description = "Number of times |psi|^2 deviated >1%"
            
            nc.n_energy_violations = int(params.get('n_energy_violations', 0))
            nc.n_energy_violations_description = "Number of times energy deviated >1%"
            
            # ============================================================
            # ADDITIONAL METADATA
            # ============================================================
            # Physics constants (for reference)
            nc.hbar = 1.0
            nc.hbar_description = "Natural units: ℏ = 1"
            
            # Computational details
            nc.method = "split_operator"
            nc.method_description = "Split-operator Fourier method with adaptive dt"
            
            nc.fft_library = "scipy.fftpack"
            nc.jit_compiler = "numba"
            
            # Data precision
            nc.precision = "float32"
            nc.precision_description = "Single precision sufficient for educational purposes"
            
            # Conventions
            nc.Conventions = "CF-1.8"  # Climate and Forecast metadata conventions
            nc.history = f"Created {datetime.now().isoformat()} by qt1d-simulate"
        
        # File is automatically closed when exiting 'with' block
        # All data has been flushed to disk
