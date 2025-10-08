import numpy as np
from netCDF4 import Dataset
from pathlib import Path
from datetime import datetime

class DataHandler:
    @staticmethod
    def save_netcdf(filename, result, metadata, output_dir="outputs"):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / filename
        
        with Dataset(filepath, 'w', format='NETCDF4') as nc:
            nc.createDimension('x', len(result['x']))
            nc.createDimension('t', len(result['t']))
            
            nc_x = nc.createVariable('x', 'f4', ('x',))
            nc_t = nc.createVariable('t', 'f4', ('t',))
            nc_psi_r = nc.createVariable('psi_real', 'f4', ('t', 'x'))
            nc_psi_i = nc.createVariable('psi_imag', 'f4', ('t', 'x'))
            nc_prob = nc.createVariable('probability', 'f4', ('t', 'x'))
            nc_V = nc.createVariable('potential', 'f4', ('x',))
            
            nc_x[:] = np.round(result['x'], 4)
            nc_t[:] = np.round(result['t'], 4)
            nc_psi_r[:] = np.round(result['psi'].real, 6)
            nc_psi_i[:] = np.round(result['psi'].imag, 6)
            nc_prob[:] = np.round(result['probability'], 6)
            nc_V[:] = np.round(result['potential'], 4)
            
            nc_x.units = "nm"
            nc_t.units = "fs"
            nc_V.units = "eV"
            
            nc.description = "1D Quantum Tunneling"
            nc.created = datetime.now().isoformat()
            nc.software = "1d-qt-ideal-solver v0.0.1"
            nc.transmission = float(result['transmission_coefficient'])
            nc.reflection = float(result['reflection_coefficient'])
            
            if 'scenario_name' in metadata:
                nc.scenario = metadata['scenario_name']
            if 'barrier_type' in metadata:
                nc.barrier_type = metadata['barrier_type']
            
            params = result['params']
            nc.particle_mass = float(params['particle_mass'])
            nc.adaptive_dt = int(params.get('adaptive', False))
            if params.get('adaptive'):
                nc.dt_initial = float(params['dt_initial'])
                nc.dt_final = float(params['dt_final'])
                nc.dt_mean = float(params['dt_mean'])
            nc.n_steps = int(params['n_steps'])
