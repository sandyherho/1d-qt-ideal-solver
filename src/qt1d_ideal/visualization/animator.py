"""Professional animation creation."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
from typing import Dict, Any

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['mathtext.fontset'] = 'stix'


class Animator:
    """Create GIF animations."""
    
    @staticmethod
    def create_gif(result: Dict[str, Any], filename: str,
                  output_dir: str = "outputs", title: str = "Quantum Tunneling",
                  fps: int = 30, dpi: int = 100) -> None:
        """Create animation showing wavefunction evolution."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / filename
        
        x = result['x']
        t = result['t']
        psi = result['psi']
        prob = result['probability']
        V = result['potential']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        plt.subplots_adjust(hspace=0.3, top=0.92, bottom=0.08)
        
        psi_max = np.max(np.abs(psi))
        prob_max = np.max(prob)
        V_max = np.max(V)
        V_scale = 0.3 * psi_max / (V_max + 1e-10)
        
        def init():
            return []
        
        def animate(frame):
            ax1.clear()
            ax2.clear()
            
            # Wavefunction
            ax1.plot(x, psi[frame].real, 'b-', lw=2, label=r'Re($\psi$)')
            ax1.plot(x, psi[frame].imag, 'r-', lw=2, label=r'Im($\psi$)')
            ax1.axhline(0, color='k', lw=0.5, alpha=0.3)
            ax1.fill_between(x, 0, V * V_scale, alpha=0.2, color='gray', 
                            label='Potential')
            ax1.set_ylabel(r'$\psi$ [nm$^{-1/2}$]', fontsize=12)
            ax1.set_ylim(-psi_max*1.1, psi_max*1.1)
            ax1.set_title(f'{title} - Wavefunction (t={t[frame]:.3f} fs)', 
                         fontsize=13)
            ax1.legend(loc='upper right', fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # Probability
            ax2.fill_between(x, 0, prob[frame], alpha=0.6, color='green',
                            label=r'$|\psi|^2$')
            ax2.plot(x, prob[frame], 'g-', lw=1.5)
            V_prob = 0.5 * prob_max / (V_max + 1e-10)
            ax2.fill_between(x, 0, V * V_prob, alpha=0.2, color='gray',
                            label='Potential')
            ax2.set_xlabel(r'Position $x$ [nm]', fontsize=12)
            ax2.set_ylabel(r'$|\psi|^2$ [nm$^{-1}$]', fontsize=12)
            ax2.set_ylim(0, prob_max*1.1)
            ax2.set_title(f'Probability (t={t[frame]:.3f} fs)', fontsize=13)
            ax2.legend(loc='upper right', fontsize=10)
            ax2.grid(True, alpha=0.3)
            
            # Frame counter
            ax1.text(0.02, 0.98, f'Frame {frame+1}/{len(t)}',
                    transform=ax1.transAxes, fontsize=10, va='top',
                    bbox=dict(boxstyle='round', fc='white', alpha=0.7))
            
            # Stats
            stats = (f'T: {result["transmission_coefficient"]:.1%}  |  '
                    f'R: {result["reflection_coefficient"]:.1%}')
            ax2.text(0.5, -0.15, stats, transform=ax2.transAxes,
                    ha='center', va='top', fontsize=10,
                    bbox=dict(boxstyle='round', fc='wheat', alpha=0.3))
            
            return []
        
        print(f"    Creating {len(t)} frame animation...")
        anim = FuncAnimation(fig, animate, init_func=init, frames=len(t),
                           interval=1000/fps, blit=False)
        print(f"    Saving to {filepath}...")
        anim.save(filepath, writer='pillow', fps=fps, dpi=dpi)
        plt.close()
        print("    Done!")
