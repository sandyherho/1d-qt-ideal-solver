import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path

plt.style.use('seaborn-v0_8-darkgrid')

class Animator:
    @staticmethod
    def create_gif(result, filename, output_dir="outputs", 
                  title="Quantum Tunneling", fps=30, dpi=100):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / filename
        
        x, t = result['x'], result['t']
        psi, prob, V = result['psi'], result['probability'], result['potential']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        plt.subplots_adjust(hspace=0.3, top=0.92, bottom=0.08)
        
        psi_max = np.max(np.abs(psi))
        prob_max = np.max(prob)
        V_max = np.max(V)
        V_scale = 0.3 * psi_max / (V_max + 1e-10)
        
        def animate(frame):
            ax1.clear()
            ax2.clear()
            
            ax1.plot(x, psi[frame].real, 'b-', lw=2, label=r'Re($\psi$)')
            ax1.plot(x, psi[frame].imag, 'r-', lw=2, label=r'Im($\psi$)')
            ax1.axhline(0, color='k', lw=0.5, alpha=0.3)
            ax1.fill_between(x, 0, V * V_scale, alpha=0.2, color='gray')
            ax1.set_ylabel(r'$\psi$ [nm$^{-1/2}$]')
            ax1.set_ylim(-psi_max*1.1, psi_max*1.1)
            ax1.set_title(f'{title} (t={t[frame]:.3f} fs)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2.fill_between(x, 0, prob[frame], alpha=0.6, color='green')
            ax2.plot(x, prob[frame], 'g-', lw=1.5)
            V_prob = 0.5 * prob_max / (V_max + 1e-10)
            ax2.fill_between(x, 0, V * V_prob, alpha=0.2, color='gray')
            ax2.set_xlabel(r'Position $x$ [nm]')
            ax2.set_ylabel(r'$|\psi|^2$ [nm$^{-1}$]')
            ax2.set_ylim(0, prob_max*1.1)
            ax2.legend([r'$|\psi|^2$'])
            ax2.grid(True, alpha=0.3)
            
            stats = (f'T: {result["transmission_coefficient"]:.1%}  |  '
                    f'R: {result["reflection_coefficient"]:.1%}')
            ax2.text(0.5, -0.15, stats, transform=ax2.transAxes, ha='center')
        
        print(f"    Creating animation ({len(t)} frames)...")
        anim = FuncAnimation(fig, animate, frames=len(t),
                           interval=1000/fps, blit=False)
        anim.save(filepath, writer='pillow', fps=fps, dpi=dpi)
        plt.close()
        print(f"    Saved: {filename}")
