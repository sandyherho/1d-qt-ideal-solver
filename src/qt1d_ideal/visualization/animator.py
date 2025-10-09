"""
Professional visualization for quantum tunneling animations.
Creates publication-quality animations with modern aesthetics.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import matplotlib as mpl

# Set professional style
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.width'] = 1.5


class Animator:
    """Create professional quantum tunneling animations."""
    
    @staticmethod
    def create_gif(result, filename, output_dir="outputs", 
                  title="Quantum Tunneling", fps=30, dpi=100):
        """
        Create professional GIF animation with black background.
        
        Args:
            result: Simulation results dictionary
            filename: Output filename
            output_dir: Output directory
            title: Animation title
            fps: Frames per second
            dpi: Resolution
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / filename
        
        x, t = result['x'], result['t']
        psi, prob, V = result['psi'], result['probability'], result['potential']
        
        # Compute scales
        psi_max = np.max(np.abs(psi))
        prob_max = np.max(prob)
        V_max = np.max(V)
        
        # Create figure with black background
        fig = plt.figure(figsize=(12, 8), facecolor='black')
        
        # Create subplots with black background
        ax1 = plt.subplot(2, 1, 1, facecolor='#0a0a0a')
        ax2 = plt.subplot(2, 1, 2, facecolor='#0a0a0a')
        
        plt.subplots_adjust(hspace=0.35, top=0.92, bottom=0.08, left=0.08, right=0.96)
        
        # Scale potential for visualization
        V_scale = 0.25 * psi_max / (V_max + 1e-10)
        V_prob = 0.4 * prob_max / (V_max + 1e-10)
        
        # Color palette (vibrant but professional)
        color_real = '#00D9FF'      # Cyan
        color_imag = '#FF3E96'      # Magenta
        color_prob = '#00FF88'      # Green
        color_barrier = '#FFD700'   # Gold
        color_grid = '#2a2a2a'      # Dark gray
        color_text = '#E0E0E0'      # Light gray
        
        def animate(frame):
            ax1.clear()
            ax2.clear()
            
            # ================================================================
            # UPPER PLOT: WAVEFUNCTION (Real & Imaginary parts)
            # ================================================================
            # Plot wavefunction components
            ax1.plot(x, psi[frame].real, color=color_real, lw=2.5, 
                    label='Re(ψ)', alpha=0.9, zorder=3)
            ax1.plot(x, psi[frame].imag, color=color_imag, lw=2.5, 
                    label='Im(ψ)', alpha=0.9, zorder=3)
            
            # Zero line
            ax1.axhline(0, color=color_grid, lw=1, alpha=0.5, zorder=1)
            
            # Barrier (filled, semi-transparent)
            ax1.fill_between(x, 0, V * V_scale, 
                           color=color_barrier, alpha=0.15, zorder=2)
            ax1.plot(x, V * V_scale, color=color_barrier, lw=1.5, 
                    alpha=0.6, linestyle='--', label='Barrier', zorder=2)
            
            # Styling
            ax1.set_ylabel('Wavefunction ψ(x,t)  [nm$^{-1/2}$]', 
                          color=color_text, fontsize=11, fontweight='bold')
            ax1.set_ylim(-psi_max*1.15, psi_max*1.15)
            ax1.set_xlim(x[0], x[-1])
            
            # Title with time
            ax1.set_title(f'{title}   |   t = {t[frame]:.3f} fs', 
                         color=color_text, fontsize=14, fontweight='bold', 
                         pad=15)
            
            # Legend (upper right, clean)
            legend1 = ax1.legend(loc='upper right', framealpha=0.7, 
                               facecolor='#1a1a1a', edgecolor=color_grid,
                               fontsize=10, ncol=3)
            for text in legend1.get_texts():
                text.set_color(color_text)
            
            # Grid and spines
            ax1.grid(True, alpha=0.15, color=color_grid, linestyle='-', linewidth=0.5)
            ax1.spines['bottom'].set_color(color_grid)
            ax1.spines['top'].set_color(color_grid)
            ax1.spines['left'].set_color(color_grid)
            ax1.spines['right'].set_color(color_grid)
            ax1.tick_params(colors=color_text, which='both')
            
            # ================================================================
            # LOWER PLOT: PROBABILITY DENSITY
            # ================================================================
            # Plot probability density (filled)
            ax2.fill_between(x, 0, prob[frame], 
                           color=color_prob, alpha=0.4, zorder=3)
            ax2.plot(x, prob[frame], color=color_prob, lw=2.5, 
                    label='|ψ(x,t)|²', alpha=0.9, zorder=4)
            
            # Barrier (filled, semi-transparent)
            ax2.fill_between(x, 0, V * V_prob, 
                           color=color_barrier, alpha=0.15, zorder=2)
            ax2.plot(x, V * V_prob, color=color_barrier, lw=1.5, 
                    alpha=0.6, linestyle='--', label='Barrier', zorder=2)
            
            # Styling
            ax2.set_xlabel('Position x  [nm]', color=color_text, 
                          fontsize=11, fontweight='bold')
            ax2.set_ylabel('Probability Density |ψ|²  [nm$^{-1}$]', 
                          color=color_text, fontsize=11, fontweight='bold')
            ax2.set_ylim(0, prob_max*1.15)
            ax2.set_xlim(x[0], x[-1])
            
            # Legend (upper right)
            legend2 = ax2.legend(loc='upper right', framealpha=0.7, 
                               facecolor='#1a1a1a', edgecolor=color_grid,
                               fontsize=10)
            for text in legend2.get_texts():
                text.set_color(color_text)
            
            # Grid and spines
            ax2.grid(True, alpha=0.15, color=color_grid, linestyle='-', linewidth=0.5)
            ax2.spines['bottom'].set_color(color_grid)
            ax2.spines['top'].set_color(color_grid)
            ax2.spines['left'].set_color(color_grid)
            ax2.spines['right'].set_color(color_grid)
            ax2.tick_params(colors=color_text, which='both')
            
            # ================================================================
            # STATISTICS BOX (Bottom center, clean)
            # ================================================================
            T = result['transmission_coefficient']
            R = result['reflection_coefficient']
            
            stats_text = (f'Transmission: {T:.1%}   |   '
                         f'Reflection: {R:.1%}   |   '
                         f'T + R: {T+R:.3f}')
            
            ax2.text(0.5, -0.18, stats_text,
                    transform=ax2.transAxes,
                    ha='center', va='top',
                    color=color_text,
                    fontsize=10,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', 
                            facecolor='#1a1a1a', 
                            edgecolor=color_grid,
                            linewidth=1.5,
                            alpha=0.8))
        
        # Create animation
        print(f"    Creating animation ({len(t)} frames)...")
        anim = FuncAnimation(fig, animate, frames=len(t),
                           interval=1000/fps, blit=False)
        
        # Save with high quality
        anim.save(filepath, writer='pillow', fps=fps, dpi=dpi)
        plt.close()
        print(f"    Saved: {filename}")
