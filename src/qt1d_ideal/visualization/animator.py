"""
Performance: Parallel rendering (~10x faster)
Physics: Shows where probability is absorbed at domain edges
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl
from multiprocessing import Pool, cpu_count
from functools import partial
import io
from PIL import Image

# Professional style
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.width'] = 1.5


def _render_single_frame(frame_data):
    """
    Render a single frame with visible absorbing boundaries.
    
    Shows:
    - Wavefunction (real/imaginary parts)
    - Probability density
    - Potential barrier
    - Absorbing boundary regions (shaded)
    """
    (frame_idx, x, t_val, psi_frame, prob_frame, V, mask,
     x_min, x_max, boundary_width,
     psi_max, prob_max, V_scale, V_prob, 
     T, R, A, title, colors) = frame_data
    
    # Create figure with black background
    fig = plt.figure(figsize=(12, 8), facecolor='black')
    
    # Create subplots
    ax1 = plt.subplot(2, 1, 1, facecolor='#0a0a0a')
    ax2 = plt.subplot(2, 1, 2, facecolor='#0a0a0a')
    
    plt.subplots_adjust(hspace=0.35, top=0.92, bottom=0.08, left=0.08, right=0.96)
    
    # ====================================================================
    # SHOW ABSORBING BOUNDARY REGIONS (NEW!)
    # ====================================================================
    x_left_boundary = x_min + boundary_width
    x_right_boundary = x_max - boundary_width
    
    # Shade absorbing regions on both plots
    for ax in [ax1, ax2]:
        # Left absorbing region
        ax.axvspan(x_min, x_left_boundary, 
                   color='red', alpha=0.08, zorder=0,
                   label='Absorbing boundary' if ax == ax1 else '')
        # Right absorbing region
        ax.axvspan(x_right_boundary, x_max, 
                   color='red', alpha=0.08, zorder=0)
        # Boundary markers
        ax.axvline(x_left_boundary, color='red', lw=1.5, 
                   alpha=0.4, linestyle=':', zorder=1)
        ax.axvline(x_right_boundary, color='red', lw=1.5, 
                   alpha=0.4, linestyle=':', zorder=1)
    
    # ====================================================================
    # UPPER PLOT: WAVEFUNCTION
    # ====================================================================
    ax1.plot(x, psi_frame.real, color=colors['real'], lw=2.5, 
            label='Re(ψ)', alpha=0.9, zorder=3)
    ax1.plot(x, psi_frame.imag, color=colors['imag'], lw=2.5, 
            label='Im(ψ)', alpha=0.9, zorder=3)
    
    ax1.axhline(0, color=colors['grid'], lw=1, alpha=0.5, zorder=1)
    
    # Potential barrier
    ax1.fill_between(x, 0, V * V_scale, 
                   color=colors['barrier'], alpha=0.2, zorder=2)
    ax1.plot(x, V * V_scale, color=colors['barrier'], lw=2.0, 
            alpha=0.7, linestyle='--', label='Barrier', zorder=2)
    
    ax1.set_ylabel('Wavefunction ψ(x,t)  [nm$^{-1/2}$]', 
                  color=colors['text'], fontsize=11, fontweight='bold')
    ax1.set_ylim(-psi_max*1.15, psi_max*1.15)
    ax1.set_xlim(x[0], x[-1])
    
    ax1.set_title(f'{title}   |   t = {t_val:.3f} fs', 
                 color=colors['text'], fontsize=14, fontweight='bold', pad=15)
    
    legend1 = ax1.legend(loc='upper right', framealpha=0.8, 
                       facecolor='#1a1a1a', edgecolor=colors['grid'],
                       fontsize=9, ncol=4)
    for text in legend1.get_texts():
        text.set_color(colors['text'])
    
    ax1.grid(True, alpha=0.15, color=colors['grid'], linestyle='-', linewidth=0.5)
    for spine in ax1.spines.values():
        spine.set_color(colors['grid'])
    ax1.tick_params(colors=colors['text'], which='both')
    
    # ====================================================================
    # LOWER PLOT: PROBABILITY DENSITY
    # ====================================================================
    ax2.fill_between(x, 0, prob_frame, 
                   color=colors['prob'], alpha=0.4, zorder=3)
    ax2.plot(x, prob_frame, color=colors['prob'], lw=2.5, 
            label='|ψ(x,t)|²', alpha=0.9, zorder=4)
    
    # Potential barrier
    ax2.fill_between(x, 0, V * V_prob, 
                   color=colors['barrier'], alpha=0.2, zorder=2)
    ax2.plot(x, V * V_prob, color=colors['barrier'], lw=2.0, 
            alpha=0.7, linestyle='--', label='Barrier', zorder=2)
    
    ax2.set_xlabel('Position x  [nm]', color=colors['text'], 
                  fontsize=11, fontweight='bold')
    ax2.set_ylabel('Probability Density |ψ|²  [nm$^{-1}$]', 
                  color=colors['text'], fontsize=11, fontweight='bold')
    ax2.set_ylim(0, prob_max*1.15)
    ax2.set_xlim(x[0], x[-1])
    
    legend2 = ax2.legend(loc='upper right', framealpha=0.8, 
                       facecolor='#1a1a1a', edgecolor=colors['grid'],
                       fontsize=9)
    for text in legend2.get_texts():
        text.set_color(colors['text'])
    
    ax2.grid(True, alpha=0.15, color=colors['grid'], linestyle='-', linewidth=0.5)
    for spine in ax2.spines.values():
        spine.set_color(colors['grid'])
    ax2.tick_params(colors=colors['text'], which='both')
    
    # ====================================================================
    # STATISTICS BOX (UPDATED with A)
    # ====================================================================
    stats_text = (f'T: {T:.1%}   |   R: {R:.1%}   |   '
                 f'A: {A:.1%}   |   T+R+A: {T+R+A:.3f}')
    
    # Color code based on conservation
    total = T + R + A
    if abs(total - 1.0) < 0.05:
        box_color = '#1a4d1a'  # Green tint for good conservation
        edge_color = '#00FF88'
    elif abs(total - 1.0) < 0.1:
        box_color = '#4d4d1a'  # Yellow tint for warning
        edge_color = '#FFD700'
    else:
        box_color = '#4d1a1a'  # Red tint for violation
        edge_color = '#FF3E96'
    
    ax2.text(0.5, -0.18, stats_text,
            transform=ax2.transAxes,
            ha='center', va='top',
            color=colors['text'],
            fontsize=10,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', 
                    facecolor=box_color, 
                    edgecolor=edge_color,
                    linewidth=1.5,
                    alpha=0.9))
    
    # Convert to PIL Image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, facecolor='black')
    buf.seek(0)
    img = Image.open(buf)
    img_copy = img.copy()
    buf.close()
    plt.close(fig)
    
    return img_copy


class Animator:
    """Create professional quantum tunneling animations with parallel rendering."""
    
    @staticmethod
    def create_gif(result, filename, output_dir="outputs", 
                  title="Quantum Tunneling", fps=30, dpi=100):
        """
        Create professional GIF animation with visible absorbing boundaries.
        
        NEW: Shows absorbing boundary regions where probability is removed
        Performance: ~10x faster via parallel rendering
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / filename
        
        x, t = result['x'], result['t']
        psi, prob, V = result['psi'], result['probability'], result['potential']
        T = result['transmission_coefficient']
        R = result['reflection_coefficient']
        A = result.get('absorbed_probability', 0.0)
        
        # Get boundary parameters
        params = result.get('params', {})
        boundary_width = params.get('boundary_width', 2.0)
        x_min, x_max = x[0], x[-1]
        
        # Compute scales
        psi_max = np.max(np.abs(psi))
        prob_max = np.max(prob)
        V_max = np.max(V)
        
        V_scale = 0.25 * psi_max / (V_max + 1e-10)
        V_prob = 0.4 * prob_max / (V_max + 1e-10)
        
        # Color palette
        colors = {
            'real': '#00D9FF',
            'imag': '#FF3E96',
            'prob': '#00FF88',
            'barrier': '#FFD700',
            'grid': '#2a2a2a',
            'text': '#E0E0E0'
        }
        
        # Create absorbing mask for visualization (not used in computation, just for reference)
        mask = np.ones(len(x))
        n_boundary = int(boundary_width / (x[1] - x[0]))
        if n_boundary > 0:
            for i in range(n_boundary):
                ratio = i / n_boundary
                cos_factor = np.cos(0.5 * np.pi * (1.0 - ratio))**2
                strength = params.get('boundary_strength', 0.1)
                mask[i] = 1.0 - strength * (1.0 - cos_factor)
                mask[-(i+1)] = mask[i]
        
        n_frames = len(t)
        print(f"    Rendering {n_frames} frames in parallel...")
        
        # Prepare frame data for parallel processing
        frame_data_list = []
        for i in range(n_frames):
            frame_data = (
                i, x, t[i], psi[i], prob[i], V, mask,
                x_min, x_max, boundary_width,
                psi_max, prob_max, V_scale, V_prob,
                T, R, A, title, colors
            )
            frame_data_list.append(frame_data)
        
        # Parallel rendering
        n_processes = max(1, cpu_count() - 1)
        print(f"    Using {n_processes} CPU cores...")
        
        with Pool(processes=n_processes) as pool:
            frames = pool.map(_render_single_frame, frame_data_list)
        
        print(f"    Saving GIF ({n_frames} frames @ {fps} fps)...")
        
        # Save as GIF
        frames[0].save(
            filepath,
            save_all=True,
            append_images=frames[1:],
            duration=1000/fps,
            loop=0,
            optimize=False
        )
        
        print(f"    ✓ Animation complete!")
