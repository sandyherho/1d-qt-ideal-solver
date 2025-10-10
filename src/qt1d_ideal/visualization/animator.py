"""
Professional visualization with zone visualization and full statistics
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl
from multiprocessing import Pool, cpu_count
import io
from PIL import Image

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.width'] = 1.5


def _render_single_frame(frame_data):
    """Render frame with color-coded zones."""
    (frame_idx, x, t_val, psi_frame, prob_frame, V,
     x_min, x_max, x_safe_left, x_safe_right,
     x_barrier_start, x_barrier_end,
     psi_max, prob_max, V_scale, V_prob, 
     T, R, A, title, colors) = frame_data
    
    fig = plt.figure(figsize=(12, 9), facecolor='black')
    
    ax1 = plt.subplot(2, 1, 1, facecolor='#0a0a0a')
    ax2 = plt.subplot(2, 1, 2, facecolor='#0a0a0a')
    
    plt.subplots_adjust(hspace=0.3, top=0.94, bottom=0.12, left=0.08, right=0.96)
    
    # Show all zones
    for ax in [ax1, ax2]:
        # Absorbing boundaries (red)
        ax.axvspan(x_min, x_safe_left, 
                   color='red', alpha=0.12, zorder=0,
                   label='Absorbing' if ax == ax1 else '')
        ax.axvspan(x_safe_right, x_max, 
                   color='red', alpha=0.12, zorder=0)
        
        # Detection zones (green/cyan)
        ax.axvspan(x_safe_left, x_barrier_start, 
                   color='green', alpha=0.08, zorder=0,
                   label='R detect' if ax == ax1 else '')
        ax.axvspan(x_barrier_end, x_safe_right, 
                   color='cyan', alpha=0.08, zorder=0,
                   label='T detect' if ax == ax1 else '')
        
        # Zone boundary lines
        ax.axvline(x_safe_left, color='red', lw=1.5, 
                   alpha=0.5, linestyle=':', zorder=1)
        ax.axvline(x_safe_right, color='red', lw=1.5, 
                   alpha=0.5, linestyle=':', zorder=1)
        ax.axvline(x_barrier_start, color='#FFD700', lw=1.5, 
                   alpha=0.6, linestyle='--', zorder=1)
        ax.axvline(x_barrier_end, color='#FFD700', lw=1.5, 
                   alpha=0.6, linestyle='--', zorder=1)
    
    # Upper plot: Wavefunction
    ax1.plot(x, psi_frame.real, color=colors['real'], lw=2.5, 
            label='Re(ψ)', alpha=0.9, zorder=3)
    ax1.plot(x, psi_frame.imag, color=colors['imag'], lw=2.5, 
            label='Im(ψ)', alpha=0.9, zorder=3)
    
    ax1.axhline(0, color=colors['grid'], lw=1, alpha=0.5, zorder=1)
    
    ax1.fill_between(x, 0, V * V_scale, 
                   color=colors['barrier'], alpha=0.25, zorder=2,
                   label='Barrier')
    ax1.plot(x, V * V_scale, color=colors['barrier'], lw=2.5, 
            alpha=0.8, linestyle='-', zorder=2)
    
    ax1.set_ylabel('Wavefunction ψ(x,t)  [nm$^{-1/2}$]', 
                  color=colors['text'], fontsize=11, fontweight='bold')
    ax1.set_ylim(-psi_max*1.15, psi_max*1.15)
    ax1.set_xlim(x[0], x[-1])
    
    ax1.set_title(f'{title}   |   t = {t_val:.3f} fs', 
                 color=colors['text'], fontsize=14, fontweight='bold', pad=15)
    
    legend1 = ax1.legend(loc='upper right', framealpha=0.85, 
                       facecolor='#1a1a1a', edgecolor=colors['grid'],
                       fontsize=8, ncol=6)
    for text in legend1.get_texts():
        text.set_color(colors['text'])
    
    ax1.grid(True, alpha=0.15, color=colors['grid'], linestyle='-', linewidth=0.5)
    for spine in ax1.spines.values():
        spine.set_color(colors['grid'])
    ax1.tick_params(colors=colors['text'], which='both')
    
    # Lower plot: Probability density
    ax2.fill_between(x, 0, prob_frame, 
                   color=colors['prob'], alpha=0.5, zorder=3,
                   label='|ψ(x,t)|²')
    ax2.plot(x, prob_frame, color=colors['prob'], lw=2.5, 
            alpha=0.95, zorder=4)
    
    ax2.fill_between(x, 0, V * V_prob, 
                   color=colors['barrier'], alpha=0.25, zorder=2,
                   label='Barrier')
    ax2.plot(x, V * V_prob, color=colors['barrier'], lw=2.5, 
            alpha=0.8, linestyle='-', zorder=2)
    
    ax2.set_xlabel('Position x  [nm]', color=colors['text'], 
                  fontsize=11, fontweight='bold')
    ax2.set_ylabel('Probability Density |ψ|²  [nm$^{-1}$]', 
                  color=colors['text'], fontsize=11, fontweight='bold')
    ax2.set_ylim(0, prob_max*1.15)
    ax2.set_xlim(x[0], x[-1])
    
    legend2 = ax2.legend(loc='upper right', framealpha=0.85, 
                       facecolor='#1a1a1a', edgecolor=colors['grid'],
                       fontsize=9)
    for text in legend2.get_texts():
        text.set_color(colors['text'])
    
    ax2.grid(True, alpha=0.15, color=colors['grid'], linestyle='-', linewidth=0.5)
    for spine in ax2.spines.values():
        spine.set_color(colors['grid'])
    ax2.tick_params(colors=colors['text'], which='both')
    
    # Statistics box
    total = T + R + A
    
    stats_text = (
        f'Transmission (T): {T:.4f} ({T*100:.2f}%)   |   '
        f'Reflection (R): {R:.4f} ({R*100:.2f}%)\n'
        f'Absorbed (A): {A:.4f} ({A*100:.2f}%)   |   '
        f'T + R + A = {total:.4f}'
    )
    
    if abs(total - 1.0) < 0.05:
        box_color = '#1a4d1a'
        edge_color = '#00FF88'
        status = '✓'
    elif abs(total - 1.0) < 0.1:
        box_color = '#4d4d1a'
        edge_color = '#FFD700'
        status = '⚠'
    else:
        box_color = '#4d1a1a'
        edge_color = '#FF3E96'
        status = '✗'
    
    stats_text = f'{status} {stats_text}'
    
    fig.text(0.5, 0.04, stats_text,
            ha='center', va='center',
            color=colors['text'],
            fontsize=10,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.8', 
                    facecolor=box_color, 
                    edgecolor=edge_color,
                    linewidth=2.0,
                    alpha=0.95),
            transform=fig.transFigure)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, facecolor='black', 
                bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    img = Image.open(buf)
    img_copy = img.copy()
    buf.close()
    plt.close(fig)
    
    return img_copy


class Animator:
    """Professional quantum tunneling animations with parallel rendering."""
    
    @staticmethod
    def create_gif(result, filename, output_dir="outputs", 
                  title="Quantum Tunneling", fps=30, dpi=100):
        """Create GIF with proper zone visualization."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / filename
        
        x, t = result['x'], result['t']
        psi, prob, V = result['psi'], result['probability'], result['potential']
        T = result['transmission_coefficient']
        R = result['reflection_coefficient']
        A = result.get('absorbed_probability', 0.0)
        
        params = result.get('params', {})
        boundary_width = params.get('boundary_width', 2.0)
        x_min, x_max = x[0], x[-1]
        
        x_safe_left = x_min + boundary_width
        x_safe_right = x_max - boundary_width
        
        detection = result.get('detection_indices', {})
        if 'barrier_start' in detection and 'barrier_end' in detection:
            x_barrier_start = x[detection['barrier_start']]
            x_barrier_end = x[detection['barrier_end']]
        else:
            barrier_mask = V > 0.1 * np.max(V)
            if np.any(barrier_mask):
                barrier_indices = np.where(barrier_mask)[0]
                x_barrier_start = x[barrier_indices[0]]
                x_barrier_end = x[barrier_indices[-1]]
            else:
                x_barrier_start = 0.0
                x_barrier_end = 0.0
        
        psi_max = np.max(np.abs(psi))
        prob_max = np.max(prob)
        V_max = np.max(V)
        
        V_scale = 0.25 * psi_max / (V_max + 1e-10)
        V_prob = 0.4 * prob_max / (V_max + 1e-10)
        
        colors = {
            'real': '#00D9FF',
            'imag': '#FF3E96',
            'prob': '#00FF88',
            'barrier': '#FFD700',
            'grid': '#2a2a2a',
            'text': '#E0E0E0'
        }
        
        n_frames = len(t)
        print(f"    Rendering {n_frames} frames in parallel...")
        
        frame_data_list = []
        for i in range(n_frames):
            frame_data = (
                i, x, t[i], psi[i], prob[i], V,
                x_min, x_max, x_safe_left, x_safe_right,
                x_barrier_start, x_barrier_end,
                psi_max, prob_max, V_scale, V_prob,
                T, R, A, title, colors
            )
            frame_data_list.append(frame_data)
        
        n_processes = max(1, cpu_count() - 1)
        print(f"    Using {n_processes} CPU cores...")
        
        with Pool(processes=n_processes) as pool:
            frames = pool.map(_render_single_frame, frame_data_list)
        
        print(f"    Saving GIF ({n_frames} frames @ {fps} fps)...")
        
        frames[0].save(
            filepath,
            save_all=True,
            append_images=frames[1:],
            duration=1000/fps,
            loop=0,
            optimize=False
        )
        
        print(f"    ✓ Animation complete!")
