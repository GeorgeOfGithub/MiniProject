from os.path import join
import sys
import os
from Optimized import load_data, jacobi, summary_stats
import numpy as np
import matplotlib.pyplot as plt


def plot_figure3_style(u, interior_mask, bid, vmin=0, vmax=25, dpi=150, save_path=None, show=False):
    """Create a Figure-3-style visualization for a single floorplan.

    If `save_path` is provided the image is written. If `show=True`, the
    plot will be displayed interactively. Returns the Figure object when
    `show=False` and `save_path` is None.
    """
    arr = u[1:-1, 1:-1]
    mask = interior_mask

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.set_facecolor('black')

    im = ax.imshow(np.where(mask, arr, np.nan), cmap='inferno', vmin=vmin, vmax=vmax)
    ticks = [0, 100, 200, 300, 400, 500]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_title(f'ID: {bid}', pad=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.08, pad=0.04)
    cbar.set_label('Temperature')

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=dpi, facecolor=fig.get_facecolor())

    if show:
        plt.show()
        return None
    return fig


def save_figure_for_building(u, interior_mask, bid, out_dir='batch_output', dpi=300, vmax=25):
    """Save a Figure-3-style PNG for `bid` into `out_dir` and return path."""
    os.makedirs(out_dir, exist_ok=True)
    save_path = join(out_dir, f'figure_{bid}.png')
    plot_figure3_style(u, interior_mask, bid, vmin=0, vmax=vmax, dpi=dpi, save_path=save_path, show=False)
    return save_path


if __name__ == '__main__':
    # This main mirrors the original main from Optimized.py with a minimal
    # addition: for each computed steady-state we export a PNG into
    # `batch_output/` using the helper above. This allows later importing
    # the functions from this module without changing Optimized.py.

    # Load data
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    if len(sys.argv) < 2:
        N = 1
    else:
        N = int(sys.argv[1])
    
    building_ids = building_ids[:N]

    # Load floor plans
    all_u0 = np.empty((N, 514, 514))
    all_interior_mask = np.empty((N, 512, 512), dtype='bool')

    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    # Run jacobi iterations for each floor plan
    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    all_u = np.empty_like(all_u0)
    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        u = jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)
        all_u[i] = u
        # minimal change: save the figure for this building
        try:
            save_figure_for_building(u, interior_mask, building_ids[i], out_dir='batch_output', dpi=300, vmax=25)
        except Exception:
            # don't fail the batch if saving fails; continue printing stats
            pass

    # Print summary statistics in CSV format
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys)) # CSV header
    
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid}, " + ", ".join(str(stats[k]) for k in stat_keys))