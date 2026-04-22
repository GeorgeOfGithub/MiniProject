from os.path import join
import sys
import os
from optimized_code import load_data, jacobi, summary_stats
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda

@cuda.jit
def jacobi_kernel(u, u_new, interior_mask):
    col, row = cuda.grid(2)          # 2-D thread index into the 512×512 interior
 
    # Guard: stay inside the interior grid
    if row >= interior_mask.shape[0] or col >= interior_mask.shape[1]:
        return
    if not interior_mask[row, col]:  # boundary / wall cell — leave untouched
        return
 
    # Indices into the padded u array (interior starts at [1,1])
    r = row + 1
    c = col + 1
 
    # Average of the four neighbours
    u_new[r, c] = 0.25 * (u[r, c - 1] + u[r, c + 1] +
                           u[r - 1, c] + u[r + 1, c])
    

def jacobi_cuda(u, interior_mask, max_iter):
    # Copy data to device
    d_u        = cuda.to_device(np.ascontiguousarray(u,             dtype=np.float64))
    d_u_new    = cuda.to_device(np.ascontiguousarray(u,             dtype=np.float64))
    d_mask     = cuda.to_device(np.ascontiguousarray(interior_mask, dtype=np.bool_))
 
    # CUDA config
    THREADS = (16, 16)
    bx = (interior_mask.shape[1] + THREADS[0] - 1) // THREADS[0]
    by = (interior_mask.shape[0] + THREADS[1] - 1) // THREADS[1]
    BLOCKS = (bx, by)
 
    # Run all Jacobi iterations on GPU
    for _ in range(max_iter):
        jacobi_kernel[BLOCKS, THREADS](d_u, d_u_new, d_mask)
        # Swap u and u_new for next iteration
        d_u, d_u_new = d_u_new, d_u
 
    # Back to host
    return d_u.copy_to_host()


def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = np.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = np.sum(u_interior < 15) / u_interior.size * 100
    return {
        'mean_temp': mean_temp,
        'std_temp': std_temp,
        'pct_above_18': pct_above_18,
        'pct_below_15': pct_below_15,
    }

if __name__ == '__main__':
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()
 
    N = int(sys.argv[1]) if len(sys.argv) >= 2 else 1
    building_ids = building_ids[:N]
 
    # Load floor plans
    all_u0            = np.empty((N, 514, 514))
    all_interior_mask = np.empty((N, 512, 512), dtype='bool')
 
    for i, bid in enumerate(building_ids):
        u0, interior_mask      = load_data(LOAD_DIR, bid)
        all_u0[i]              = u0
        all_interior_mask[i]   = interior_mask
 
    MAX_ITER = 20_000
 
    # Run CUDA Jacobi for each floor plan
    all_u = np.empty_like(all_u0)
    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        all_u[i] = jacobi_cuda(u0, interior_mask, MAX_ITER)
 
    # Print summary statistics in CSV format
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))
 
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid}, " + ", ".join(str(stats[k]) for k in stat_keys))