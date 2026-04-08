import sys
import multiprocessing as mp
from os.path import join
from Optimized import load_data, jacobi, summary_stats

def do_jacobi(args):
    # Unpack arguments
    bid, load_dir, max_iter, abs_tol = args
    # Load data
    u0, interior_mask = load_data(load_dir, bid)

    # Do the jacobi iterations
    u = jacobi(u0, interior_mask, max_iter, abs_tol)

    # Calculate stats
    stats = summary_stats(u, interior_mask)
    return bid, stats


if __name__ == '__main__':
    # Load data
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    
    # Safely load IDs
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    if len(sys.argv) < 2:
        N = 1
    else:
        N = int(sys.argv[1])
    
    building_ids = building_ids[:N]

    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    # 2. Package the arguments for each task
    tasks = [(bid, LOAD_DIR, MAX_ITER, ABS_TOL) for bid in building_ids]

    # Print CSV header
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))

    # 3. Setup the Multiprocessing Pool
    # mp.cpu_count() automatically scales to the number of cores on your machine/node.
    n_processes = mp.cpu_count() 
    
    with mp.Pool(processes=n_processes) as pool:
        # imap_unordered yields results as soon as they are ready, regardless of input order.
        # chunksize=1 guarantees strict dynamic load balancing (workers pull 1 task at a time).
        for bid, stats in pool.imap_unordered(do_jacobi, tasks, chunksize=1):
            print(f"{bid}, " + ", ".join(str(stats[k]) for k in stat_keys))