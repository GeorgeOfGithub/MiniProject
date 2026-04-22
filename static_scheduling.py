from concurrent.futures import ProcessPoolExecutor
from os.path import join
import sys

from optimized_code import load_data, jacobi, summary_stats


def run_chunk(args):
    load_dir, chunk_bids, start_idx, max_iter, abs_tol = args
    chunk_results = []
    for offset, bid in enumerate(chunk_bids):
        u0, interior_mask = load_data(load_dir, bid)
        u = jacobi(u0, interior_mask, max_iter, abs_tol)
        stats = summary_stats(u, interior_mask)
        chunk_results.append((start_idx + offset, bid, stats))
    return chunk_results

if __name__ == '__main__':
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    if len(sys.argv) < 2:
        N = 1
    else:
        N = int(sys.argv[1])

    if len(sys.argv) < 3:
        NUM_WORKERS = 1
    else:
        NUM_WORKERS = int(sys.argv[2])

    if NUM_WORKERS < 1:
        raise ValueError('NUM_WORKERS skal vaere mindst 1.')

    if N < 1:
        raise ValueError('N skal vaere mindst 1.')
    
    building_ids = building_ids[:N]
    N = len(building_ids)

    base_tasks_per_worker = N // NUM_WORKERS
    remainder_tasks = N % NUM_WORKERS

    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    worker_args = []
    start = 0
    for worker_id in range(NUM_WORKERS):
        # Distribute remaining tasks statically: first workers get one extra task.
        tasks_for_this_worker = base_tasks_per_worker + (1 if worker_id < remainder_tasks else 0)
        end = start + tasks_for_this_worker
        worker_args.append((LOAD_DIR, building_ids[start:end], start, MAX_ITER, ABS_TOL))
        start = end

    results = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for chunk_result in executor.map(run_chunk, worker_args):
            results.extend(chunk_result)

    results.sort(key=lambda x: x[0])

    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))

    for _, bid, stats in results:
        print(f"{bid}, " + ", ".join(str(stats[k]) for k in stat_keys))