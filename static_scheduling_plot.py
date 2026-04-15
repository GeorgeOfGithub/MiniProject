import sys
import re
import glob
import os
import matplotlib.pyplot as plt

def parse_timing_file(path):
    workers = []
    elapsed = []
    pattern = re.compile(r'workers=(\d+)\s+elapsed=([\d.]+)')
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                workers.append(int(m.group(1)))
                elapsed.append(float(m.group(2)))
    return workers, elapsed

def main():
    if len(sys.argv) >= 2:
        path = sys.argv[1]
    else:
        # Find nyeste static_scaling fil automatisk
        files = glob.glob('batch_output/static_scaling_*.txt')
        if not files:
            print('Ingen timing-fil fundet i batch_output/. Angiv stien som argument.')
            sys.exit(1)
        path = max(files, key=os.path.getmtime)
        print(f'Bruger fil: {path}')

    workers, elapsed = parse_timing_file(path)
    if not workers:
        print('Ingen timing-data fundet i filen.')
        sys.exit(1)

    t1 = elapsed[workers.index(1)] if 1 in workers else elapsed[0]
    speedup = [t1 / t for t in elapsed]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(workers, speedup, 'o-', label='Measured speedup')
    ax.set_xlabel('Number of Workers')
    ax.set_ylabel('Speedup')
    ax.set_title('Static scheduling speedup')
    ax.legend()
    ax.set_xticks(workers)
    ax.grid(True)

    out = 'batch_output/speedup_plot3.png'
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f'Plot saved: {out}')

if __name__ == '__main__':
    main()
