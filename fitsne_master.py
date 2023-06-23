import os
from run_fitsne import apply_fast_tsne
import sys


def get_folders(path):
    folders = []
    for entry in os.scandir(path):
        if entry.is_dir():
            folders.append(entry.name)
    return folders


if __name__ == "__main__":
    path = 'results/'
    folders = get_folders(path)

    if len(sys.argv) >= 2:
        p = int(sys.argv[1])
    else:
        p = None

    for fd in folders:
        if fd == "experiment_000" or "100000" in fd:
            print(f"Skipped {fd}")
            continue
        else:
            print(f"Starting experiment {fd}")
            if p is None:
                apply_fast_tsne(fd)
            else:
                apply_fast_tsne(fd, perplexity=p)

