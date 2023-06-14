import os
from run_tsne import apply_tsne
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
        if "100000" not in fd:
            print(f"Skipped {fd}")
            continue
        print(f"Starting experiment {fd}")
        if p is None:
            apply_tsne(fd)
        else:
            apply_tsne(fd, perplexity=p)
