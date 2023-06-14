import bhtsne.bhtsne as bhtsne
import numpy as np
import time
import pickle
import sys, os
from experiment_functions import *

"""Reads data from a manifold.npy file in the specified directory,
applies t-SNE dimension reduction to the specified dimensions with the
specified perplexity and writes the resulting embedding into another .npy file.
"""


def apply_tsne(directory, perplexity=None, random_seed=42):
    dir_path = os.getcwd() + "/results/" + directory
    log_path = dir_path + "/log_tsne.txt"

    # Load original data
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s - %(levelname)-10s - %(message)s')
    if perplexity is not None:
        logging.info(f"Perplexity: {perplexity}")
    else:
        logging.info("Perplexity: default value")

    logging.info(f"Random seed: {random_seed}")

    manifold = np.load(dir_path + "/manifold.npy")

    logging.info("Beginning Barnes-Hut t-SNE embedding")
    with warnings.catch_warnings(record=True) as w:
        # Create embedding
        if perplexity is not None:
            start = time()
            embedding = bhtsne.run_bh_tsne(manifold, perplexity=perplexity, randseed=random_seed,
                                           initial_dims=manifold.shape[1])
            end = time()
        else:
            start = time()
            embedding = bhtsne.run_bh_tsne(manifold, randseed=random_seed,
                                           initial_dims=manifold.shape[1])
            end = time()

        logging.info(f"Barnes-Hut t-SNE finished in {end-start} seconds")

        for warning in w:
            logging.warning(str(warning.message))

        np.save(dir_path + f"/tsne_embedding_p_{perplexity}.npy", embedding)

        sample = np.load(dir_path + "/sample.npy")
        fig, ax = plot_2d_curve(embedding, sample)
        plt.savefig(dir_path + f"/tsne_plot_p_{perplexity}.png")


if __name__ == "__main__":
    """
    Args:
        directory: directory where the data is and where to save the result
        p: perplexity
    """
    directory = sys.argv[1]
    if len(sys.argv) >= 3:
        p = sys.argv[2]
    else:
        p = None
    if len(sys.argv) >= 4:
        random_seed = int(sys.argv[3])
    else:
        random_seed = 42

    apply_tsne(directory, perplexity=p, random_seed=random_seed)
