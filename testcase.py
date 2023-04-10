import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap
from numba import config




def main():
    # penguins = pd.read_csv(
    #     "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/c19a904462482430170bfe2c718775ddb7dbb885/inst/extdata/penguins.csv")
    # penguins = penguins.dropna()
    # penguin_data = penguins[
    #     [
    #         "bill_length_mm",
    #         "bill_depth_mm",
    #         "flipper_length_mm",
    #         "body_mass_g",
    #     ]
    # ].values
    # scaled_penguin_data = StandardScaler().fit_transform(penguin_data)
    # reducer = umap.UMAP()
    # embedding = reducer.fit_transform(scaled_penguin_data)

    # plt.scatter(
    #     embedding[:, 0],
    #     embedding[:, 1],
    #     c=[sns.color_palette()[x] for x in penguins.species.map({"Adelie": 0, "Chinstrap": 1, "Gentoo": 2})])
    # plt.gca().set_aspect('equal', 'datalim')
    # plt.title('UMAP projection of the Penguin dataset', fontsize=24)
    # plt.show()

    num_samples = 20

    # make a simple unit circle
    theta = np.linspace(0, 2 * np.pi, num_samples)
    a, b = 1 * np.cos(theta), 1 * np.sin(theta)

    # generate the points
    # theta = np.random.rand((num_samples)) * (2 * np.pi)
    x, y = np.cos(theta), np.sin(theta)

    # plots
    # plt.figure(figsize=(7, 6))
    # plt.scatter(x, y, label='Samples', marker="o")
    # plt.ylim([-1.5, 1.5])
    # plt.xlim([-1.5, 1.5])
    # plt.grid()
    # plt.legend(loc='upper right')
    # plt.show(block=True)

    orig_data = np.zeros((x.shape[0], 2))

    orig_data[:, 0] = x
    orig_data[:, 1] = y

    # If given a list as the parameter n_epochs, then we have all of the embeddings in:
    # mapper.embedding_list

    epochs = [10,20,30]
    # epochs = None
    mapper = umap.UMAP(random_state = 42, n_epochs=epochs, init=orig_data, negative_sample_rate=20,
                       force_no_optimization=True)
    mapper.fit(orig_data)
    embedding = mapper.transform(orig_data)

    plt.figure(figsize=(7, 6))
    plt.scatter(orig_data[:, 0], orig_data[:, 1], label='Samples', marker="o")
    plt.ylim([-1.5, 1.5])
    plt.xlim([-1.5, 1.5])
    plt.grid()
    plt.legend(loc='upper right')
    plt.show(block=True)

    if isinstance(epochs, list):
        intermediate_embeddings = mapper.embedding_list_
        for k in range(len(epochs)):
            plt.figure(figsize=(7, 6))
            plt.scatter(intermediate_embeddings[k][:, 0],
                        intermediate_embeddings[k][:, 1], label='Samples', marker="o")
            plt.grid()
            plt.legend(loc='upper right')
            plt.title(f"Epoch {epochs[k]}")
            plt.show(block=True)
    else:
        plt.figure(figsize=(7, 6))
        plt.scatter(embedding[:, 0],
                    embedding[:, 1], label='Samples', marker="o")
        plt.grid()
        plt.legend(loc='upper right')
        plt.show(block=True)



if __name__ == "__main__":
    main()
    print("end")
