import os
import umap
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from PIL import Image
import glob


def get_penguins():
    penguins = pd.read_csv(
        "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/c19a904462482430170bfe2c718775ddb7dbb885/inst/extdata/penguins.csv")
    penguins = penguins.dropna()
    penguin_data = penguins[
        [
            "bill_length_mm",
            "bill_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
        ]
    ].values
    data = StandardScaler().fit_transform(penguin_data)
    labels = np.array(list(penguins.species))
    return data, labels


def make_graph(data=None, labels=None, name=None, data_path=None, epochs=None, random_state=42,
               force_no_optimization=False, dim=2):
    # Example data in here

    # Set up directories for image saving
    directory = os.path.dirname(__file__)
    if name is None:
        res_dir = os.path.join(directory, "graphs_no_name/")
    else:
        res_dir = os.path.join(directory, "graphs_"+name+"/")
    if not os.path.isdir(res_dir):
        os.makedirs(res_dir)
    img_dir = os.path.join(res_dir, "images/")
    if not os.path.isdir(img_dir):
        os.makedirs(img_dir)
    gif_dir = os.path.join(res_dir, "gif/")
    if not os.path.isdir(gif_dir):
        os.makedirs(gif_dir)

    if data is None:
        data, labels = get_penguins()
    else:
        if data_path is None:
            data_path = directory
        else:
            data_path = os.path.join(directory, data_path)
        if name is None:
            raise KeyError("Please include the name of your data. Your data should be called 'data_<name>.csv'")
        data = np.loadtext(data_path + f"data_{name}.csv", dtype=float)

    if epochs is None:
        reducer = umap.UMAP(random_state=random_state, force_no_optimization=force_no_optimization,
                            n_components=dim)
    else:
        reducer = umap.UMAP(random_state=random_state, force_no_optimization=force_no_optimization,
                            n_components=dim, n_epochs=epochs)

    embedding = reducer.fit_transform(data)

    unique_labels = list(set(labels))
    labels_dict = {unique_labels[i]: i for i in range(len(unique_labels))}
    if isinstance(epochs, list):
        intermediate_embeddings = reducer.embedding_list_

        for k in range(len(epochs)):
            # Create image
            plt.figure(figsize=(7, 6))
            plt.scatter(intermediate_embeddings[k][:, 0],
                        intermediate_embeddings[k][:, 1], label='Samples', marker="o",
                        c=[sns.color_palette()[x] for x in list(map(lambda x: labels_dict[x], labels))])
            # plt.grid()
            plt.legend(loc='upper right')
            plt.title(f"Epoch {epochs[k]}")
            plt.savefig(img_dir + f"epoch_{epochs[k]}.png")

    # Create GIF
        frames = []
        imgs = glob.glob(img_dir + "*.png")
        for i in imgs:
            new_frame = Image.open(i)
            frames.append(new_frame)
        frames[0].save(gif_dir + f'{name}.gif', format='GIF',
                       append_images=frames[1:],
                       save_all=True,
                       duration=350*len(epochs), loop=0)
    else:
        plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=[sns.color_palette()[x] for x in data.species.map({"Adelie": 0, "Chinstrap": 1, "Gentoo": 2})])
        plt.gca().set_aspect('equal', 'datalim')
        plt.title('UMAP projection of the Penguin dataset', fontsize=24)
        plt.show()


if __name__ == "__main__":
    make_graph(epochs=[0, 10, 20, 30], name="penguins", data_path="penguins")
