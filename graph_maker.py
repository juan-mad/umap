import os
import umap
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from PIL import Image
import glob



from get_data import get_penguins


def make_graph(data=None, init=None, labels=None, name=None, data_path=None, epochs=None, random_state=42,
               force_no_optimization=False, dim=2, cmap="viridis", duration=None):
    # Example data in here

    # Set up directories for image saving
    directory = os.path.dirname(__file__)
    if name is None:
        res_dir = os.path.join(directory, "graphs_no_name/")
    else:
        res_dir = os.path.join(directory, "graphs_" + name + "/")
    if not os.path.isdir(res_dir):
        os.makedirs(res_dir)
    img_dir = os.path.join(res_dir, "images/")
    if not os.path.isdir(img_dir):
        os.makedirs(img_dir)
    gif_dir = os.path.join(res_dir, "gif/")
    if not os.path.isdir(gif_dir):
        os.makedirs(gif_dir)

    if name is None:
        data, labels = get_penguins()
    else:
        if data_path is None:
            data_path = directory
        else:
            data_path = os.path.join(directory, "data/" + data_path)
        if name is None:
            raise KeyError("Please include the name of your data. Your data should be called 'data_<name>.csv'")
        data = np.loadtxt(data_path + f"data_{name}.csv", dtype=float)
        labels = np.loadtxt(data_path + f"labels_{name}.csv")

    if epochs is None:
        reducer = umap.UMAP(random_state=random_state, force_no_optimization=force_no_optimization,
                            n_components=dim, init=init)
    else:
        reducer = umap.UMAP(random_state=random_state, force_no_optimization=force_no_optimization,
                            n_components=dim, n_epochs=epochs, init=init)

    embedding = reducer.fit_transform(data)

    unique_labels = list(np.unique(labels))
    labels_dict = {unique_labels[i]: i for i in range(len(unique_labels))}
    print(labels_dict)
    if isinstance(epochs, list):
        intermediate_embeddings = reducer.embedding_list_
        c = [sns.color_palette()[x] for x in list(map(lambda x: labels_dict[x], list(labels)))]
        for k in range(len(epochs)):
            # Create image

            fig, ax = plt.subplots()
            scatter = ax.scatter(x=intermediate_embeddings[k][:, 0], y=intermediate_embeddings[k][:, 1],
                                 c=labels, cmap=plt.get_cmap(cmap))
            legend1 = ax.legend(*scatter.legend_elements(), title="Classes", loc="upper right", bbox_to_anchor=(0,1))
            ax.add_artist(legend1)

            # plt.figure(figsize=(7, 6))
            # scatter = plt.scatter(intermediate_embeddings[k][:, 0],
            #                       intermediate_embeddings[k][:, 1], label="s", marker="o",
            #                       c=c)
            # print(list(map(lambda x: labels_dict[x], list(labels))))
            # # plt.grid()
            # plt.legend(loc='upper right', handles=scatter.legend_elements()[0], title="Labels")
            plt.title(f"Epoch {epochs[k]}")
            plt.savefig(img_dir + f"epoch_{epochs[k]}.png")
            plt.close()

        # Create GIF
        frames = []
        imgs = glob.glob(img_dir + "*.png")
        for i in imgs:
            new_frame = Image.open(i)
            frames.append(new_frame)
        if duration is None:
            duration = 350 * len(epochs)

        frames[0].save(gif_dir + f'{name}.gif', format='GIF',
                       append_images=frames[1:],
                       save_all=True,
                       duration=duration, loop=0)
    else:
        plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=[sns.color_palette()[x] for x in list(map(lambda x: labels_dict[x], labels))])
        plt.gca().set_aspect('equal', 'datalim')
        plt.title('UMAP projection of the Penguin dataset', fontsize=24)
        plt.show()
        plt.close()


if __name__ == "__main__":
    make_graph(epochs=list(range(0, 201, 10)), name="digits", data_path="digits/", init="random", cmap="tab10",
               duration=300)
