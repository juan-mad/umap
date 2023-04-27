# from generate_data_sphere import *
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.spatial.transform import Rotation as R
from sklearn.model_selection import ParameterGrid
import os, sys
from time import time

from os import listdir
from os.path import isfile, join

parent = os.path.abspath('.')
sys.path.insert(1, parent)

def plot_3d_curve(curve, sample, fig=None, ax=None, cmap="hsv"):
    if fig is None or ax is None:
        fig = plt.figure()
        ax = plt.axes(projection="3d")

    ax.scatter3D(curve[:, 0], curve[:, 1], curve[:, 2], c=sample, cmap=cmap)

    return fig, ax


def plot_2d_curve(curve, sample, fig=None, ax=None, cmap="hsv"):
    if fig is None or ax is None:
        fig = plt.figure(figsize=(10,10))
        ax = plt.axes()

    ax.scatter(curve[:, 0], curve[:, 1], c=sample, cmap=cmap, s=2)

    return fig, ax


if __name__ == "__main__":

    experiment_name = sys.argv[1]
    my_path = "./" + experiment_name + "/"
    if not os.path.exists(my_path + "images/"):
        os.makedirs(my_path + "images/")
    image_path = my_path + "images/"

    onlyfiles = [f for f in listdir(my_path) if isfile(join(my_path, f))]
    sample = np.load(my_path + "sample.npy")
    count = 0

    for f in onlyfiles:
        if f[-4:] != ".npy":
            print("Found a file that is not .npy: "+ f)
            break
        if f == "sample.npy":
            continue
        else:
            data = np.load(my_path + f)
            if data.shape[1] == 2:
                fig, ax = plot_2d_curve(data, sample)
                fig.savefig(image_path + f[:-4] + ".png")
                plt.close()
                count += 1
            if data.shape[1] == 3:
                fig, ax = plot_3d_curve(data, sample)
                fig.savefig(image_path + f[:-4] + ".png")
                plt.close()
                count += 1
        if count % 20 == 0:
            print(f"Printed {count} images")
