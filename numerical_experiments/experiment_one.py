# from generate_data_sphere import *
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.spatial.transform import Rotation as R
from sklearn.model_selection import ParameterGrid
import os, sys
from time import time
import pyDOE
import rpy2
import logging
import warnings

import umap


parent = os.path.abspath('.')
sys.path.insert(1, parent)


def add_noise(curve, noise, rng):
    return curve + rng.normal(scale=noise, size=curve.shape)


def get_rng(seed_or_rng):
    if isinstance(seed_or_rng, np.random._generator.Generator):
        return seed_or_rng
    else:
        return default_rng(seed=seed_or_rng)


def rotate(curve, angle_y=None, angle_z=None, degrees=True, seed=None):
    """
    Args:
        curve: curve to rotate
        angle_y: angle of rotation around y-axis
        angle_z: angle of rotation around z-axis
        degrees (bool): whether to use degrees (True) or radians (False) for the rotation
        seed: random
    Returns:
        Rotated curve
    """
    rng = get_rng(seed)

    factor = 360.0 if degrees else 2 * np.pi
    if angle_y is None:
        angle_y = factor * rng.random()
    if angle_z is None:
        angle_z = factor * rng.random()

    rotation = R.from_euler("yz", [angle_y, angle_z], degrees=degrees)
    return rotation.apply(curve)


def plot_3d_curve(curve, sample, fig=None, ax=None, cmap="hsv"):
    if fig is None or ax is None:
        fig = plt.figure()
        ax = plt.axes(projection="3d")

    ax.scatter3D(curve[:, 0], curve[:, 1], curve[:, 2], c=sample, cmap=cmap)

    return fig, ax


def plot_2d_curve(curve, sample, fig=None, ax=None, cmap="hsv"):
    if fig is None or ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes()

    ax.scatter(curve[:, 0], curve[:, 1], c=sample, cmap=cmap, s=2)

    return fig, ax


def generate_2d_astroid(sample_size=200, a=1, seed=None, noise=0.01):
    """ Returns the curve of a 2d astroid as well as the sample in parameter space
    Args:
        sample_size (int): number of points to generate
        a (float): size parameter
        seed : random state seed, or np.random._generator.Generator
        noise: std of gaussian noise

    Returns:
        curve: numpy array with the sample inside the curve
        sample: points in parameter space

    """
    rng = get_rng(seed)

    sample = rng.random(sample_size) * 2 * np.pi
    curve = np.zeros((sample_size, 2))
    curve[:, 0] = a * np.cos(sample) ** 3
    curve[:, 1] = a * np.sin(sample) ** 3

    if noise:
        curve = add_noise(curve, noise, rng)

    return curve, sample


def generate_2d_involute(sample_size=200, a=1, loops=10, seed=None, noise=0.01):
    rng = get_rng(seed)
    sample = rng.random(sample_size) * 2 * np.pi * loops
    curve = np.zeros((sample_size, 2))
    curve[:, 0] = a * (np.cos(sample) + sample * np.sin(sample))
    curve[:, 1] = a * (np.sin(sample) - sample * np.cos(sample))

    if noise:
        curve = add_noise(curve, noise, rng)

    return curve, sample


# def generate_3d_astroid(sample_size=200, a=1, seed=None, noise=0.01, angle_y=None, angle_z=None, degrees=True):
#     print("Remember not to call this function if you already added noise to the 2D curve!")
#     rng = get_rng(seed)

#     curve = np.zeros((sample_size, 3))
#     curve[:, :2], sample = generate_2d_astroid(sample_size=sample_size,
#                                                a=a,
#                                                seed=rng,
#                                                noise=noise)
#     curve = rotate(curve, angle_y=angle_y, angle_z=angle_z, degrees=degrees)

#     return curve, sample


def generate_2d_circle(sample_size=200, a=1, seed=None, noise=0.01):
    """ Returns the curve of a circle as well as the sample in parameter space
    Args:
        sample_size (int): number of points to generate
        a (float): radius
        seed : random state seed, or np.random._generator.Generator
        noise: std of gaussian noise

    Returns:
        curve: numpy array with the sample inside the curve
        sample: points in parameter space

    """
    rng = get_rng(seed)

    sample = rng.random(sample_size) * 2 * np.pi
    curve = np.zeros((sample_size, 2))
    curve[:, 0] = a * np.cos(sample)
    curve[:, 1] = a * np.sin(sample)

    if noise:
        curve = add_noise(curve, noise, rng)

    return curve, sample


def generate_3d_curve_from_2d(curve, seed=None, angle_y=None, angle_z=None, degrees=True, noise=0.01):
    print("Remember not to call this function if you already added noise to the 2D curve!")
    rng = get_rng(seed)

    new_curve = np.zeros((curve.shape[0], 3))
    new_curve[:, :2] = curve
    new_curve = rotate(new_curve, angle_y=angle_y, angle_z=angle_z, degrees=degrees)

    if noise:
        new_curve = add_noise(new_curve, noise, rng)
    return new_curve


def generate_embeddings(params, data):
    print("Working on: " + str(params))
    logging.info("Working on: " + str(params))
    try:
        reducer = umap.UMAP(**params)
        start = time()
        embedding = reducer.fit_transform(data)
        end = time()
        it_took = end - start
        print(f"Finished in {it_took} s\n")
        logging.info(f"Finished in {it_took} s\n")

        if "n_epochs" in params.keys():
            if isinstance(params["n_epochs"], list):
                return reducer.embedding_list_

        return embedding
    except Exception as e:
        print(f"Encountered exception: {e}")
        logging.error(f"Encountered exception: {e}; skipping")
        print("--- skipping ---\n")
        return None


def exp_one(exp_name, sample_size, curve_type):
    rng = default_rng(seed=42)

    if curve_type == "astroid":
        curve, sample = generate_2d_astroid(seed=rng, noise=0.2, a=30, sample_size=sample_size)
    elif curve_type == "involute":
        curve, sample = generate_2d_involute(seed=rng, a=0.2, loops=3, sample_size=sample_size, noise=0)
    else:
        print("Introduce valid curve type")
        return None
    # fig, ax = plot_2d_curve(curve, sample)
    # plt.show()

    # newcurve = generate_3d_curve_from_2d(curve, seed=rng)
    # print(newcurve.shape)
    # fig, ax = plot_3d_curve(newcurve, sample)
    # plt.show()

    n_neighbors = 15
    n_epochs = None
    learning_rate = 1.0
    min_dist = 0.1
    spread = 1
    random_state = 42

    n_neighbors_list = list(range(5, 32, 10))
    # n_epochs_list = list(range(1000, 10001, 1000))
    # learning_rate_list = list(np.arange(0.2, 2.1, 0.2))
    min_dist_list = list(np.arange(0.1, 1.1, 0.2))
    spread_list = list(np.arange(0.2, 2.1, 0.4))

    param_grid = ParameterGrid({
        "n_neighbors": n_neighbors_list,
        "min_dist": min_dist_list,
        "spread": spread_list,
        "random_state": [random_state]
    })

    np.save(exp_name + "/2d_curve.npy", curve)
    # np.save(exp_name + "/3d_curve.npy", newcurve)
    np.save(exp_name + "/sample.npy", sample)

    for params in list(param_grid):
        embedding = generate_embeddings(params, curve)
        if embedding is not None:
            np.save(exp_name + "/" + str(params) + ".npy", embedding)
    print("Finished all")


if __name__ == "__main__":
    experiment_name = 0
    sample_size = 0
    curve_type = 0

    exp_one("aaa", 100, "astroid")
