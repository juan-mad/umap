import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.spatial.transform import Rotation as R
from sklearn.model_selection import ParameterGrid
import os, sys
from time import sleep, time
import logging
import warnings

import umap


def add_noise(manifold, noise, rng):
    return manifold + rng.normal(scale=noise, size=manifold.shape)


def add_noise_in_other_dim(manifold, noise, rng, dims):
    if isinstance(noise, list) and len(noise) != dims:
        print("Length of noise does not match the number of dimensions specified")

    n_samples, n_features = manifold.shape
    expanded_manifold = np.empty((n_samples, n_features + dims))
    expanded_manifold[:, :n_features] = manifold
    if isinstance(noise, list):
        expanded_manifold[:, n_features:] = rng.normal(size=n_samples,
                                                       mean=np.zeros(dims),
                                                       cov=np.diag(noise))
    else:
        expanded_manifold[:, n_features:] = rng.normal(scale=noise, size=(n_samples, dims))

    return expanded_manifold


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
    """Plots 3D curve as a scatterplot
    Args:
        curve: curve to plot
        sample: points in sample space for coloring
        fig: figure object
        ax: axis object
        cmap: colormap

    Returns:
        fig, ax
    """
    if fig is None or ax is None:
        fig = plt.figure()
        ax = plt.axes(projection="3d")

    ax.scatter3D(curve[:, 0], curve[:, 1], curve[:, 2], c=sample, cmap=cmap)

    return fig, ax


def plot_2d_curve(curve, sample, fig=None, ax=None, cmap="hsv"):
    """Plots 2D curve as a scatterplot
    Args:
        curve: curve to plot
        sample: points in sample space for coloring
        fig: figure object
        ax: axis object
        cmap: colormap

    Returns:
        fig, ax
    """
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


def generate_2d_circle(sample_size: object = 200, a: object = 1, seed: object = None, noise: object = 0.01) -> object:
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


def generate_embeddings(params, manifold, return_umap=False):
    """
    Trains a UMAP object with parameters contained in dict "params" and "data", and returns the embedding(s)
    Args:
        params (dict): Dictionary with parameters to pass to UMAP.
            If `n_epochs` is one of them and is a list, a list with the embeddings corresponding to the epochs
            specified in the list is returned.
        manifold: Numpy array of the form (n_samples, n_features) to embed into lower dimension.
        return_umap (bool)
    Returns:
        embedding(s) [, UMAP object]
    """
    print("Working on: " + str(params))
    logging.info("Working on: " + str(params))
    try:
        reducer = umap.UMAP(**params)
        start = time()
        embedding = reducer.fit_transform(manifold)
        end = time()
        it_took = end - start
        print(f"Finished in {it_took} s\n")
        logging.info(f"Finished in {it_took} s\n")

        if "n_epochs" in params.keys():
            if isinstance(params["n_epochs"], list):
                if return_umap:
                    return reducer.embedding_list_, reducer
                else:
                    return reducer.embedding_list_
        if return_umap:
            return embedding, reducer
        else:
            return embedding

    except Exception as e:
        # print(f"Encountered exception: {e}")
        logging.error(f"Encountered exception: {e}; skipping")
        sleep(0.5)
        # print("--- skipping ---\n")
        sleep(0.5)
        return None
