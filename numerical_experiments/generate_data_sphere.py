"""

"""
import numpy as np
from numpy.random import default_rng
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.spatial.transform import Rotation as R


def generate_tennisball(sample_size=200, radius=1, a=None, b=None, seed=None, noise=0.01):
    """ Returns the curve of a tennisball as well as the sample in parameter space

    Args:
        sample_size (int): number of points to generate
        radius (float): radius of the sphere
        a (float): parameter for the curve. Must verify a+b=radius
        b (float): parameter for the curve. Must verify a+b=radius
        seed (float): random state seed

    Returns:
        curve: numpy array with the sample inside the curve
        sample: points in parameter space

    """
    rng = default_rng(seed=seed)

    sample = rng.random(sample_size) * 2 * np.pi

    if a and b:
        print(f"Ignoring radius value -- new value is a+b={a + b}")
        radius = a + b
    if a and not b:
        b = radius - a
    if b and not a:
        a = radius - b
    if not a and not b:
        a = 0.75
        b = radius - a

    c = 2 * np.sqrt(a * b)

    curve = np.zeros((sample_size, 3))
    curve[:, 0] = a * np.cos(sample) + b * np.cos(3 * sample)
    curve[:, 1] = a * np.sin(sample) - b * np.sin(3 * sample)
    curve[:, 2] = c * np.sin(2 * sample)

    if noise:
        curve = add_noise(curve, noise, radius, rng)

    return curve, sample


def generate_noise_in_sphere(sample_size, radius=1, how="normal_projection", scale=0.01, seed=None, spheres=1):
    rng = np.random.default_rng(seed=seed)

    def gen_normal_proj(scale, sample_size, radius):
        sample = rng.normal(scale=scale, size=(sample_size, 3))
        sample = radius * sample / np.linalg.norm(sample, 2, 1).reshape((sample_size, 1))
        return sample

    if how == "normal_projection":
        if spheres == 1:
            return gen_normal_proj(scale, sample_size, radius)
        else:
            return np.hstack([gen_normal_proj(scale, sample_size, radius) for _ in range(spheres)])


def generate_rhumb(sample_size=50, radius=1, theta=0, phi=0, beta=1, seed=None, noise=0.01):
    print("Warning: not uniformly sampled along the curve!")
    rng = default_rng(seed=seed)

    def gdinv(x):
        return 2 * np.arctanh(np.tan(0.5 * x))

    sample = rng.random(sample_size) * np.pi * 2 - np.pi
    sample = np.linspace(-2 * np.pi, 2 * np.pi, sample_size)
    curve = np.zeros((sample_size, 3))

    psi = (sample - theta) * (1 / np.tan(beta)) + gdinv(phi)

    sech = 1 / np.cosh(psi)

    curve[:, 0] = radius * np.cos(sample) * sech
    curve[:, 1] = radius * np.sin(sample) * sech
    curve[:, 2] = radius * np.tanh(psi)

    if noise:
        curve = add_noise(curve, noise, radius, rng)
    return curve, sample


def generate_satellite_curve(sample_size=500, a=1 / np.sqrt(2), b=1 / np.sqrt(2),
                             alpha=np.pi / 2, num=1, dem=2, seed=None, noise=0.01):
    """Generates a satellite curve
    Generates a satellite curve as described in https://mathcurve.com/courbes3d.gb/satellite/satellite.shtml
    The radius of the corresponding sphere is sqrt(b**2 + a**2 / sin(alpha)**2)

    Args:
        sample_size (int): number of points to generate
        a: parameter
        b: parameter
        alpha: parameter, different from 0
        num: numerator of k
        dem: denominator of k
        seed: random seed

    Returns:
        curve: numpy array with the sample inside the curve
        sample: points in parameter space
        radius: radius of the sphere where the curve lies
    """
    if alpha == 0:
        raise ZeroDivisionError("In this implementation alpha must be non-zero")

    rng = default_rng(seed=seed)
    k = num / dem
    sample = rng.random(sample_size) * 2 * np.pi * 2
    curve = np.zeros((sample_size, 3))

    cosalpha = np.cos(alpha)
    coskt = np.cos(k * sample)
    sinkt = np.sin(k * sample)
    cost = np.cos(sample)
    sint = np.sin(sample)
    aux = a + b * cosalpha * coskt
    curve[:, 0] = aux * cost - b * sinkt * sint
    curve[:, 1] = aux * sint + b * sinkt * cost
    curve[:, 2] = b * np.sin(alpha) * coskt - a / np.tan(alpha)

    radius = np.sqrt(b ** 2 + (a / np.sin(alpha)) ** 2)

    if noise:
        curve = add_noise(curve, noise, radius, rng)
    return curve, sample, radius


def generate_spherical_helix(sample_size=200, seed=None, radius=1, radius_fixed_num=1, radius_fixed_dem=3, noise=0.01):
    print("Warning: not uniformly sampled along the curve?!")
    rng = default_rng(seed=seed)
    sample = rng.random(sample_size) * np.pi * 10
    curve = np.zeros((sample_size, 3))
    radius_fixed = radius_fixed_num / radius_fixed_dem
    k = radius_fixed / radius

    cost = np.cos(sample)
    sint = np.sin(sample)
    coskt = np.cos(k * sample)
    sinkt = np.sin(k * sample)

    curve[:, 0] = radius * (k * cost * coskt + sint * sinkt)
    curve[:, 1] = radius * (k * sint * coskt - cost * sinkt)
    curve[:, 2] = radius * np.sqrt(1 - k ** 2) * coskt

    if noise:
        curve = add_noise(curve, noise, radius, rng)
    return curve, sample


def add_noise(curve, noise, radius, rng):
    curve = curve + rng.normal(scale=noise, size=curve.shape)
    curve = radius * curve / np.linalg.norm(curve, 2, 1).reshape((curve.shape[0], 1))

    return curve


def circle(sample_size=200, seed=None, latitude=0.5, noise=0.01, radius=1, rotate_y=0, rotate_z=0, degrees=True):
    """Creates a small circle on the sphere
    First a circle is created with the given parameters with its center in the z-axis. Then it is rotated,
    first along the y-axis (centre moves in the ZX plane) and then along the z-axis (in the XY plane)
    Args:
        sample_size (int): number of points to generate
        seed: random seed
        latitude: angle that the circle's position will have with respect to its parallel equator
            (i.e. latitude)
        radius: radius of the sphere where the circle lies
        rotate_y: angle of rotation around y-axis
        rotate_z: angle of rotation around z-axis
        degrees (bool): whether to use degrees (True) or radians (False) for the rotation

    Returns:
        curve: numpy array with the sample inside the curve
        sample: points in parameter space
    """
    rng = default_rng(seed=seed)
    sample = rng.random(sample_size) * np.pi * 2
    sample = np.linspace(0, 2 * np.pi, sample_size)
    curve = np.zeros((sample_size, 3))

    cos_lat = np.cos(latitude)
    sin_lat = np.sin(latitude)
    curve[:, 0] = radius * cos_lat * np.cos(sample)
    curve[:, 1] = radius * cos_lat * np.sin(sample)
    curve[:, 2] = radius * sin_lat

    rotation = R.from_euler("yz", [rotate_y, rotate_z], degrees=degrees)
    curve = rotation.apply(curve)

    # Apply noise
    # curve = np.apply_along_axis(lambda x: rng.multivariate_normal(x, cov=noise*np.ones((3,3))), 1, curve)
    if noise:
        curve = add_noise(curve, noise, radius, rng)

    return curve, sample


def generate_sphere(radius, detail=30):
    det = detail * 1j
    u, v = np.mgrid[0:2 * np.pi:2 * det, 0:np.pi:det]

    sphere = np.zeros((detail * 2, detail, 3))
    x = radius * np.cos(u) * np.sin(v)
    y = radius * np.sin(u) * np.sin(v)
    z = radius * np.cos(v)
    return x, y, z


def plot_3d_curve(curve, sample, over_sphere=True, radius=1):
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    ax.scatter3D(curve[:, 0], curve[:, 1], curve[:, 2], c=sample, cmap="rainbow")

    if over_sphere:
        x, y, z = generate_sphere(radius)
        ax.plot_wireframe(x, y, z, alpha=0.1, color="black")

    return fig, ax


def plot_3d_points_over_fig(points, fig, ax, color="black"):
    ax.scatter3D(points[:, 0], points[:, 1], points[:, 2], c=color)
    return fig, ax


if __name__ == "__main__":
    radius = 1
    # curve, sample = circle(latitude=np.pi / 3, rotate_y=30, rotate_z=60, degrees=True)

    # curve, sample = generate_tennisball(sample_size=200, radius=radius, noise=0)

    curve, sample = generate_rhumb(sample_size=200)

    fig, ax = plot_3d_curve(curve, sample, radius=radius)

    # fig, ax = plot_3d_points_over_fig(generate_noise_in_sphere(200, radius, ), fig, ax)

    plt.show()
    plt.close()
