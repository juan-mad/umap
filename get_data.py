import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import *

DIR = os.path.dirname(__file__)


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


def save_digits(data_path = "data"):
    data, labels = load_digits(return_X_y=True)
    path = os.path.join(DIR, data_path)
    path = os.path.join(path, "digits/")
    if not os.path.isdir(path):
        os.makedirs(path)
    np.savetxt(path + "data_digits.csv", data)
    np.savetxt(path + "labels_digits.csv", labels, fmt="%d")


if __name__ == "__main__":
    save_digits()
