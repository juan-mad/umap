import pandas as pd
import numpy as np
import umap
import pickle
import os, sys
import logging
import warnings
from time import time, sleep
from fitsne.fast_tsne import fast_tsne
import bhtsne.bhtsne as bhtsne


datasets = ["cylinder",
            "gaussian_clusters_plane"
            "half_cylinder",
            "hilbert",
            "punto_silla",
            "shapes",
            "shapes_noise",
            "swiss_roll",
            "torus"]


def run_experiment_on_dataset(ds_name, experiment_name, embdim=2):
    random_state = 42

    data_path = os.getcwd() + f"/synth_data_gen/synth_datasets/{ds_name}/"
    exp_path = os.getcwd() + f"/synth_data_gen/experiments/{ds_name}/"
    logfile_path = os.getcwd() + f"/synth_data_gen/experiments/{ds_name}/log_{experiment_name}.log"
    out_path = os.getcwd() + f"/synth_data_gen/experiments/{ds_name}/{experiment_name}"

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    ########################################################

    # Set up logger
    logging.basicConfig(filename=logfile_path, level=logging.INFO,
                        format='%(asctime)s - %(levelname)-10s - %(message)s')
    logging.info(f"Beginning experiment {experiment_name}")
    logging.info(f"Dataset type: {ds_name}")

    # Get parameters of all dataset instances
    params_df = pd.read_csv(data_path + "params.csv")

    times = dict()

    # Load experiment set
    with open(f'synth_data_gen/experiments/{experiment_name}.pickle', 'rb') as handle:
        exp = pickle.load(handle)

    # For each of the dataset instance
    for idx in params_df.index:
        with open(f'synth_data_gen/synth_datasets/{ds_name}/set_{idx}.pickle', 'rb') as handle:
            high = pickle.load(handle)
        print(f"STARTING WITH DATASET NUMBER {idx}")
        logging.info(f"STARTING WITH DATASET NUMBER {idx}")
        # for each HP set
        for i, HP in exp.items():
            with warnings.catch_warnings(record=True) as w:
                logging.info(f"Starting UMAP with HP set number {i}")
                logging.info(f"HPs: {str(HP)}")
                print(f"Starting UMAP with HP set number {i}")
                print(f"HPs: {str(HP)}")

                try:
                    # Apply UMAP
                    reducer = umap.UMAP(**HP, random_state=random_state, n_jobs=-1, n_components=embdim)
                    start = time()
                    embedding = reducer.fit_transform(high["data"])
                    end = time()
                    it_took = end - start

                    # Report time
                    logging.info(f"Finished in {it_took}")
                    print(f"Finished in {it_took}")
                    times[(f"set_{idx}", "HP_i")] = it_took

                    # Save embedding with labels/ordering/etc
                    out = dict()
                    for key in high.keys():
                        if key == "data":
                            out[key] = embedding
                        else:
                            out[key] = high[key]
                    with open(f'synth_data_gen/experiments/{ds_name}/{experiment_name}/set_{idx}_HP_{i}',
                              'wb') as handle:
                        pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    print("Data saved")
                    logging.info("Data saved")

                except:
                    print("UMAP failed")
                    logging.info("UMAP failed")
                    times[(f"set_{idx}", "HP_i")] = -1

                for warning in w:
                    sleep(0.1)
                    logging.warning(str(warning.message))
                sleep(0.1)
        with open(f'synth_data_gen/experiments/{ds_name}/{experiment_name}/times.pickle',
                  'wb') as handle:
            pickle.dump(times, handle, protocol=pickle.HIGHEST_PROTOCOL)



def run_experiment_on_dataset_fitsne(ds_name, experiment_name, embdim=2):
    random_state = 42

    data_path = os.getcwd() + f"/synth_data_gen/synth_datasets/{ds_name}/"
    exp_path = os.getcwd() + f"/synth_data_gen/experiments/{ds_name}/"
    logfile_path = os.getcwd() + f"/synth_data_gen/experiments/{ds_name}/log_{experiment_name}_fitsne.log"
    out_path = os.getcwd() + f"/synth_data_gen/experiments/{ds_name}/{experiment_name}"

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    ########################################################

    # Set up logger
    logging.basicConfig(filename=logfile_path, level=logging.INFO,
                        format='%(asctime)s - %(levelname)-10s - %(message)s')
    logging.info(f"Beginning experiment {experiment_name}")
    logging.info(f"Dataset type: {ds_name}")

    # Get parameters of all dataset instances
    params_df = pd.read_csv(data_path + "params.csv")

    times = dict()

    # Load experiment set
    with open(f'synth_data_gen/experiments/{experiment_name}.pickle', 'rb') as handle:
        exp = pickle.load(handle)

    # For each of the dataset instance
    for idx in params_df.index:
        with open(f'synth_data_gen/synth_datasets/{ds_name}/set_{idx}.pickle', 'rb') as handle:
            high = pickle.load(handle)
        print(f"STARTING WITH DATASET NUMBER {idx}")
        logging.info(f"STARTING WITH DATASET NUMBER {idx}")
        # for each HP set
        for i, HP in exp.items():
            with warnings.catch_warnings(record=True) as w:
                logging.info(f"Starting FItsne with HP set number {i}")
                logging.info(f"HPs: {str(HP)}")
                print(f"Starting FItsne with HP set number {i}")
                print(f"HPs: {str(HP)}")

                try:
                    start = time()
                    embedding = fast_tsne(X=high["data"], seed=42,
                                          map_dims=embdim, **HP)
                    end = time()
                    it_took = end - start

                    # Report time
                    logging.info(f"Finished in {it_took}")
                    print(f"Finished in {it_took}")
                    times[(f"set_{idx}", "HP_i")] = it_took

                    # Save embedding with labels/ordering/etc
                    out = dict()
                    for key in high.keys():
                        if key == "data":
                            out[key] = embedding
                        else:
                            out[key] = high[key]
                    with open(f'synth_data_gen/experiments/{ds_name}/{experiment_name}/set_{idx}_HP_{i}',
                              'wb') as handle:
                        pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    print("Data saved")
                    logging.info("Data saved")

                except Exception as e:
                    print("FItsne failed")
                    logging.info("FItsne failed")
                    sleep(0.1)
                    print(e)
                    logging.info(e)
                    times[(f"set_{idx}", "HP_i")] = -1

                for warning in w:
                    sleep(0.1)
                    logging.warning(str(warning.message))
                sleep(0.1)
        with open(f'synth_data_gen/experiments/{ds_name}/{experiment_name}/times.pickle',
                  'wb') as handle:
            pickle.dump(times, handle, protocol=pickle.HIGHEST_PROTOCOL)
def run_experiment_on_dataset_bhtsne(ds_name, experiment_name, embdim=2):
    random_state = 42

    data_path = os.getcwd() + f"/synth_data_gen/synth_datasets/{ds_name}/"
    exp_path = os.getcwd() + f"/synth_data_gen/experiments/{ds_name}/"
    logfile_path = os.getcwd() + f"/synth_data_gen/experiments/{ds_name}/log_{experiment_name}_fitsne.log"
    out_path = os.getcwd() + f"/synth_data_gen/experiments/{ds_name}/{experiment_name}"

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    ########################################################

    # Set up logger
    logging.basicConfig(filename=logfile_path, level=logging.INFO,
                        format='%(asctime)s - %(levelname)-10s - %(message)s')
    logging.info(f"Beginning experiment {experiment_name}")
    logging.info(f"Dataset type: {ds_name}")

    # Get parameters of all dataset instances
    params_df = pd.read_csv(data_path + "params.csv")

    times = dict()

    # Load experiment set
    with open(f'synth_data_gen/experiments/{experiment_name}.pickle', 'rb') as handle:
        exp = pickle.load(handle)

    # For each of the dataset instance
    for idx in params_df.index:
        with open(f'synth_data_gen/synth_datasets/{ds_name}/set_{idx}.pickle', 'rb') as handle:
            high = pickle.load(handle)
        print(f"STARTING WITH DATASET NUMBER {idx}")
        logging.info(f"STARTING WITH DATASET NUMBER {idx}")
        # for each HP set
        for i, HP in exp.items():
            with warnings.catch_warnings(record=True) as w:
                logging.info(f"Starting Barnes Hut t-SNE with HP set number {i}")
                logging.info(f"HPs: {str(HP)}")
                print(f"Starting Barnes Hut t-SNE with HP set number {i}")
                print(f"HPs: {str(HP)}")

                try:
                    start = time()
                    embedding = bhtsne.run_bh_tsne(high["data"], **HP, randseed=random_state,
                                                   initial_dims=high["data"].shape[1],
                                                   no_dims=embdim)

                    end = time()
                    it_took = end - start

                    # Report time
                    logging.info(f"Finished in {it_took}")
                    print(f"Finished in {it_took}")
                    times[(f"set_{idx}", "HP_i")] = it_took

                    # Save embedding with labels/ordering/etc
                    out = dict()
                    for key in high.keys():
                        if key == "data":
                            out[key] = embedding
                        else:
                            out[key] = high[key]
                    with open(f'synth_data_gen/experiments/{ds_name}/{experiment_name}/set_{idx}_HP_{i}',
                              'wb') as handle:
                        pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    print("Data saved")
                    logging.info("Data saved")

                except Exception as e:
                    print("FItsne failed")
                    logging.info("FItsne failed")
                    sleep(0.1)
                    print(e)
                    logging.info(e)
                    times[(f"set_{idx}", "HP_i")] = -1

                for warning in w:
                    sleep(0.1)
                    logging.warning(str(warning.message))
                sleep(0.1)
        with open(f'synth_data_gen/experiments/{ds_name}/{experiment_name}/times.pickle',
                  'wb') as handle:
            pickle.dump(times, handle, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    
    ds_name = sys.argv[1]
    experiment_name = sys.argv[2]
    if len(sys.argv) >= 4:
        embdim = int(sys.argv[3])
    else:
        embdim = 2

    is_fitsne = False
    is_bhtsne = False
    if len(sys.argv) >= 5:
        is_fitsne = sys.argv[4] == "fitsne"
        is_bhtsne = sys.argv[4] == "bhtsne"

    if is_fitsne:
        run_experiment_on_dataset_fitsne(ds_name, experiment_name, embdim)
    elif is_bhtsne:
        run_experiment_on_dataset_bhtsne(ds_name, experiment_name, embdim)
    else:
        run_experiment_on_dataset(ds_name, experiment_name, embdim)
