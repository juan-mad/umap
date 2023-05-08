import pickle

from experiment_functions import *

# EXP_NAME = "experiment_" + datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
EXP_NAME = "experiment_circle_a_1_sample_1000_spectral"
RESULT_PATH = "results/"
SAMPLE_SIZE = 1000
RADIUS = 1


def run_experiment():
    """"""
    exp_path = RESULT_PATH + EXP_NAME
    log_path = exp_path + f"/log_{EXP_NAME}.txt"
    # Set up folder system
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    rng = default_rng(seed=42)
    # Create manifold to be embedded

    manifold, sample = generate_2d_circle(sample_size=SAMPLE_SIZE, a=1, seed=rng, noise=0.05)

    # Save original manifold
    np.save(exp_path + "/manifold.npy", manifold)
    if sample is not None:
        np.save(exp_path + "/sample.npy", sample)

    # Get all set of parameters to consider in the experiment as a list of dictionaries
    # Each dictionary will contain the parameter values as key-value pairs
    experiment_list = [
        {"n_neighbors": n_n} for n_n in range(5, 101, 5)
    ]  # etc

    n_epochs = list(range(0, 10001, 1000))
    init = "spectral"

    # Now we run the experiment
    # Set up log file
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s - %(levelname)-10s - %(message)s')
    logging.info(f"Beginning experiment {EXP_NAME}")
    logging.info(f"Sample size: {manifold.shape[0]}")
    for i, params in enumerate(experiment_list):
        if "n_epochs" not in params.keys():
            params["n_epochs"] = n_epochs
        if "init" not in params.keys():
            params["init"] = init

        with warnings.catch_warnings(record=True) as w:
            embeddings = generate_embeddings(params, manifold)

            for warning in w:
                logging.warning(str(warning.message))

            if embeddings is not None:
                if isinstance(n_epochs, list):
                    for j, e in enumerate(n_epochs):
                        np.save(exp_path + f"/params_{i}_epoch_{e}.npy", embeddings[j])
                else:
                    np.save(exp_path + f"/params_{i}.npy", embeddings)

    with open(exp_path + f"/list_{EXP_NAME}.txt", "w") as f:
        for i, ps in enumerate(experiment_list):
            f.write(f"Experiment {i}\n")
            for key in ps.keys():
                f.write(f"\t{key}:\t{ps[key]}\n")
            f.write("\n")
    with open(exp_path + f"/dict_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump(experiment_list, f)


def create_report():
    """"""

    exp_path = RESULT_PATH + EXP_NAME
    with open(exp_path + f"/dict_{EXP_NAME}.pkl", "rb") as f:
        experiment_list = pickle.load(f)

    epochs_list = [params["n_epochs"] for params in experiment_list]
    max_epoch_len = max(len(eps) for eps in epochs_list)

    fig = plt.figure(constrained_layout=True, figsize=(6 * max_epoch_len, 6 * len(epochs_list)))
    fig.suptitle(f"REPORT: {EXP_NAME}", fontsize=80)

    subfigs = fig.subfigures(nrows=len(experiment_list), ncols=1)

    sample = np.load(exp_path + "/sample.npy")

    for i, (params, subfig) in enumerate(zip(experiment_list, subfigs)):
        ne = params["n_epochs"]
        subfig.suptitle("Params: " + str(params), fontsize=45)

        axes = subfig.subplots(nrows=1, ncols=len(ne))
        for j, e in enumerate(ne):
            try:
                emb = np.load(exp_path + f"/params_{i}_epoch_{e}.npy")
                if emb.shape[1] != 2:
                    print("Only configured for 2D!!!")
                    raise BaseException
                fig, axes[j] = plot_2d_curve(emb, sample, fig=fig, ax=axes[j])
                axes[j].set_title(f"Epochs: {e}", fontsize=30)
            except:
                print("ERROR")
    plt.savefig(exp_path + f"/report_{EXP_NAME}.png")


if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1] == "run":
        run_experiment()

    if sys.argv[1] == "report":
        create_report()
