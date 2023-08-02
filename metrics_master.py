from metrics_util import *

import pickle


#
#
# print("GCP")
#
# experiment_names = [
#     "neighbor_init_sweep",
#     "mindist_spread_sweep",
#     "fitsne_perplexity_sweep",
#     "bhtsne_perplexity_sweep",
#     "hyp1"
# ]
#
# ds_name = "gaussian_clusters_plane"
#
# gcp_metrics = dict()
# gcp_params = None
# for en in experiment_names:
#     print(f"Metrics for experiment {en}")
#     gcp_metrics[en], gcp_params = evaluate_experiments(experiment_name=en,
#                      ds_name=ds_name,
#                      has_labels=True, verbose=2)
#
# with open(f'synth_data_gen/experiments/{ds_name}/metrics.pickle', 'wb') as handle:
#     pickle.dump(gcp_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#
#
# print("Shapes")
#
# experiment_names = [
#     "neighbor_init_sweep",
#     "mindist_spread_sweep",
#     "fitsne_perplexity_sweep",
#     "bhtsne_perplexity_sweep",
#     "hyp1"
# ]
#
# ds_name = "shapes"
#
# shapes_metrics = dict()
# shapes_params = None
# for en in experiment_names:
#     print(f"Metrics for experiment {en}")
#     shapes_metrics[en], shapes_params = evaluate_experiments(experiment_name=en,
#                      ds_name=ds_name,
#                      has_labels=True, verbose=2)
#
# with open(f'synth_data_gen/experiments/{ds_name}/metrics.pickle', 'wb') as handle:
#     pickle.dump(shapes_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Hilbert")

experiment_names = [
    "neighbor_init_sweep",
    "mindist_spread_sweep",
    "fitsne_perplexity_sweep",
    "bhtsne_perplexity_sweep",
    "hyp1"
]

ds_name = "hilbert"

hilbert_metrics = dict()
hilbert_params = None
for en in experiment_names:
    if en != "hyp1":
        print(f"Metrics for experiment {en}_2d")
        hilbert_metrics[en], hilbert_params = evaluate_experiments(experiment_name=en,
                         ds_name=ds_name,
                         has_labels=False, verbose=2, extra_name="_2d")
    else:
        print(f"Metrics for experiment {en}")
        hilbert_metrics[en], hilbert_params = evaluate_experiments(experiment_name=en,
                         ds_name=ds_name,
                         has_labels=False, verbose=2)

with open(f'synth_data_gen/experiments/{ds_name}/metrics.pickle', 'wb') as handle:
    pickle.dump(hilbert_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Punto Silla")

experiment_names = [
    "neighbor_init_sweep",
    "mindist_spread_sweep",
    "fitsne_perplexity_sweep",
    "bhtsne_perplexity_sweep",
    "hyp1"
]

ds_name = "punto_silla"

punto_silla_metrics = dict()
punto_silla_params = None
for en in experiment_names:
    print(f"Metrics for experiment {en}")
    punto_silla_metrics[en], punto_silla_params = evaluate_experiments(experiment_name=en,
                     ds_name=ds_name,
                     has_labels=False, verbose=2)

with open(f'synth_data_gen/experiments/{ds_name}/metrics.pickle', 'wb') as handle:
    pickle.dump(punto_silla_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)




print("Shapes noise")

experiment_names = [
    "neighbor_init_sweep",
    "mindist_spread_sweep",
    "fitsne_perplexity_sweep",
    "bhtsne_perplexity_sweep",
    "hyp1"
]

ds_name = "shapes_noise"

shapes_noise_metrics = dict()
shapes_noise_params = None
for en in experiment_names:
    print(f"Metrics for experiment {en}")
    shapes_noise_metrics[en], shapes_noise_params = evaluate_experiments(experiment_name=en,
                     ds_name=ds_name,
                     has_labels=True, verbose=2)

with open(f'synth_data_gen/experiments/{ds_name}/metrics.pickle', 'wb') as handle:
    pickle.dump(shapes_noise_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Sphere")

experiment_names = [
    "neighbor_init_sweep",
    "mindist_spread_sweep",
    "bhtsne_perplexity_sweep",
    "hyp1"
]

ds_name = "sphere"

sphere_metrics = dict()
sphere_params = None
for en in experiment_names:
    print(f"Metrics for experiment {en}")
    sphere_metrics[en], sphere_params = evaluate_experiments(experiment_name=en,
                     ds_name=ds_name,
                     has_labels=False, verbose=2)

with open(f'synth_data_gen/experiments/{ds_name}/metrics.pickle', 'wb') as handle:
    pickle.dump(sphere_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("sphere unif")

experiment_names = [
    "neighbor_init_sweep",
    "mindist_spread_sweep",
    "bhtsne_perplexity_sweep",
    "hyp1"
]

ds_name = "sphere_unif"

sphere_unif_metrics = dict()
sphere_unif_params = None
for en in experiment_names:
    print(f"Metrics for experiment {en}")
    sphere_unif_metrics[en], sphere_unif_params = evaluate_experiments(experiment_name=en,
                     ds_name=ds_name,
                     has_labels=False, verbose=2)

with open(f'synth_data_gen/experiments/{ds_name}/metrics.pickle', 'wb') as handle:
    pickle.dump(sphere_unif_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("torus")

experiment_names = [
    "neighbor_init_sweep",
    "mindist_spread_sweep",
    "bhtsne_perplexity_sweep",
    "hyp1"
]

ds_name = "torus"

torus_metrics = dict()
torus_params = None
for en in experiment_names:
    print(f"Metrics for experiment {en}")
    torus_metrics[en], torus_params = evaluate_experiments(experiment_name=en,
                     ds_name=ds_name,
                     has_labels=False, verbose=2)

with open(f'synth_data_gen/experiments/{ds_name}/metrics.pickle', 'wb') as handle:
    pickle.dump(torus_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)


print("mnist")

experiment_names = [
    "neighbor_init_sweep",
    "mindist_spread_sweep",
    "fitsne_perplexity_sweep",
    "hyp1"
]   

ds_name = "mnist"

mnist_metrics = dict()
mnist_params = None
for en in experiment_names:
    print(f"Metrics for experiment {en}")
    mnist_metrics[en], mnist_params = evaluate_experiments(experiment_name=en,
                     ds_name=ds_name,
                     has_labels=True, verbose=2)

with open(f'synth_data_gen/experiments/{ds_name}/metrics.pickle', 'wb') as handle:
    pickle.dump(mnist_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("swiss roll")

experiment_names = [
    "neighbor_init_sweep",
    "mindist_spread_sweep",
    "fitsne_perplexity_sweep",
    "bhtsne_perplexity_sweep",
    "hyp1"
]

ds_name = "swiss_roll"

swiss_roll_metrics = dict()
swiss_roll_params = None
for en in experiment_names:
    print(f"Metrics for experiment {en}")
    swiss_roll_metrics[en], swiss_roll_params = evaluate_experiments(experiment_name=en,
                     ds_name=ds_name,
                     has_labels=False, verbose=2)
with open(f'synth_data_gen/experiments/{ds_name}/metrics.pickle', 'wb') as handle:
    pickle.dump(swiss_roll_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)


print("half cylinder")

experiment_names = [
    "neighbor_init_sweep",
    "mindist_spread_sweep",
    "fitsne_perplexity_sweep",
    "bhtsne_perplexity_sweep",
    "hyp1"
]

ds_name = "half_cylinder"

half_cylinder_metrics = dict()
half_cylinder_params = None
for en in experiment_names:
    print(f"Metrics for experiment {en}")
    half_cylinder_metrics[en], half_cylinder_params = evaluate_experiments(experiment_name=en,
                     ds_name=ds_name,
                     has_labels=False, verbose=2)

with open(f'synth_data_gen/experiments/{ds_name}/metrics.pickle', 'wb') as handle:
    pickle.dump(half_cylinder_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("cylinder")

experiment_names = [
    "neighbor_init_sweep",
    "mindist_spread_sweep",
    "fitsne_perplexity_sweep",
    "bhtsne_perplexity_sweep",
    "hyp1"
]

ds_name = "cylinder"

cylinder_metrics = dict()
cylinder_params = None
for en in experiment_names:
    print(f"Metrics for experiment {en}")
    cylinder_metrics[en], cylinder_params = evaluate_experiments(experiment_name=en,
                     ds_name=ds_name,
                     has_labels=False, verbose=2)


with open(f'synth_data_gen/experiments/{ds_name}/metrics.pickle', 'wb') as handle:
    pickle.dump(cylinder_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)
