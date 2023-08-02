import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import pickle
import os

from sklearn.manifold import trustworthiness
from sklearn.metrics.pairwise import euclidean_distances

from scipy.stats import spearmanr


class Metrics():
    def __init__(self, highdata, lowdata=None, K=-1, metric="euclidean"):
        self.N = highdata.shape[0]
        self.high = highdata
        self.metric = metric
        self.low = None

        if K < 0:
            self.K = int(self.N / 10)
        else:
            self.K = K
        
        self.high_distance_matrix = None
        self.low_distance_matrix = None
        
        self.high_rank_matrix = None
        self.low_rank_matrix = None
        
        self.trustworthiness = None
        self.continuity = None
        self.normalised_stress = None
        self.neighbourhood_hit = None
        self.shepard_goodness = None
        self.average_local_errors = None

        self.compute_high_distance_matrix()
        self.compute_high_rank_matrix()
        
        if lowdata is not None:
            self.compute_low_distance_matrix()
            self.compute_low_rank_matrix()

    def compute_high_distance_matrix(self):
        """ Computes distance matrix of high dimensional data """
        assert self.metric == "euclidean"
        self.high_distance_matrix = euclidean_distances(self.high)
        
    def compute_low_distance_matrix(self):
        """ Computes distance matrix of low dimensional data """
        assert self.metric == "euclidean"
        assert self.low is not None
        self.low_distance_matrix = euclidean_distances(self.low)
        
    def compute_high_rank_matrix(self):
        self.high_rank_matrix = self._compute_rank_matrix(self.high_distance_matrix)
    
    def compute_low_rank_matrix(self):
        self.low_rank_matrix = self._compute_rank_matrix(self.low_distance_matrix)
    
    def _compute_rank_matrix(self, matrix):
        return np.array([np.argsort(np.argsort(r)) for r in matrix])
        
    def set_low_data(self, low):
        self.low = low
        self.compute_low_distance_matrix()
        self.compute_low_rank_matrix()
        
    def set_labels(self, labels):
        self.labels = labels
        
    def get_trustworthiness(self):
        assert self.high_distance_matrix is not None
        assert self.low_distance_matrix is not None
        
        self.trustworthiness = trustworthiness(X=self.high_distance_matrix, X_embedded=self.low_distance_matrix)
        return self.trustworthiness
    
    def get_continuity(self):
        assert self.low_rank_matrix is not None
        assert self.high_rank_matrix is not None
        
        self.continuity = 0
        
        high_neighbours = np.concatenate([
            np.where(self.high_rank_matrix == k)[1].reshape(-1, 1) for k in range(1, self.K+1)
        ], axis=1)
        
        for j in range(self.K):
            self.continuity += np.sum(
                    np.max(
                        [np.zeros(self.N, dtype=int),
                         (self.low_rank_matrix[range(self.N), high_neighbours[:,j]]) - self.K],
                        axis=0
                    )
            )
        self.continuity = 1 - (2 / (self.N*self.K*(2*self.N-3*self.K-1)))*self.continuity
            
        return self.continuity
    
    def get_normalised_stress(self):
        assert self.high_distance_matrix is not None
        assert self.low_distance_matrix is not None
        
        indices = np.triu_indices(self.N, k=1)
        
        
        
        self.normalised_stress = 1 - (
            np.sum((self.high_distance_matrix[indices] - self.low_distance_matrix[indices]) ** 2)
            /
            np.sum((self.high_distance_matrix[indices])**2)
        )
        return self.normalised_stress
        
    
    def get_neighbourhood_hit(self):
        assert self.low_rank_matrix is not None
        assert self.high_rank_matrix is not None
        
        self.neighbourhood_hit = 0
        
        low_neighbours = np.concatenate([
            np.where(self.low_rank_matrix == k)[1].reshape(-1, 1) for k in range(1, self.K+1)
        ], axis=1)
        
        for i in range(self.N):
            self.neighbourhood_hit += (
                np.sum(self.labels[low_neighbours[i,:]] == self.labels[i])
            )
        self.neighbourhood_hit = self.neighbourhood_hit / (self.N * self.K)
        return self.neighbourhood_hit
        
    def get_shepard_goodnees(self):
        assert self.high_distance_matrix is not None
        assert self.low_distance_matrix is not None
        
        indices = np.triu_indices(self.N, k=1)
        
        self.shepard_goodness = spearmanr(
            self.high_distance_matrix[indices],
            self.low_distance_matrix[indices]
        )
        
        return self.shepard_goodness.statistic
    
    def get_average_local_error(self):
        assert self.high_distance_matrix is not None
        assert self.low_distance_matrix is not None
        
        indices = np.triu_indices(self.N, k=1)
        max_high = np.max(self.high_distance_matrix[indices])
        max_low = np.max(self.low_distance_matrix[indices])
        
        self.average_local_errors = (1/(self.N-1)) * np.sum(np.abs(
            (1/max_high) * self.high_distance_matrix[indices]
            -
            (1/max_low) * self.low_distance_matrix[indices]
        ))
        
        return self.average_local_errors
    
    def get_metrics(self, lowdata=None, labels=None, mean_score=False):
        if lowdata is not None:
            self.set_low_data(lowdata)
            
        if labels is not None:
            self.set_labels(labels)
            
        tw = self.get_trustworthiness()
        ct = self.get_continuity()
        ale = self.get_average_local_error()
        
        if labels is not None:
            nh = self.get_neighbourhood_hit()
        else:
            nh = float('nan')
        
        sg = self.get_shepard_goodnees()
        
        if mean_score:
            if labels is not None:
                score = 0.2 * (tw + ct + ale + nh + sg)
            else:
                score = 0.25 * (tw + ct + ale + sg)
            return score
        else:
            if labels is not None:
                return [tw, ct, ale, nh, sg]
            else:
                return [tw, ct, ale, sg]
            
            
def evaluate_experiments(ds_name, experiment_name, extra_name="", has_labels=False, verbose=0, max_dataset_size=10000):
    HP = pd.read_csv(f"synth_data_gen/experiments/{experiment_name}.csv")
    params = pd.read_csv(f"synth_data_gen/synth_datasets/{ds_name}/params.csv")
    
    low_data = dict()
    high_data = dict()
    
    metrics = dict()
    
    for idx in params.index:
        try:
            with open(f'synth_data_gen/synth_datasets/{ds_name}/set_{idx}.pickle', 'rb') as handle:
                high_data[idx] = pickle.load(handle)
        except Exception as e:
            print(f"Failed to load high dimensional data, idx={idx}")
            print(e)
            continue

        if high_data[idx]["data"].shape[0] > max_dataset_size:
            print(f"Dataset has size {high_data[idx]['data'].shape[0]}, greater than {max_dataset_size}")
            continue

        for i in HP.index:
            try:
                with open(f'synth_data_gen/experiments/{ds_name}/{experiment_name}{extra_name}/set_{idx}_HP_{i}', 'rb') as handle:
                    low_data[(idx, i)] = pickle.load(handle)
            except Exception as e:
                print(f"Failed to load low dimensional data, idx={idx}, i={i}")
                print(e)
                continue

        shape_metric = Metrics(high_data[idx]["data"])
        if has_labels:
            if "labels" in high_data[idx].keys():
                labels = high_data[idx]["labels"]
                if type(labels) == list:
                    labels = np.array(labels, dtype=int)
            elif "classes" in high_data[idx].keys():
                labels = high_data[idx]["classes"]
                if type(labels) == list:
                    labels = np.array(labels, dtype=int)
            else:
                raise Exception("Found no label data named 'labels' or 'classes'")

        else:
            labels = None
        if verbose > 0:
            print(f"Computing metrics for set {idx}")
        m = []
        for i in HP.index:
            if verbose > 1:
                print(f"\t HPs number {i}")
            try:
                m.append(
                    shape_metric.get_metrics(lowdata = low_data[(idx,i)]["data"], labels=labels)
                )
            except Exception as e:
                if verbose > 1:
                    print(f"\t\t Exception {e}; passing these HPs")
        metrics_df = pd.concat([HP, pd.DataFrame(m)], axis=1)

        if has_labels:
            metric_names = ["trustworthiness", "continuity", "normalised_stress", "neighbourhood_hit", "shepard_goodness"]
        else:
            metric_names = ["trustworthiness", "continuity", "normalised_stress", "shepard_goodness"]

        metrics[idx] = metrics_df.rename(columns={i: metric_names[i] for i in range(len(metric_names))})

    return metrics, params
