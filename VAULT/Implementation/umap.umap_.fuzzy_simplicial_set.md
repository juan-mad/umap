Given a set of data X, a neighborhood size, and a measure of distance compute the fuzzy simplicial set (here represented as a fuzzy graph in the form of a sparse matrix) associated to the data. This is done by locally approximating geodesic distance at each point, creating a fuzzy simplicial set for each such point, and then combining all the local fuzzy simplicial sets into a global one via a fuzzy union.

- If no knn has been computed
	- `knn_indices, knn_dists` returned from `nearest_neighbors`
- $\sigma$'s' and $\rho$'s' are returned from `smooth_knn_dist` [[umap.umap_.smooth_knn_dist]]
- `rows, cols, vals, dist` returned from `compute_membership_strengths` [[umap.umap_.compute_membership_strengths]]
- `result` returned from `scipy.sparse.coo_matrix` (sparse matrix?)
- Eliminate zeroes from `result`.
- Return distances if asked to.

`return result, sigmas, rhos[, dists]`
