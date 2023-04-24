Construct the membership strength data for the 1-skeleton of each local fuzzy simplicial set -- this is formed as a sparse matrix where each row is a local fuzzy simplicial set, with a membership strength for the 1-simplex to each other data point.

Arguments:
- `knn_distances`: indices of the neighbours closest to each sample
- `knn_dists`: distances to those neighbours
- `sigmas`: normalization factors derived from the metric tensor approximation
- `rhos`: local connectivity adjustment
- (opt, False) `return_dists`: whether to return pairwise distances associated with each edge
- (opt, False) `bipartite`: Does the nearest neighbour set represent a bipartite graph? That is, are the nearest neighbor indices from the same point as the row indices?

Returns:
- `rows`: row data for the resulting sparse matrix (coo format)
- `cols`: column data for the resulting sparse matrix (coo format)
- `vals`: entries for the resulting sparse matrix (coo format)
- `dists`: distances associated with each entry in the resulting sparse matrix

Function:
- `rows, cols, vals` <- arrays of 0 with same shape as `knn_indices`
- If `return_dists:
	- `dists` <- 0's as well
- else:
	- `dists = None`
- For each sample $i$:
	- For each neighbor $j$:
		- If `knn_indices[i,j] == -1`:
			- `continue` (# we didn't get the full KNN of $i$) (????)
		- If applied to an adjacency matrix, points shouldn't be similar to themselves.
		- If applied to an incidence matrix (or bipartite), then the row and column indices are different
		- If `bipartite == False and knn_indices[i,j] == i` (meaning, we are not dealing with a bipartite graph and *$i$ is registered as a neighbour to itself*)
			- `val = 0`
		- else if `knn_dists[i, j] - rhos[i] <= 0.0 or sigmas[i] == 0.0`:
			- `val = 1`
		- else:
			- `val = np.exp(-((knn_dists[i, j] - rhos[i]) / (sigmas[i])))`
		- Set values appropriately for coo sparse matrix format

`return rows, cols, vals, dists`

#UMAP_algorithm 