Compute a continuous version of the distance to the kth nearest neighbor. That is, this is similar to knn-distance but allows continuous k values rather than requiring an integral k. In essence we are simply computing the distance such that the cardinality of fuzzy set we generate is k.

Arguments:
- distances (for each sample to its neighbors)
- k: number of nearest neighbors to approximate for
- (opt, 64) n_iter (for binary search)
- (opt, 1) local_connectivity
- (opt, 1) bandwidth

Returns:
- knn_dist: distance to kth neighbor
- nn_dist: distance to 1st neighbor

Function:

- `target` $=\log_2(k)\times$ `bandwidth`
- Initialise matrix of zeroes for the $\rho$'s, called `rho`, and for the $\sigma$'s, called `result`
- Compute the mean of all the distances -> `mean_distances` -> scalar! (?)
- For each sample point

	- Take the row of distances to neighbors for this sample point -> `ith_distances`
	- Consider the subarray of non-zero distances -> `non_zero_dists`

	- In order to ensure local connectivity of the manifold, according to the parameter `local_connectivity`, $\rho_i$ is chosen so that $\lfloor$`local_connectivity`$\rfloor$ weights in the graph will be equal to 1. In the case that `local_connectivity` is not a whole number, its value is linearly interpolated between the distance to the  $\lfloor$ `local_connectivity` $\rfloor$-neighbor and the next.

	- Binary search for $\sigma_i$:
	-  Initialise values for binary search for $\sigma_i$ :`lo = 0.0, hi = NPY_INFINITY, mid = 1.0`
	- For as many as `n_iter` iterations:
		- `psum = 0`
		- For $j$ in $\{1,2,\dots,k-1\}$   (where $k$ is the number of neighbors)
			- `d` = $dist(i, j) - \rho_i$
			- if `d > 0`:
				- `psum += exp(- d / mid)`
			- else: (i.e. we want this to be "distance zero" due to local connectivity)
				- `psum += 1`
		- If $|$ `psum - target` $|$ < SMOOTH_K_TOLERANCE ($= 10^{-5}$)
			- `break` (finish binary search)
		- If `psum > target`:
			- `hi = mid`
			- `mid = (lo + hi) / 2`
		- else:
			- `lo = mid`
			- if `hi = INF`:
				- `mid *= 2`
			- else:
				- `mid = (lo + hi) / 2`
	- `result[i] = mid`
	- $\sigma_i$ is set to `mid` or to a low bound relative to the mean of distances

`return result, rho`
