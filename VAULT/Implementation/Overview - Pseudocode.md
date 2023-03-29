Pseudo code from OU.

### Algorithm 1: UMAP
#Alg1
```python
def UMAP(X, n, d, min_dist, n_epochs):
	# Construct the relevant weighted graphs
	for x in X:
		fs_set[x] = LocalFuzzySimplicialSet(X, x, n)
	top_rep = {fs_set[x] for x in X}
	
	# Perform optimization of the layout
	Y = SpectralEmbedding(top_rep, d)
	Y = OptimizeEmbedding(top_rep, Y, min_dist, n_epochs)
	return Y
```


### Algorithm 2: Constructing a local fuzzy simplicial set
#Alg2
```python
def LocalFuzzySimplicialSet(X, x, n):
	knn, knn_dist = ApproxNearestNeighbors(X, x, n)
	rho = knn_dist[1] # Distance to nearest neighbor; node is its own neighbour
	sigma = SmoothKNNDist(knn_dist, n, rho)
	fs_set[0] = X
	fs_set[1] = {([x,y],0) for y in X}
	for y in knn:
		dxy = max(0, dist(x,y) - rho)/sigma
		fs_set[1] = fs_set[1].add(([x,y], exp(-dxy)))
	return fs_set
```

### Algorithm 3: Compute the normalising factor for distances $\sigma$
#Alg3
```python
def SmoothKNNDist(knn_dist, n, rho):
	# Binary search for sigma such that the sum of the exp(-dxy) is equal to
	# log_2(n)
	return sigma 
```

### Algorithm 4: Spectral embedding for initialisation
#Alg4
```python
def SpectralEmbeddding(top_rep, d):
	A <- # 1-skeleton of top_rep expressed as a weighted adjacency matrix
	D <- # degree matrix for the graph A
	L <- D**(1/2) * (D-A) * D**(1/2)
	evec <- # Eigenvectors of L, sorted
	Y <- evec[1:(d+1)] # 0-base indexing assumed; recall first neighbour is self
	return Y
```

### Algorithm 5: Optimising the embedding
#Alg5
```python
def OptimizeEmbedding(top_rep, Y, min_dist, n_epochs):
	alpha = 1.0
	# Fit Phi from Psi defined by min_dist
	for e in range(1,n_epochs+1):
		for ([a,b], p) in top_rep:
			if Random() <= p: # Sample with probability p
				Y[a] = Y[a] + alpha * grad(log Phi)(Y[a], Y[b])
				for i in range(1,n_neg_samples+1):
					c = # Random sample from Y
					Y[a] = Y[a] + alpha * grad(log (1-Phi))(Y[a], Y[c])
		alpha = 1.0 - e / n_epochs
	return Y
```

