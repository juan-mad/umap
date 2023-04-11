Perform a fuzzy simplicial set embedding, using a specified initialisation method and then minimizing the fuzzy set cross entropy between the 1-skeletons of the high and low dimensional fuzzy simplicial sets.

Parameters:
- `data`: source data to be embedded by UMAP, 
- `graph`: sparse matrix with the 1-skeleton of the high-dim fuzzy simplicial set, representing the weighted adjacency matrix
- `n_components`: dimension of the euclidean space into which to embed the data
- `initial_alpha`: initial learning rate for SGD
- `a`, `b` : parameters for differentiable approximation of right adjoint functor
- `gamma`: weight to apply to negative samples
- (opt, 5) `negative_sample_rate`: number of negative samples to select per positive sample during optimization. "Increasing this value will result in greater repulsive forces, greater optimization cost but slightly better accuracy"
- (opt, 0) `n_epochs`: number of training epochs. If 0 then 200 or 500 is selected for large and small datasets resp. If a list of integers, intermediate embeddings are returned in `aux_data["embedding_list"]`.
- `init`: How to initialize the low dim embedding. Options are:
	- spectral: spectral embedding of the fuzzy 1-skeleton
	- random
	- pca: use the first `n_components` from PCA applied to the input data.
	- A numpy array of initial embedding positions.
- `random_state`: state capable of being used as a numpy random state
- `metric`: the metric used to measure distance in high-dim space. (Used if multiple connected components need to be laid out (?))
- `metric_kwds`: keyword arguments to be passed to the metric function
- `densmap`: whether to use densMAP or not.
- `densmap_kwds`
- `output_dens`: whether to output local radii in the original data and the embedding
- `output_metric`: function returning the distance between two points in embedding space and the gradient of the distance wrt the first argument.
- `output_metric_kwds`
- `euclidean_output`: whether to use the faster code specialised for euclidean output metrics
- (opt, False) `parallel`: whether to run the computation using numba parallel. If used, there is no reproducibility (non-deterministic)
- (opt, False) `verbose`
- `tqdm_kwds`

Returns:
- `embedding`: the optimised of `graph` into an `n_components`-dim euclidean space.
- `aux_data`: auxiliary output returned with the embedding

Function:
- Makes sure that the original data graph is stored as a sparse matrix
- Set number of maximum epochs
- Some "normalisation" is done: according to the number of maximum epochs to consider, those *edges with a weight lower than the maximum weight divided by the max number of epochs* are **set to zero**.
- Generate initial embedding coordinates according to the given initialisation method given as an argument.
- `epochs_per_sample = ` [[umap.umap_.UMAP().make_epochs_per_sample]] #question
- There is linear scaling done on the embedding coordinates (which, at this point, is just the initialization)
- Optimise the embedding layout, with an improved / faster version for the euclidean case
- [[umap.layouts.optimize_layout_generic]]
- [[umap.umap_.UMAP().optimize_layout_euclidean]]
- `aux_data` is stored
- Return `embedding` and `aux_data`

------------

```python
simplicial_set_embedding(  
    data=X,  
    graph=self.graph_,  
    n_components=self.n_components,  
    initial_alpha=self._initial_alpha,  
    a=self._a,  
    b=self._b,  
    gamma=self.repulsion_strength,  
    negative_sample_rate=self.negative_sample_rate,  
    n_epochs=n_epochs,  
    init=init,  
    ramdon_state=random_state,  
    metric=self._input_distance_func,  
    metric_kwds=self._metric_kwds,  
    densmap=self.densmap,  
    densmap_kwds=self._densmap_kwds,  
    output_dens=self.output_dens,  
    output_metric=self._output_distance_func,  #dist.named_distances_with_gradients["euclidean"]
    output_metric_kwds=self._output_metric_kwds,  # {}
    euclidean_output=self.output_metric in ("euclidean", "l2"), #True   
    parallel=self.random_state is None,  #False
    verbose=self.verbose,  #False 
    tqdm_kwds=self.tqdm_kwds,  #None
)
```

First, makes sure that `graph` is in `coo` format (sparse matrix), and gets rid  of duplicate entries by summing the values of that repeated edge.

We define `n_vertices` has the number of columns of `graph`.

In the case that our dataset has few samples (lower tham 10,000), we set `default_epochs` to 500 (since it will not be so costly to compute). If not, we set it to 200.

~~If densMAP, add another 200 epochs.~~

If we were given a list of epochs in `n_epochs`, we define `n_epochs_max` as the maximum value of the list; if it is just an int, then it is set to `n_epochs`.

**If `n_epochs_max > 10`, we do the following:** 

	graph.data[graph.data < (graph.data.max() / float(n_epochs_max))] = 0.0

**and if not**, 

	graph.data[graph.data < (graph.data.max() / float(default_epochs))] = 0.0

What these lines do is: let $e_{ij}$ be an edge connecting $i$ to $j$ . If $$e_{ij} < \frac{\max\limits_{s,t}{e_{st}}}{\max(10, \texttt{n\_epochs\_max})}$$ then, $e_{ij}$ is set to 0. Afterwards, remove all entries with a value of 0 (we are using a sparse matrix type so this saves memory).

~
*Initialisation* -> stored in `embedding`, which is a `(graph.shape[0], n_components)` numpy array (that is, with as many rows as vertices and with as many columns as the dimension of the embedding space)
~

Now, we call to `make_epochs_per_sample(weights=graph.data, n_epochs=n_epochs_max)` [[umap.umap_.UMAP().make_epochs_per_sample]]. What this code returns and is stored in `epochs_per_sample` is, basically, a vector with an entry for each edge, where each entry, say assigned to $e_{ij}$, is $$\begin{cases}\frac{\max\limits_{s,t}e_{st}}{e_{ij}} & \text{ if } e_{ij}>0 \\ -1 & \text{ if } e_{ij} \leq 0\end{cases}$$
*In theory there are no $e_{ij}=0$ left, and I don't know if it is possible for any edge to be negative.*
What this represents is the frequency with which an edge will be sampled. The stronger edges will have lower values in the corresponding entry, which signifies *the number of epochs that have to pass until we sample that edge*.

Define `head = graph.row`, `tail = graph.col` and `weight=graph.data`.

There is some linear scaling performed on the initialised embedding:
	Let $\mathbf y_j = (y_{1j},\dots,y_{mj})$ be the coordinates of vertex $j$ in the embedding space. Then, the linear transformation is such that $$\mathbf y_j = 10 \cdot \frac{\mathbf y_j - \left(\min\limits_j y_{1j}, \dots, \min_j y_{mj}\right)}{\left(\max\limits_j y_{1j}, \dots, \max_j y_{mj}\right)-\left(\min\limits_j y_{1j}, \dots, \min_j y_{mj}\right)}$$     which is to say, each coordinate is normalised according to the minimum and maximum value reached by all points, and then multiplied by 10.

*Now the algorithm distinguishes between an optimisation when the embedding space has an euclidean distance and when it does not. The following is according to the general and not the optimised version.*

Finally, we optimise the embedding according to the loss function by calling the following function [[umap.layouts.optimize_layout_generic]]:

```python
embedding = optimize_layout_generic(  
    embedding,  
    embedding,  
    head,  
    tail,  
    n_epochs,  
    n_vertices,  
    epochs_per_sample,  
    a,  
    b,  
    rng_state,  
    gamma,  
    initial_alpha,  
    negative_sample_rate,  
    output_metric,  
    tuple(output_metric_kwds.values()),  
    verbose=verbose,  
    tqdm_kwds=tqdm_kwds,  
    move_other=True,  
)
```