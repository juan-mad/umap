- Check correct format of input data X
- Handle optional arguments a, b
- Check if initial embedding points are provided.
	- If not, compute them as indicated
- Set initial learning reate
- Precomputed knn (if they have been done beforehand)
- Validate parameters
- Parallelisation handling of threads?
- Check if data should be unique'd
- Data handling in case data isn't unique
- Check n_neighbors based on data size

## Construct fuzzy simplicial set

- If metric is precomputed and sparse data:
	- Rows are sorted to check for neighbors
	- Call `fuzzy_simplicial_set` [[umap.umap_.fuzzy_simplicial_set]]
	- Ensure vertides with degree 0 were properly disconnected
- Elif small number of data:
	- Handle possility of sklearn working with sparse data and/or custom metric
	- Call `fuzzy_simplicial_set` [[umap.umap_.fuzzy_simplicial_set]]
- Else (standard case):
	- If `self.knn_dists is None`:
		- Call `nearest_neighbors` 
		- Returns to `self._knn_indices, self._kmm_dists, self._knn_search_index`
	- Call `fuzzy_simplicial_set` [[umap.umap_.fuzzy_simplicial_set]]
- Some checks on labels in case we are using supervised mode

## Construct embedding
- If `self.transform_mode == "embedding"`
	- Call `self._fit_embed_data` [[umap.umap_.UMAP()._fit_embed_data]]
		- returns to `self.embedding_, aux_data`


`return self` 



#UMAP_algorithm
