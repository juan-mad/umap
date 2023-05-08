# Hyper parameters
- Study effect of main hyperparameters such as `n_neighbor`, `min_dist`, `spread` and `n_epochs` among others.
- `n_epochs` and `learning_rate` are related to each other: `learning_rate` decreases from its inital value towards 0 in steps of size `learning_rate / n_epochs`.
- Consider size of the dataset, that is, number of samples, as a factor as well.
- Consider the scale of the dataset. Distance is affected by scale, and by default UMAP initialises every embedding inside $[0,10]^d$ where $d$ is the embedding dimension.
	- This could mean the initial "diameter" of the dataset as well as width of, say, an annulus.
	- This last point should be checked against On UMAP's True Loss Function.



# Artificial data
## 1-manifolds (curves)
- Create curves in 2D, colored in a cyclic way, add noise and embed them into 2D again
- Create curves in $n$ dimensions, add noise and embed them into 2D.
- Create curves in 2D, add noise in other dimensions and reduce back into 2D.
	- Maybe curves in 3D plus other noise as well


## 2-manifolds (surfaces)

- Surfaces in 3D, either colored in an appropriate way, or associate it with an image
- Surfaces in 3D plus noise (in 3D)
- Surfaces in 2D/3D plus noise in additional dimensions, and reduce back.

## 3-manifolds
- Volume could be represented by a GIF / animated image. How could we visualise the embedding back in 3D?



