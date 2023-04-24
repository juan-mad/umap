---
aliases: ["changes"]
---

## Normalisation of $\sigma_i$
[[umap.umap_.smooth_knn_dist]]. Page 15.

In the original paper, for each point $i$ the value of $\sigma_i$ is computed so that $$\sum_{j=1}^k\exp\left( \frac{-\max(0, d(x_i,x_j)-\rho_i)}{\sigma_i} \right) = \log_2(k)$$
The implementation introduces as a parameter for `smooth_knn_dist` the **bandwidth**, such that the right side of the equation becomes $$\log_2(k)\cdot \text{bandwidth}$$
This is not mentioned in the paper nor is it's effect studied.



## Local connectivity
[[umap.umap_.smooth_knn_dist]]. Page 15.

In order to ensure that the manifold in which the data lies is locally connected, the values of $\rho_i$ are selected so as to ensure that at least one neighbour has similarity equal to 1 with the point in question. According to the paper, this saves us problems in high-dimension **where other algorithms such as t-SNE start to suffer from the curse of dimensionality**. 
	Is this because distances "grow" fast with the number of dimensions and these algorithms do not account for it?




## Removal of "weak" edges
[[umap.umap_.simplicial_set_embedding]]
What this appears to try and do is remove edges when we specify very few epochs for training. If the total number of epochs is fewer than 10, then all edges with weights smaller than $$\frac{\max e_{st}}{200}$$ is deleted (500 in the denominator when the sample is small). If the total number of epochs is larger than 10 (for example, 20 or 2000) then the edges with weight smaller than $$\frac{\max e_{st}}{\texttt{n\_epochs\_max}}$$ are deleted. Notice that when the epochs are 10, fewer edges are deleted compared to when we use 11 epochs!


## Linear scaling of initial embedding
[[umap.umap_.simplicial_set_embedding]]

All coordinates of the initial embedding are scaled, so that they all fall inside the range $[0,10]$.

#Important UMAP's true loss paper claims that "the rings appear more compact than the original, even when the initialisation are the original points themselves". This could be caused by 1) this normalisation to the interval $[0,10]$ of each coordinate, and 2) the minimal distance in the embedding space or `min-dist` in the original paper, which regulates how near can neighbors be in the embedding space. This is a hyper-parameter, and the mean distance between two points inside the ring could (maybe?) be computed and entered to produce a similar width?

## Clipping
[[umap.layouts._optimize_layout_generic_single_epoch]]
[[umap.layouts._optimize_layout_euclidean_single_epoch]]
