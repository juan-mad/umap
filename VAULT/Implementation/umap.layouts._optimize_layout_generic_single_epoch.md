```python
_optimize_layout_generic_single_epoch(  
	epochs_per_sample=epochs_per_sample,  
	epoch_of_next_sample=epoch_of_next_sample,  
	head=head,  
	tail=head,  
	head_embedding=head_embedding,  
	tail_embedding=tail_embedding,  
	output_metric=output_metric,  
	output_metric_kwds=output_metric_kwds,  
	dim=dim,  
	alpha=alpha,  
	move_other=move_other,  
	n=n,  
	epoch_of_next_negative_sample=epoch_of_next_negative_sample,  
	epochs_per_negative_sample=epochs_per_negative_sample,  
	rng_state=rng_state,  
	n_vertices=n_vertices,  
	a=a,  
	b=b,  
	gamma=gamma
)
```

In principle, `head_embedding` is the embedding we want to optimise and `tail_embedding` is a reference one. The latter will stay fixed if `move_other` is False and will be updated if it is True. In our case, both embeddings are the same and we update them both.

----


For $i$ in `range(epochs_per_sample.shape[0])`(i.e. every edge):
	If `epoch_of_next_sample[i] <= n` (whether to sample this edge):
		`j = head[i]` (*origin* vertex index)
		`k = tail[i]`(*destiny* vertex index)
		We get the embedding coordinates of both vertices
		`current = head_embedding[j]`
		`other = tail_embedding[k]`
		We then cmpute the distance and gradient of the distance between these two points, stored in `dist_output` and `grad_dist_output`. We also compute the "reverse gradient" by changing their places in the distance function, and store it in `rev_grad_dist_output`.
		When the distance is positive, `dist_output > 0.0`:
			$$w_l = \frac{1}{1+a\cdot\text{dist\_output}^{2b}}$$
		`else`:
			$$w_l = 1$$
		The gradient is then computed as ($\varepsilon=10^{-6})$ $$\text{grad\_coeff} = \frac{2\cdot b\cdot (w_l-1)}{\text{dist\_output}+ \varepsilon}$$
		Then, `for`  each of the coordinates (`d in range(dim)`):
			We "clip" 