Improve an embedding using SFD to minimise the fuzzy set cross entropy between the 1-skeletos of the high-dim and low-dim fuzzy simplicial sets. In practice this is done by sampling edges based on their membership strength, with the repulsive terms sampled using **negative sampling**.

Parameters:
 - `head_embedding`: initial embedding to be improved
 - `tail_embedding`: reference embedding of embedded points. If we are not embedding new unseen points with respect to an already existing embedding, this is just `head_embedding` again; otherwise it provides an existing embedding to embed with respect to.
 - `head`: indices of the heads of the 1-simplices with non-zero membership.
 - `tail`: indices of the tails of the 1-simplices with non-zero membership.
 - `n_epochs`: number of training epochs
 - `n_vertices`: number of vertices (0-simplices) in the dataset.
 - `epochs_per_sample`: float value with the number of epochs per 1-simplex: weaker membership means that more epochs will pass between being sampled.
 - `a`, `b`: parameters for differentiable approximation of right adjoint functor
 - `rng_state`
 - (opt, 1) `gamma`: weight to apply to negative samples
 - (opt, 1) `initial_alpha`: initial learning rate for SGD
 - (opt, 5) `negative_sample_rate`: number of negative samples to user per positive sample.
 - (opt, dist.euclidean) `output_metric`
 - (opt, ()) `output_metric_kwds`
 - (opt, False) `verbose`
 - (opt, None) `tqdm_kwds`
 - (opt, False) `move_other`: weather to adjust `tail_embedding` alongside `head_embedding`

Returns
- `embedding`: the optimised embedding

Function:
- `dim` <- embeddiing space dimension
- `alpha` <- `initial_alpha`
- `epochs_per_negative_sample` is just `epochs_per_sample` divided by `negative_sample_rate`
- `epoch_of_next_negative_sample` is a copy of `epochs_per_negative_sample`
- `epoch_of_next_sample` is a copy of `epochs_per_sample`

------
*Called by `simplicial_set_embedding`*

```python
optimize_layout_generic(  
    head_embedding,=embedding  
    tail_embedding=embedding,  
    head=head,  
    tail=tail,  
    n_epochs=n_epochs,  
    n_vertices=n_vertices,  
    epochs_per_sample=epochs_per_sample,  
    a=a,  
    b=b,  
    rng_state=rng_state,  
    gamma=gamma, #1.0,  
    initial_alpha=initial_alpha, #1.0,  
    negative_sample_rate=negative_sample_rate, #5.0,  
    output_metric=output_metricc, #dist.euclidean,  
    output_metric_kwds=tuple(output_metric_kwds.values(), #(),  
    verbose=verbose, #False,  
    tqdm_kwds=tqdm_kwds, #None,  
    move_other=True #False,  
):
```

We begin by defining `dim` as the dimension of the embedding space, and `alpha` as the value of `initial_alpha`.

Now, define:
 - `epochs_per_negative_sample` as the value of `epochs_per_sample` divided by `negative_sample_rate`. 
 - `epoch_of_next_negative_sample` as a copy of `epochs_per_negative_sample`.
 - `epoch_of_next_sample` as a copy of `epochs_per_sample`.

For each epoch $n$ from 0 to `n_epochs`:
 - Call `_optimize_layout_generic_single_epoch` [[umap.layouts._optimize_layout_generic_single_epoch]]
 - Set the new learning rate to `alpha = initial_alpha * (1 - n / n_epochs)`

Then, return `head_embedding`.

#UMAP_algorithm 