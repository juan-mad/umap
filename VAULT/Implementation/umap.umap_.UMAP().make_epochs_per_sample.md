Given a set of weights and number of epochs generate the number of epochs per sample for each weight.
Arguments:
- `weights`: weights of how much we want to sample each edge
- `n_epochs`: total number of epochs we want to train for.

Returns:
 - An array of number of epochs per sample, one for each 1-simplex.

Function:
- Declare `result` as a 1-d array of $-1$ with as many entries as `weights.shape[0]` (that is, I think, the number of edges?)
- For those edges that have a positive weight, the value of the array is subtituted by the *maximum weight divided by that weight*.

How is, in this sense, the array `result` an *array of number of epochs per sample, one for each 1-simplex i.e. one for each edge*? -> It appears that, in a sense, is the number of epochs that must pass between samplings.

