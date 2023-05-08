OU: "**O**riginal" **U**MAP paper 
TLU: True Loss UMAP paper

To consider:
 - Theoretical comparison with t-SNE? How extensive? technique before using some other ML technique, and compare against PCA, t-SNE, etc?
 - Consider using UMAP not only for visualisation, but as a DR
 - 

# 1. Introduction
- Dimension reduction: linear and non-linear
	- Timeline and mention of other non-linear DR techniques (e.g. t-SNE)
- How UMAP appears to be different
	- UMAP's main perceived advantages (speed, good results in challenging case studies)
- Mention to how theory and implementation do not match ("Una piedra en el camino")
- Motivation
- Structure of the dissertation

# 2. UMAP

## Basic structure
- Main assumptions / axioms
- Steps to UMAP:
	- Graph construction
	- Graph Layout
	Could be taken from Section 3 of OU: quick pointers / takeaways from the theoretical background
- Mention hyper-parameters

## Theoretical background
- _Could be an appendix, going into the mathematical details_?
- (Not so) deep explanation on the theory behind UMAP (manifolds, Riemannian metric, category theory, fuzzy sets, ...)
- 
- Explanation for the motivation behind the **theoretical loss function** (fuzzy set cross-entropy, membership function)

## Algorithm
- More detailed step-by-step workings of UMAP, less theoretical, a description of the original implementation
- Explanation of the sampling used, in particular **negative sampling**, which is the reason of the divide between theory and implementation of UMAP
- Explain hyper-parameter tuning

## True Loss Function
- Full derivation of the effective loss function as shown in TLU.
	- Analysis of each component of the loss function
	- How the loss function is updated in each epoch
	- How the effective loss function is derived as the expectation of each update under the probability of the sampling used
- Justification for the changes that TLU introduces to the implementation
	- changes in the gradient so as to find a real loss function
	- updating embeddings of both vertices inciding on an edge instead of only one...)

## UMAP's derivatives
 
### Other metric spaces

### densUMAP

### Parametric UMAP

### Supervised Learning Application?


# 3. Experiments

Hyper-parameter tuning / selection. How they affect the 

## UMAP's successes
- Showcase how UMAP has found success and is now one of the main non-linear DR techniques employed in fields like biology
	- Some toy examples
	- Examples in other spaces
	- Biology examples
	- Find new field in which it also succeeds?
- Comparison between UMAP and densMAP


## UMAP's failure
- Consider toy examples showing qualitatively how UMAP doesn't actually minimise its purported loss function
	- Numerical examples computing the full loss function might be too time-demanding, computation-wise
- Datasets in which it doesn't work that well


## Comparison with t-SNE

# 4. Conclusions
- Motivation, original examples, etc


Experimentos numéricos dentro de cada sección, para el fin que busquemos en cada sección (comparación con t-SNE, funcionamiento propio, fallos, etc)