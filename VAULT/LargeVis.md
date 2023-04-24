Given a large scale and high dimensional data set $X=\{x_i\in \mathbb R^d\}_{i=1}^N$ , we want to represent each data point $x_i$ with a low dimensional vector $y_i\in\mathbb R^s$ where $s$ is usually 2 or 3.

## Efficient KNN Graph Construction
We need a measure of distance: we use the euclidean distance in high-dim space just as t-SNE does. Constructing the exact graph takes $O(N^2d)$.

Instead of random projection trees for the approximation of the graph, LargeVis uses *neighbor exploring techniques*, where the idea is that "a neighbor of my neighbor is likely to be my neighbor".  (for details, see Page 4 of the paper). The weights for each edge in the KNN graph are the same as for t-SNE.

## Probabilistic Model for Graph Visualisation
To visualise one just need to project into 2D/3D. With a probabilistic model we try to keep similar vertices close to each other and dissimilar vertices far apart in low-dim. Given a pair of vertices $v_i, v_j$ , the probability of observing the binary edge $e_{ij}=1$ is defined as $$P(e_{ij}=1) = f(||y_i-y_j||),$$ where $f$ is a probabilistic function with respect to the distance in low-dim space. Many probabilistic functions can be used, e.g. $f(x) = (1+ax^2)^{-1}$ or $f(x)=(1+\exp(x^2))^{-1}$ . To define the probability with respect to the weight of an edge, we define $$P(e_{ij}=w_{ij}) = P(e_{ij}=1)^{w_{ij}}$$ Then, the likelihood of the graph can be calculated as $$O = \prod_{(i,j)\in E}P(e_{ij}=1)^{w_{ij}}\prod_{(i,j)\notin E}(1-P(e_{ij}=1))^\gamma  $$
$$ \sum_{(i,j)\in E}w_{ij}\log P(e_{ij}=1) + \sum_{(i,j)\notin E}\gamma \log (1-P(e_{ij}=1)$$ where $E$ is the set of the edges and $\gamma$ is **an unified weight assigned  to the negative edges**. 

Maximising first part implies closeness of similar vertices, while maximising the second part imples that dissimilar vertices will be far apart.

#### Optimisation
Direct maximisation is inefficient since the number of negative edges is quadratic to the number of nodes. Inspired by negative sampling, we randomly sample some negative edges:
 - For each node $i$ we randomly sample some vertices $j$ according to a noisy distribution $P_n(j)$ and treat $(i,j)$ as negative edges. LargeVis uses $P_n(j)\propto d_j^{0.75}$ . Let $M$ be the number of negative samples for each positive edge. The objective function is then redefined as

$$ O = \sum_{(i,j)\in E}w_{ij}\left( \log P(e_{ij}=1) + \sum_{k=1}^M\mathbb E_{j_k\sim P_n(j)}\gamma\log(1-P(e_{ij}=1)) \right) $$ Stochastic gradient descent is problematic for optimisation: when sampling the edge $(i,j)$ the weight $w_{ij}$ will be multiplied into the gradient, and when the values of the weights diverge so do the norms of the gradients, so choosing a good learning rate is difficult. Instead, **we randomly sample edges with the probability proportional to their weights and treat them as binary edges**.


