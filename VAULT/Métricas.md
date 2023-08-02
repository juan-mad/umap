

- Trustworthiness (`sklearn.manifold.tristworthiness)
	- Proporción de puntos en $X$ que también están cerca en $Y$. Puntúan negativamente aquellos kNNs en $Y$ que no son kNN en $X$.
- Continuity
	- Puntúan negativamente aquellos kNNs en X que no son kNN en $Y$.
- Normalised stress <- creo que aquí no tiene mucho sentido
- Neighbourhood hit: en $Y$, proporción de kNNs de un punto que tienen la misma etiqueta.
- Shepard goodness: Spearman's rank correlation coefficient of distance scatterplot
- Average local errors
