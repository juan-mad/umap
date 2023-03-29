
First we import UMAP and the sample dataset.
```python
import umap # import umap
from sklearn.datasets import load_digits # import MNIST digit dataset

digits = load_digits()
```

Now we create a model: [[umap.umap_.UMAP()]]
```python
reducer = umap.UMAP(random_state = 42)
type(reducer)
```

```
> umap.umap_.UMAP
```

Fit the model: [[umap.umap_.UMAP().fit]]
```python
reducer.fit(digits.data)
```



In order to access the embedding, we can access the `embedding_` attribute of `reducer` or call `.transform()`.
```python
import numpy as np

embedding1 = reducer.transform(digits.data)
embedding2 = reducer.embedding_

assert(np.all(embedding1 == embedding2))
```


