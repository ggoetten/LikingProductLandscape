# Liking Product Landscape #

Liking Product Landscape (LPL) can be easily used for understanding the comparison of products based on consumers' perceptions 
evaluations.

We are preparing a journal paper where LPL methodology can be described. 

## Quick Start ##

### Preparation for using LPL methodology 

- Download and install anaconda distribution

```python
# Importing EvoDAG ensemble
from EvoDAG.model import EvoDAGE
# Importing iris dataset from sklearn
from sklearn.datasets import load_iris

# Reading data
data = load_iris()
X = data.data
y = data.target

#train the model
m = EvoDAGE(n_estimators=30, n_jobs=4).fit(X, y)

#predict X using the model
hy = m.predict(X)
```
