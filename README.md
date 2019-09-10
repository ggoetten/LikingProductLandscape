# Liking Product Landscape #

Liking Product Landscape (LPL) can be easily used for understanding the comparison of products based on consumers' perceptions 
evaluations.

We are preparing a journal paper where LPL methodology can be described. 

## Quick Start ##

### Preparation for using LPL methodology 

- Download and install [anaconda distribution](https://www.anaconda.com)

### Use of Liking Product Landscape

```python
import numpy as np
import pandas
from LPL import LikingProductLandscape

## Read the data
X = pandas.read_csv('Experiments/Exp2/data/wine.csv').values

# Read the overall liking and attributes perception
overall_liking = X[:,[10,16,22,28,34]]
sweetness = X[:,[5,11,17,23,29]]
acidity = X[:,[6,12,18,24,30]]
astringency = X[:,[7,13,19,25,31]]
body = X[:,[8,14,20,26,32]]
fruity = X[:,[9,15,21,27,33]]

# Liking Product Landscape
ol = overall_liking
jar = np.concatenate((sweetness,acidity,astringency,body,fruity),axis=1)
oljar = np.cocatenate((ol,jar),axis=1)

lpl = LikingProductLandscape(oljar,consumers_map='MDS',preference_map='SVM')
lpl.products_overall_liking(['W1','W2','W3','W4','W5'],overall_liking)
lpl.attribute('Sweetness',sweetness)
lpl.attribute('Acidity',acidity)
lpl.attribute('Astringency',astringency)
lpl.attribute('Body',body)
lpl.attribute('Fruity',fruity)
lpl.execute(filename='Experiments/Exp2/results/exp2_')
```
