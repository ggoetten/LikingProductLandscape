# Liking Product Landscape #

Liking Product Landscape (LPL) can be easily used for understanding the comparison of products based on consumers' perceptions 
evaluations.

We are preparing a journal paper where LPL methodology can be described. 

## Quick Start ##

### Preparation for using LPL methodology 

- Download and install [anaconda distribution](https://www.anaconda.com)
- Know the fundamentals of [python](https://www.python.org/)

### An example of the using of Liking Product Landscape

First, it is needed to import the python libraries.
```python
import numpy as np
import pandas
```

The LPL.py file in this repository, which contains the source code of LPL methodology, needs to be downloaded and paste in the actual project folder. For using the LPL code in a new code file, we must add the next line:
```python
from LPL import LikingProductLandscape
```

It is recommended to capture the data in a .csv file, like the one in this repository Experiments/Exp2/data/wine.csv. For reading a .csv file the command pandas.read_csv can be used, specifically for the data in the experiment 2 we can use:
```python
X = pandas.read_csv('Experiments/Exp2/data/wine.csv').values
```

An important step is to read the consumers' evaluations from the file. It is recommended to use columns for evaluations and rows for consumers. Then, we can read the overall liking and the attributes evaluations indicating the column numbers. It is important to mention that first column corresponds to number 0. For example, for reading the evaluations of the data file Experiments/Exp2/data/wine.csv we use: 
```python
overall_liking = X[:,[10,16,22,28,34]]
sweetness = X[:,[5,11,17,23,29]]
acidity = X[:,[6,12,18,24,30]]
astringency = X[:,[7,13,19,25,31]]
body = X[:,[8,14,20,26,32]]
fruity = X[:,[9,15,21,27,33]]
```

Then, we need to organize the data that corresponds to overall liking (ol), jar (attributes perception) and oljar (ol+jar). 
```python
ol = overall_liking
jar = np.concatenate((sweetness,acidity,astringency,body,fruity),axis=1)
oljar = np.cocatenate((ol,jar),axis=1)
```

The parameters of LikingProductLandscape must be defined. The data that will be used, it could be ol, jar, or oljar. The consumers' map technique, it could be 'MDS', 'IPM' or 'PCA'. The preference map technique, it could be 'Danzart' or 'SVM'. The recommended parameters are the ones showed in the sample line code:
```python
lpl = LikingProductLandscape(oljar,consumers_map='MDS',preference_map='SVM')
```

Then, we need to specify the product names and send the overall liking evaluations with the command products_overall_liking. 
```python
lpl.products_overall_liking(['W1','W2','W3','W4','W5'],overall_liking)
```

In addition, the attributes names and their evaluations need to be sended using the command attribute, one per one. 
```python
lpl.attribute('Sweetness',sweetness)
lpl.attribute('Acidity',acidity)
lpl.attribute('Astringency',astringency)
lpl.attribute('Body',body)
lpl.attribute('Fruity',fruity)
```

Finally, we need to execute the LPL methodology and specify the folder where the files will be stored. 
```python
lpl.execute(filename='Experiments/Exp2/results/')
```

For running the whole code, the complete script is:
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
lpl.execute(filename='Experiments/Exp2/results/')
```
