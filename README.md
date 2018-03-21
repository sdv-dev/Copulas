# Copula Library
A python library for building different types of copulas and using them for sampling.
## Installation
You can create a virtual environment and install the dependencies using the following commands.
```bash
$ virtualenv venv --no-site-packages
$ source venv/bin/activate
$ pip install -r requirements.txt
```
## Usage
In this library you can model univariatee distributions and create copulas from a numeric dataset. For this example, we will use the iris dataset in the data folder.
###Creating Univariate Distribution
First we will retrieve the data from the data folder and create a univariate distribution. For this example, we will create a normal distribution. First type the following commands into a python terminal.
```bash
>>> from copulas.univariate.GaussianUnivariate import GaussianUnivariate
>>> import numpy as np
>>> import pandas as pd
>>> data = pd.read_csv('data/iris.data.csv')
>>> data
     feature_01  feature_02  feature_03  feature_04
0           5.1         3.5         1.4         0.2
1           4.9         3.0         1.4         0.2
2           4.7         3.2         1.3         0.2
3           4.6         3.1         1.5         0.2
4           5.0         3.6         1.4         0.2
5           5.4         3.9         1.7         0.4
6           4.6         3.4         1.4         0.3
7           5.0         3.4         1.5         0.2
8           4.4         2.9         1.4         0.2
9           4.9         3.1         1.5         0.1
...
```
Once we have the data, we can pass it into the GaussianUnivariate class.
```bash
>>> feature1 = data['feature_01']
>>> gu = GaussianUnivariate()
>>> gu.fit(feature1)
Distribution Type: Gaussian
mean =  5.843333333333335
standard deviation =  0.8253012917851409
max =  7.9
min =  4.3
```
Once you fit the distribution, you can get the pdf or cdf of data points and you can sample from the distribution.
```bash
>>> gu.get_pdf(5)
0.28678585054723732
>>> gu.get_cdf(5)
0.15342617720079199
>>> gu.sample(1)
array([ 6.14745446])
```
###Creating a Gaussian Copula
When you have a numeric data table, you can also create a copula and use it to sample from the multivariate distribution. In this example, we will use a Gaussian Copula.
```bash
>>> from copulas.multivariate.GaussianCopula import GaussianCopula
>>> gc = GaussianCopula()
>>> gc.fit(data)
Fitting Gaussian Copula
Distribution Type: Gaussian
Variable name:  feature_01
mean =  5.843333333333335
standard deviation =  0.8253012917851409
max =  7.9
min =  4.3
Distribution Type: Gaussian
Variable name:  feature_02
mean =  3.0540000000000007
standard deviation =  0.4321465800705435
max =  4.4
min =  2.0
Distribution Type: Gaussian
Variable name:  feature_03
mean =  3.7586666666666693
standard deviation =  1.7585291834055201
max =  6.9
min =  1.0
Distribution Type: Gaussian
Variable name:  feature_04
mean =  1.1986666666666672
standard deviation =  0.760612618588172
max =  2.5
min =  0.1
Copula Distribution:
     feature_01  feature_02  feature_03  feature_04
0     -0.900681    1.032057   -1.341272   -1.312977
1     -1.143017   -0.124958   -1.341272   -1.312977
2     -1.385353    0.337848   -1.398138   -1.312977
3     -1.506521    0.106445   -1.284407   -1.312977
4     -1.021849    1.263460   -1.341272   -1.312977
5     -0.537178    1.957669   -1.170675   -1.050031
6     -1.506521    0.800654   -1.341272   -1.181504
7     -1.021849    0.800654   -1.284407   -1.312977
8     -1.748856   -0.356361   -1.341272   -1.312977
9     -1.143017    0.106445   -1.284407   -1.444450
...
Covariance matrix:  [[ 1.26935536  0.64987728  0.94166734 ..., -0.57458312 -0.14548004
  -0.43589371]
 [ 0.64987728  0.33302068  0.4849735  ..., -0.29401609 -0.06772633
  -0.21867228]
 [ 0.94166734  0.4849735   0.72674568 ..., -0.42778472 -0.04608618
  -0.27836438]
 ..., 
 [-0.57458312 -0.29401609 -0.42778472 ...,  0.2708685   0.0786054
   0.19208669]
 [-0.14548004 -0.06772633 -0.04608618 ...,  0.0786054   0.17668562
   0.14455133]
 [-0.43589371 -0.21867228 -0.27836438 ...,  0.19208669  0.14455133
   0.22229033]]
Means:  [-1.4684549872375404e-15, -1.7763568394002505e-15, -1.4210854715202005e-15, -7.1054273576010023e-16]
```

Once you have fit the copula, you can sample from it. 
```bash
gc.sample(5)
   feature_01  feature_02  feature_03  feature_04
0    5.529610    2.966947    3.162891    0.974260
1    5.708827    3.011078    3.407812    1.149803
2    4.623795    2.712284    1.283194    0.213796
3    5.952688    3.086259    4.088219    1.382523
4    5.360256    2.920929    2.844729    0.826919
```