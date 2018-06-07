

# Copula Library
A python library for building different types of copulas and using them for sampling.

- Free software: MIT license
- Documentation: https://DAI-Lab.github.io/copulas

[travis-img]: https://travis-ci.org/DAI-Lab/copulas.svg?branch=master
[travis-url]: https://travis-ci.org/DAI-Lab/copulas
[pypi-img]: https://img.shields.io/pypi/v/copulas.svg
[pypi-url]: https://pypi.python.org/pypi/copulas
## Installation
You can create a virtual environment and install the dependencies using the following commands.
```bash
$ virtualenv venv --no-site-packages
$ source venv/bin/activate
$ pip install -r requirements.txt
```
## Usage
In this library you can model univariate distributions and create copulas from a numeric dataset. For this example, we will use the iris dataset in the data folder.
### Creating Univariate Distribution
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
### Creating a Gaussian Copula
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
Covariance matrix:  [[ 1.00671141 -0.11010327  0.87760486  0.82344326]
 [-0.11010327  1.00671141 -0.42333835 -0.358937  ]
 [ 0.87760486 -0.42333835  1.00671141  0.96921855]
 [ 0.82344326 -0.358937    0.96921855  1.00671141]]
Means:  [-3.315866100213801e-16, -7.8159700933611021e-16, 2.8421709430404008e-16, -2.3684757858670006e-16]
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
