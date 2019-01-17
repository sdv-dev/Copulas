<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“Copulas” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>


[![][pypi-img]][pypi-url]
[![][travis-img]][travis-url]

# Copulas

[travis-img]: https://travis-ci.org/DAI-Lab/Copulas.svg?branch=master
[travis-url]: https://travis-ci.org/DAI-Lab/Copulas
[pypi-img]: https://img.shields.io/pypi/v/copulas.svg
[pypi-url]: https://pypi.python.org/pypi/copulas

## Overview

A python library for building different types of [copulas](https://en.wikipedia.org/wiki/Copula_(probability_theory)) and using them for sampling.

- Free software: MIT license
- Documentation: https://DAI-Lab.github.io/Copulas

## Supported Copulas

### Bivariate

- Clayton
- Frank
- Gumbel

Accesible from `copulas.bivariate.copulas.Copula`

### Multivariate
- Gaussian [[+ info]](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Gaussian_copula)

Accesible from `copulas.multivariate.models.CopulaModel`


## Installation

### Install with pip

The easiest way to install Copulas is using `pip`

```
pip install copulas
```

### Install from sources

You can also clone the repository and install it from sources

```
git clone git@github.com:DAI-Lab/Copulas.git
cd Copulas
python setup.py install
```

## Data Requirements

This package works under the assumption that the data is perfectly clean, that means that:

- There are no missing values.
- All values are numerical

## Usage

In this library you can model univariate distributions and create copulas from a numeric dataset.
For this example, we will use the iris dataset in the data folder.

### Creating Univariate Distribution

First we will retrieve the data from the data folder and create a univariate distribution.
For this example, we will create a normal distribution. First type the following commands on
a python terminal.

```python
>>> from copulas.univariate.gaussian import GaussianUnivariate
>>> import pandas as pd
>>> data = pd.read_csv('data/iris.data.csv')
>>> data.head()
   feature_01  feature_02  feature_03  feature_04
0         5.1         3.5         1.4         0.2
1         4.9         3.0         1.4         0.2
2         4.7         3.2         1.3         0.2
3         4.6         3.1         1.5         0.2
4         5.0         3.6         1.4         0.2
```

Once we have the data, we can pass it into the GaussianUnivariate class.

```python
>>> feature1 = data['feature_01']
>>> gu = GaussianUnivariate()
>>> gu.fit(feature1)
>>> print(gu)
Distribution Type: Gaussian
Variable name: feature_01
Mean: 5.843333333333334
Standard deviation: 0.8253012917851409
```

Once you fit the distribution, you can get the pdf or cdf of data points and you can sample
from the distribution.

```python
>>> gu.probability_density(5)
0.2867858505472377
>>> gu.cumulative_distribution(5)
0.15342617720079227
>>> gu.sample(1)
array([6.14745446])
```

### Creating a Gaussian Copula

When you have a numeric data table, you can also create a copula and use it to sample from
the multivariate distribution. In this example, we will use a Gaussian Copula.

```python
>>> from copulas.multivariate.gaussian import GaussianMultivariate
>>> gc = GaussianMultivariate()
>>> gc.fit(data)
>>> print(gc)
feature_01
===============
Distribution Type: Gaussian
Variable name: feature_01
Mean: 5.843333333333334
Standard deviation: 0.8253012917851409

feature_02
===============
Distribution Type: Gaussian
Variable name: feature_02
Mean: 3.0540000000000003
Standard deviation: 0.4321465800705435

feature_03
===============
Distribution Type: Gaussian
Variable name: feature_03
Mean: 3.758666666666666
Standard deviation: 1.7585291834055212

feature_04
===============
Distribution Type: Gaussian
Variable name: feature_04
Mean: 1.1986666666666668
Standard deviation: 0.7606126185881716

Covariance matrix:
[[ 1.26935536  0.64987728  0.94166734 ... -0.57458312 -0.14548004
  -0.43589371]
 [ 0.64987728  0.33302068  0.4849735  ... -0.29401609 -0.06772633
  -0.21867228]
 [ 0.94166734  0.4849735   0.72674568 ... -0.42778472 -0.04608618
  -0.27836438]
 ...
 [-0.57458312 -0.29401609 -0.42778472 ...  0.2708685   0.0786054
   0.19208669]
 [-0.14548004 -0.06772633 -0.04608618 ...  0.0786054   0.17668562
   0.14455133]
 [-0.43589371 -0.21867228 -0.27836438 ...  0.19208669  0.14455133
   0.22229033]]
```

Once you have fit the copula, you can sample from it.

```python
gc.sample(5)
   feature_01  feature_02  feature_03  feature_04
0    5.529610    2.966947    3.162891    0.974260
1    5.708827    3.011078    3.407812    1.149803
2    4.623795    2.712284    1.283194    0.213796
3    5.952688    3.086259    4.088219    1.382523
4    5.360256    2.920929    2.844729    0.826919
```
