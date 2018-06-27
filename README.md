[![][pypi-img]][pypi-url]
[![][travis-img]][travis-url]

<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“Copulas” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>

# Copulas

[travis-img]: https://travis-ci.org/DAI-Lab/Copulas.svg?branch=master
[travis-url]: https://travis-ci.org/DAI-Lab/Copulas
[pypi-img]: https://img.shields.io/pypi/v/copulas.svg
[pypi-url]: https://pypi.python.org/pypi/copulas

## Overview

A python library for building different types of [copulas](https://en.wikipedia.org/wiki/Copula_(probability_theory)) and using them for sampling.

- Free software: MIT license
- Documentation: https://DAI-Lab.github.io/copulas

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

In this library you can model univariate distributions and create copulas from a numeric dataset. For this example, we will use the iris dataset in the data folder.

### Creating Univariate Distribution

First we will retrieve the data from the data folder and create a univariate distribution. For this example, we will create a normal distribution. First type the following commands into a python terminal.
```python
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
```python
>>> feature1 = data['feature_01']
>>> gu = GaussianUnivariate()
>>> gu.fit(feature1)
>>> print(gu)
Distribution Type: Gaussian
mean =  5.843333333333335
standard deviation =  0.8253012917851409
max =  7.9
min =  4.3
```
Once you fit the distribution, you can get the pdf or cdf of data points and you can sample from the distribution.
```python
>>> gu.get_pdf(5)
0.28678585054723732
>>> gu.get_cdf(5)
0.15342617720079199
>>> gu.sample(1)
array([ 6.14745446])
```
### Creating a Gaussian Copula
When you have a numeric data table, you can also create a copula and use it to sample from the multivariate distribution. In this example, we will use a Gaussian Copula.
```python
>>> from copulas.multivariate.GaussianCopula import GaussianCopula
>>> gc = GaussianCopula()
>>> gc.fit(data)
>>> print(gc)
feature_01
===============
Distribution Type: Gaussian
Variable name: feature_01
Mean: 5.843333333333334
Standard deviation: 0.8253012917851409
Max: 7.9
Min: 4.3

feature_02
===============
Distribution Type: Gaussian
Variable name: feature_02
Mean: 3.0540000000000003
Standard deviation: 0.4321465800705435
Max: 4.4
Min: 2.0

feature_03
===============
Distribution Type: Gaussian
Variable name: feature_03
Mean: 3.758666666666666
Standard deviation: 1.7585291834055212
Max: 6.9
Min: 1.0

feature_04
===============
Distribution Type: Gaussian
Variable name: feature_04
Mean: 1.1986666666666668
Standard deviation: 0.7606126185881716
Max: 2.5
Min: 0.1

Copula Distribution:
     feature_01  feature_02  feature_03  feature_04
0     -0.900681    1.032057   -1.341272   -1.312977
1     -1.143017   -0.124958   -1.341272   -1.312977
2     -1.385353    0.337848   -1.398138   -1.312977
3     -1.506521    0.106445   -1.284407   -1.312977
4     -1.021849    1.263460   -1.341272   -1.312977
5     -0.537178    1.957669   -1.170675   -1.050031
...

[150 rows x 4 columns]

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

Means:
[-3.315866100213801e-16, -7.815970093361102e-16, 2.842170943040401e-16, -2.3684757858670006e-16]

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

Release Workflow
----------------

The process of releasing a new version involves several steps combining both ``git`` and
``bumpversion`` which, briefly:

1. Merge what is in ``master`` branch into ``stable`` branch.
2. Update the version in ``setup.cfg``, ``copulas/__init__.py`` and ``HISTORY.md`` files.
3. Create a new TAG pointing at the correspoding commit in ``stable`` branch.
4. Merge the new commit from ``stable`` into ``master``.
5. Update the version in ``setup.cfg`` and ``copulas/__init__.py`` to open the next
   development interation.

**Note:** Before starting the process, make sure that ``HISTORY.md`` has a section titled
**Unreleased** with the list of changes that will be included in the new version, and that
these changes are committed and available in ``master`` branch.
Normally this is just a list of the Pull Requests that have been merged since the latest version.

Once this is done, just run the following commands::

    git checkout stable
    git merge --no-ff master    # This creates a merge commit
    bumpversion release   # This creates a new commit and a TAG
    git push --tags origin stable
    make release
    git checkout master
    git merge stable
    bumpversion --no-tag patch
    git push
