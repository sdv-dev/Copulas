<p align="left">
<img width=20% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“sdv-dev” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>

[![Development Status](https://img.shields.io/badge/Development%20Status-2%20--%20Pre--Alpha-yellow)](https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha)
[![PyPi Shield](https://img.shields.io/pypi/v/copulas.svg)](https://pypi.python.org/pypi/copulas)
[![Travis CI Shield](https://travis-ci.org/sdv-dev/Copulas.svg?branch=master)](https://travis-ci.org/sdv-dev/Copulas)
[![Coverage Status](https://codecov.io/gh/sdv-dev/Copulas/branch/master/graph/badge.svg)](https://codecov.io/gh/sdv-dev/Copulas)
[![Downloads](https://pepy.tech/badge/copulas)](https://pepy.tech/project/copulas)

# Copulas

* License: [MIT](https://github.com/sdv-dev/Copulas/blob/master/LICENSE)
* Development Status: [Pre-Alpha](https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha)
* Documentation: https://sdv-dev.github.io/Copulas
* Homepage: https://github.com/sdv-dev/Copulas

# Overview

Copulas is a python library for building multivariate distributuions using
[copulas](https://en.wikipedia.org/wiki/Copula_%28probability_theory%29) and using them
for sampling. In short, you give a table of numerical data without missing values as a
2-dimensional `numpy.ndarray` and copulas models its distribution and using it to generate
new records, or analyze its statistical properties.

This repository contains multiple implementations of bivariate and multivariate copulas,
further functionality include:

* Most usual statistical functions from the underlying distribution.
* Built-in inverse-transform sampling method.
* Easy save and load of models.
* Create copulas directly from their parameters.

## Supported Copulas

### Bivariate copulas

* Clayton
* Frank
* Gumbel
* Independence

### Multivariate

* Gaussian [[+ info]](https://en.wikipedia.org/wiki/Copula_%28probability_theory%29#Gaussian_copula)
* D-Vine
* C-Vine
* R-Vine

# Install

## Requirements

**Copulas** has been developed and tested on [Python 3.5, and 3.6](https://www.python.org/downloads/)

Also, although it is not strictly required, the usage of a [virtualenv](https://virtualenv.pypa.io/en/latest/)
is highly recommended in order to avoid interfering with other software installed in the system where **Copulas**
is run.

These are the minimum commands needed to create a virtualenv using python3.6 for **Copulas**:

```bash
pip install virtualenv
virtualenv -p $(which python3.6) copulas-venv
```

Afterwards, you have to execute this command to have the virtualenv activated:

```bash
source copulas-venv/bin/activate
```

Remember about executing it every time you start a new console to work on **Copulas**!


## Install with pip

After creating the virtualenv and activating it, we recommend using
[pip](https://pip.pypa.io/en/stable/) in order to install **Copulas**:

```bash
pip install copulas
```

This will pull and install the latest stable release from [PyPi](https://pypi.org/).

## Install from source

Alternatively, with your virtualenv activated, you can clone the repository and install it from
source by running `make install` on the `stable` branch:

```bash
git clone git@github.com:sdv-dev/Copulas.git
cd Copulas
git checkout stable
make install
```

## Install for Development

If you want to contribute to the project, a few more steps are required to make the project ready
for development.

First, please head to [the GitHub page of the project](https://github.com/sdv-dev/Copulas)
and make a fork of the project under you own username by clicking on the **fork** button on the
upper right corner of the page.

Afterwards, clone your fork and create a branch from master with a descriptive name that includes
the number of the issue that you are going to work on:

```bash
git clone git@github.com:{your username}/Copulas.git
cd Copulas
git branch issue-xx-cool-new-feature master
git checkout issue-xx-cool-new-feature
```

Finally, install the project with the following command, which will install some additional
dependencies for code linting and testing.

```bash
make install-develop
```

Make sure to use them regularly while developing by running the commands `make lint` and
`make test`.

# Quickstart

In this short tutorial we will guide you through the a series of steps that will help you getting
started with the most basic usage of **Copulas** in order to generate samples from a simple
dataset.

**NOTE:** To be able to run this demo you will need to install the package from its sources.

## 1. Load the data

The first step is to load the data we will use to fit **Copulas**. In order to do so, we will
first import the module `pandas` and call its function `read_csv` with the path to our
example dataset.

In this case, we will load the `iris` dataset into a `pandas.DataFrame`.

```python
import pandas as pd
data = pd.read_csv('data/iris.data.csv')
```

This will be return us a dataframe with 4 columns:

```
              0    1    2
feature_01  5.1  4.9  4.7
feature_02  3.5  3.0  3.2
feature_03  1.4  1.4  1.3
feature_04  0.2  0.2  0.2
```

## 2. Create a Copula instance

The next step is to import Copulas and create an instance of the desired copulas.

To do so, we need to import the `copulas.multivariate.GaussianMultivariate` and call it, in order
to create a GaussianMultivariate instance with the default arguments:

```python
from copulas.multivariate import GaussianMultivariate
copula = GaussianMultivariate()
```

## 3. Fit the model

Once we have a **Copulas** instance, we can proceed to call its `fit` method passing the `data`
that we loaded bfore in order to start the fitting process:

```python
copula.fit(data)
```

## 4. Sample new data

After the model has been fitted, we are ready to generate new samples by calling the `sample`
method of the `Copulas` instance passing it the desired amount of samples:

```python
num_samples = 1000
samples = copula.sample(num_samples)
```

This will return a DataFrame with the same number of columns as the original data.

```
     feature_01  feature_02  feature_03  feature_04
0      6.855178    2.831508    6.664971    2.636296
1      5.185542    2.812182    1.906017    0.778288
2      5.289364    2.670617    4.612196    1.374458
3      5.055029    3.248253    2.247922    0.062830
4      5.048349    2.911649    4.704453    1.282351
```

The returned object, `samples`, is a `pandas.DataFrame` containing a table of synthetic data with
the same format as the input data and 1000 rows as we requested.

## 5. Load and save a model

For some copula models the fitting process can take a lot of time, so we probably would like to
avoid having to fit every we want to generate samples. Instead we can fit a model once, save it,
and load it every time we want to sample new data.

If we have a fitted model, we can save it by calling it's `save` method, that only takes
as argument the path where the model will be stored. Similarly, the `load` allows to load
a model stored on disk by passing as argument the path where the model is stored.

```python
model_path = 'mymodel.pkl'
copula.save(model_path)
```

Once the model is saved, it can be loaded back as a **Copulas** instance by using the `load`
method:

**NOTE**: In order to load a saved model, you need to load it using the same class that was used to save it.

```python
new_copula = GaussianMultivariate.load(model_path)
```

At this point we could use this model instance to generate more samples.

```python
new_samples = new_copula.sample(num_samples)
```

## 6. Extract and set parameters

In some cases it's more useful to obtain the parameters from a fitted copula than to save
and load from disk.

Once our copula is fitted, we can extract it's parameters using the `to_dict` method:

```python
copula_params = copula.to_dict()
```

This will return a dictionary containing all the copula parameters:

```
{
  "covariance": [
    [
      0.8362770742116029,
      -0.08653126313887204,
      0.680824746388168,
      0.6243889406272886
    ],
    [
      -0.08653126313887204,
      0.8788019924547759,
      -0.2489051243517041,
      -0.22041817349390733
    ],
    [
      0.680824746388168,
      -0.2489051243517041,
      0.699554023096054,
      0.660015366760104
    ],
    [
      0.6243889406272886,
      -0.22041817349390733,
      0.660015366760104,
      0.7215112866475774
    ]
  ],
  "univariates": [
    {
      "type": "copulas.univariate.base.Univariate",
      "fitted": true,
      "instance_type": "copulas.univariate.gaussian_kde.GaussianKDE",
      "lower": 0.15966936011068533,
      "upper": 12.040330639889316,
      "dataset": [
        [
          5.1,
          4.9,
          5.0,
          4.5,
          ...
          5.9
        ]
      ]
    },
    {
      "type": "copulas.univariate.base.Univariate",
      "fitted": true,
      "instance_type": "copulas.univariate.gaussian_kde.GaussianKDE",
      "lower": -0.16797155681086862,
      "upper": 6.567971556810869,
      "dataset": [
        [
          3.5,
          3.0,
          3.2,
          3.1,
          ...
          3.0
        ]
      ]
    },
    {
      "type": "copulas.univariate.base.Univariate",
      "fitted": true,
      "instance_type": "copulas.univariate.gaussian_kde.GaussianKDE",
      "lower": -7.8221020997613095,
      "upper": 15.72210209976131,
      "dataset": [
        [
          1.4,
          1.4,
          1.3,
          5.4,
          ...
          5.1
        ]
      ]
    },
    {
      "type": "copulas.univariate.base.Univariate",
      "fitted": true,
      "instance_type": "copulas.univariate.gaussian_kde.GaussianKDE",
      "lower": -3.7158037085042066,
      "upper": 6.315803708504207,
      "dataset": [
        [
          0.2,
          0.2,
          0.2,
          0.3,
          ...
          1.8
        ]
      ]
    }
  ],
  "columns": [
    "feature_01",
    "feature_02",
    "feature_03",
    "feature_04"
  ],
  "type": "copulas.multivariate.gaussian.GaussianMultivariate",
  "fitted": true,
  "distribution": "copulas.univariate.Univariate"
}
```

Once we have all the parameters we can create a new identical **Copula** instance by using the method `from_dict`:

```python
new_copula = GaussianMultivariate.from_dict(copula_params)
```

At this point we could use this model instance to generate more samples.

```python
new_samples = new_copula.sample(num_samples)
```

# What's next?

For more details about **Copulas** and all its possibilities and features, please check the
[documentation site](https://sdv-dev.github.io/Copulas/).

There you can learn more about [how to contribute to Copulas](https://sdv-dev.github.io/Copulas/contributing.html)
in order to help us developing new features or cool ideas.

# Credits

Copulas is an open source project from the Data to AI Lab at MIT which has been built and maintained
over the years by the following team:

* Manuel Alvarez <manuel@pythiac.com>
* Carles Sala <carles@pythiac.com>
* José David Pérez <jose@pythiac.com>
* (Alicia)Yi Sun <yis@mit.edu>
* Andrew Montanez <amontane@mit.edu>
* Kalyan Veeramachaneni <kalyan@csail.mit.edu>
* paulolimac <paulolimac@gmail.com>
* Kevin Alex Zhang <kevz@mit.edu>


## Related Projects

### SDV

[SDV](https://github.com/HDI-Project/SDV), for Synthetic Data Vault, is the end-user library for
synthesizing data in development under the [HDI Project](https://hdi-dai.lids.mit.edu/).
SDV allows you to easily model and sample relational datasets using Copulas thought a simple API.
Other features include anonymization of Personal Identifiable Information (PII) and preserving
relational integrity on sampled records.

### TGAN

[TGAN](https://github.com/sdv-dev/TGAN) is a GAN based model for synthesizing tabular data.
It's also developed by the [MIT's Data to AI Lab](https://sdv-dev.github.io/) and is under
active development.
