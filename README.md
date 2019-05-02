<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“Copulas” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>

[![][pypi-img]][pypi-url]
[![][travis-img]][travis-url]

[travis-img]: https://travis-ci.org/DAI-Lab/Copulas.svg?branch=master
[travis-url]: https://travis-ci.org/DAI-Lab/Copulas
[pypi-img]: https://img.shields.io/pypi/v/copulas.svg
[pypi-url]: https://pypi.python.org/pypi/copulas

# Copulas

A python library for building multivariate distributuions with [copulas](https://en.wikipedia.org/wiki/Copula_(probability_theory)) and using them for sampling.

* Free software: MIT license
* Documentation: [https://DAI-Lab.github.io/Copulas](https://DAI-Lab.github.io/Copulas)

# Overview

This repository contains multiple implementations of bivariate and multivariate copulas, 

* Most usual statistical functions from the underlying distribution.
* Built-in inverse-transform sampling method.
* Easy save and load of fitted copulas.
* Create copulas directly from their parameters.

## Data Format

This package works under the assumption that the data is perfectly clean, that means that:

* Has no missing or invalid values.
* Has columns of types `int` or `float`.

# Concepts

## Probability

We call **probability** `P` to the measure assigned to the chance of an event happening. For example, in a dice, there are 6 sides, each with the same chance of being on top.

If we consider `0` to be **impossible** and `1` **absolute certain**, we can explain its probability like this:

```
Table of values for probability P

 ·    -> 1/6
 :    -> 1/6
 :·   -> 1/6
 ::   -> 1/6
 :·:  -> 1/6
 :::  -> 1/6
```

## Random variable

A **random variable** `X` is a function mapping elements from the sample space (in our case, the dice sides) into ℝ.

In our case we have:

```
Table of values for random variable X and their probability P
      X       P
 ·    ->   1  ->  1/6
 :    ->   2  ->  1/6
 :·   ->   3  ->  1/6
 ::   ->   4  ->  1/6
 :·:  ->   5  ->  1/6
 :::  ->   6  ->  1/6
```

## Distribution

A **distribution** is a function that describes the behavior of a **random variable**, like rolling a dice, and the probability of events related to them.

Usually a distribution is presented as a function F: ℝ -> [0, 1], called the **cumulative distribution function** or **cdf**, that has the following properties:

* Is strictly **non-decreasing**
* Is **right-continous**
* It's limit to negative infinity exists and is 0.
* It's limit to positive infinite exists and is 1.

Below we can see the cdf of the distribution of rolling a standard, 6 sided, dice:

![](docs/images/dice_cdf.png)

We can see as the cumulative probability raises by steps of 1/6 at each integer between 1 and 6,
as those are the only values that can appear.

## Types of distributions

There are as many different distributions as different random phenomenon, but usually we classify them using this three aspects:

* Continuity: We call a random variable a **continous random variable** if it's `cdf` is continuous, that it have no steps. Otherwise, we call it **discrete random variable**.
  In the example of the dice, we have discrete random variable.
* Dimensionality: When a random variable represents the behavior of a single random phenomenon, we call it a **univariate distribution**, analogously we define **bivariate** and **multivariate** distribution.
* Type: Most distribution have a type, defined by its behavior, some of the most common types of distributions are: **uniform**, **gaussian**, **exponential**,...

### Copulas

Copulas are multivariate distributions whose marginals are uniform. Using them with distributions to model the marginals they allow us to generate
**multivariate random variables** for any kind of phenomena

## Supported Copulas

### Bivariate copulas

* Clayton
* Frank
* Gumbel
* Independence

### Multivariate

* Gaussian [[+ info]](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Gaussian_copula)
* Vines

# Getting started

## Installation

The simplest and recommended way to install **Copulas** is using `pip`

```
pip install copulas
```

For development, you can also clone the repository and install it from sources

```
git clone git@github.com:DAI-Lab/Copulas.git
cd Copulas
python setup.py install
```

# Usage

In this short tutorial we will guide you through the a series of steps that will help you getting
started with the most basic usage of **Copulas** in order to generate samples from a simple
dataset.

**NOTE:** To be able to run this demo you will need to install the package from its sources.

## 1. Load the data

The first step is to load the data we will use to fit **Copulas**. In order to do so, we will first import the module `pandas` and call its function `read_csv` with the path to our example dataset.

In this case, we will load the `iris` dataset into a `pandas.DataFrame`.

```
>>> import pandas as pd
>>> data = pd.read_csv('data/iris.data.csv')
>>> data.head(3).T

              0    1    2
feature_01  5.1  4.9  4.7
feature_02  3.5  3.0  3.2
feature_03  1.4  1.4  1.3
feature_04  0.2  0.2  0.2
```

## 2. Create a Copula instance

The next step is to import Copulas and create an instance of the desired copulas.

To do so, we need to import the `copulas.multivariate.GaussianMultivariate` and call it, in order to create a GaussianMultivariate instance with the default arguments:

```
>>> from copulas.multivariate import GaussianMultivariate
>>> copula = GaussianMultivariate()
```

## 3. Fit the model

Once we have a **Copulas** instance, we can proceed to call its `fit` method passing the `data` that we loaded bfore in order to start the fitting process:

```
>>> copula.fit(data)
```

## 4. Sample new data

After the model has been fitted, we are ready to generate new samples by calling the `sample`
method of the `Copulas` instance passing it the desired amount of samples:

```
>>> num_samples = 1000
>>> samples = copula.sample(num_samples)
>>> samples.head(3).T

                   0         1         2
feature_01  7.534814  7.255292  5.723322
feature_02  2.723615  2.959855  3.282245
feature_03  6.465199  6.896618  2.658393
feature_04  2.267646  2.442479  1.109811
```

The returned object, `samples`, is a `pandas.DataFrame` containing a table of synthetic data with the same format as the input data and 1000 rows as we requested.

## 5. Load and save a model

For some copula models the fitting process can take a lot of time, so we probably would like to
avoid having to fit every we want to generate samples. Instead we can fit a model once, save it,
and load it every time we want to sample new data.

If we have a fitted model, we can save it by calling it's `save` method, that only takes
as argument the path where the model will be stored. Similarly, the `load` allows to load
a model stored on disk by passing as argument the path where the model is stored.

```
>>> model_path = 'mymodel.pkl'
>>> copula.save(model_path)
```

Once the model is saved, it can be loaded back as a **Copulas** instance by using the `load`
method:

**NOTE**: In order to load a saved model, you need to load it using the same class that was used to save it.

```
>>> new_copula = GaussianMultivariate.load(model_path)
```

At this point we could use this model instance to generate more samples.

```
>>> new_samples = new_copula.sample(num_samples)
>>> new_samples.head(3).T

                   0         1         2
feature_01  4.834213  5.441848  4.802118
feature_02  2.488793  2.499855  2.770923
feature_03  3.379794  5.181586  2.552305
feature_04  1.345214  2.101377  1.001049
```

## 6. Extract parameters and reconstruct instance from parameters

In some cases it's more useful to obtain the parameters from a fitted copula than to save and load from disk.

Once our copula is fitted, we can extract it's parameters using the `to_dict` method:

```
>>> copula_params = copula.to_dict()
>>> copula_params
{'covariance': [[1.006711409395973,
   -0.11010327176239859,
   0.877604856347186,
   0.8234432550696282],
  [-0.11010327176239859,
   1.006711409395972,
   -0.4233383520816992,
   -0.3589370029669185],
  [0.877604856347186,
   -0.4233383520816992,
   1.006711409395973,
   0.9692185540781538],
  [0.8234432550696282,
   -0.3589370029669185,
   0.9692185540781538,
   1.006711409395974]],
 'distribs': {'feature_01': {'type': 'copulas.univariate.gaussian.GaussianUnivariate',
   'fitted': True,
   'constant_value': None,
   'mean': 5.843333333333334,
   'std': 0.8253012917851409},
  'feature_02': {'type': 'copulas.univariate.gaussian.GaussianUnivariate',
   'fitted': True,
   'constant_value': None,
   'mean': 3.0540000000000003,
   'std': 0.4321465800705435},
  'feature_03': {'type': 'copulas.univariate.gaussian.GaussianUnivariate',
   'fitted': True,
   'constant_value': None,
   'mean': 3.758666666666666,
   'std': 1.7585291834055212},
  'feature_04': {'type': 'copulas.univariate.gaussian.GaussianUnivariate',
   'fitted': True,
   'constant_value': None,
   'mean': 1.1986666666666668,
   'std': 0.7606126185881716}},
 'type': 'copulas.multivariate.gaussian.GaussianMultivariate',
 'fitted': True,
 'distribution': 'copulas.univariate.gaussian.GaussianUnivariate'}
```

Once we have all the parameters we can create a new identical **Copula** instance by using the method `from_dict`:

```
new_copula = GaussianMultivariate.from_dict(copula_params)
```

At this point we could use this model instance to generate more samples.

```
>>> new_samples = new_copula.sample(num_samples)
>>> new_samples.head(3).T

                   0         1         2
feature_01  6.009206  6.653476  5.802923
feature_02  2.848561  2.771476  2.948189
feature_03  4.092759  5.612561  3.865684
feature_04  1.384638  2.043285  1.476101
```
