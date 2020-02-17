# Overview
**Copulas** is a Python library for modeling multivariate distributions and sampling from them
using [copula functions](https://en.wikipedia.org/wiki/Copula_%28probability_theory%29). Given 
a table containing numerical data, we can use Copulas to learn the distribution, analyze its 
statistical properties, and generate synthetic rows.

Some of the features provided by this library include:

 * A variety of distributions for modeling univariate data.
 * Multiple Archimedean copulas for modeling bivariate data.
 * Gaussian and Vine copulas for modeling multivariate data.
 * Automatic selection of univariate distributions and bivariate copulas.

# Installation
**Copulas** has been developed and tested on Python 3.5, 3.6, and 3.7.

### From PyPi
The simplest way to install Copulas is using *pip*:

```bash
pip install copulas
```

### From Source

```bash
git clone git@github.com:sdv-dev/Copulas.git
cd Copulas
git checkout stable
make install
```

# Getting Started

In this short tutorial we will guide you through the a series of steps that will help you getting
started with the most basic usage of **Copulas** in order to generate samples from a simple
dataset.

**NOTE:** To be able to run this demo you will need to install the package from its sources.

### 1. Load the data

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

### 2. Create a Copula instance

The next step is to import Copulas and create an instance of the desired copulas.

To do so, we need to import the `copulas.multivariate.GaussianMultivariate` and call it, in order
to create a GaussianMultivariate instance with the default arguments:

```python
from copulas.multivariate import GaussianMultivariate
copula = GaussianMultivariate()
```

### 3. Fit the model

Once we have a **Copulas** instance, we can proceed to call its `fit` method passing the `data`
that we loaded bfore in order to start the fitting process:

```python
copula.fit(data)
```

### 4. Sample new data

After the model has been fitted, we are ready to generate new samples by calling the `sample`
method of the `Copulas` instance passing it the desired amount of samples:

```python
num_samples = 1000
samples = copula.sample(num_samples)
```

This will return a DataFrame with the same number of columns as the original data.

```
                   0         1         2
feature_01  7.534814  7.255292  5.723322
feature_02  2.723615  2.959855  3.282245
feature_03  6.465199  6.896618  2.658393
feature_04  2.267646  2.442479  1.109811
```

The returned object, `samples`, is a `pandas.DataFrame` containing a table of synthetic data with
the same format as the input data and 1000 rows as we requested.

### 5. Load and save a model

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

### 6. Extract and set parameters

In some cases it's more useful to obtain the parameters from a fitted copula than to save
and load from disk.

Once our copula is fitted, we can extract it's parameters using the `to_dict` method:

```python
copula_params = copula.to_dict()
```

This will return a dictionary containing all the copula parameters:

```
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

```python
new_copula = GaussianMultivariate.from_dict(copula_params)
```

At this point we could use this model instance to generate more samples.

```python
new_samples = new_copula.sample(num_samples)
```
