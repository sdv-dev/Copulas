<p align="left">
<img width=20% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“sdv-dev” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>



[![PyPi Shield](https://img.shields.io/pypi/v/copulas.svg)](https://pypi.python.org/pypi/copulas)
[![Travis CI Shield](https://travis-ci.org/sdv-dev/Copulas.svg?branch=master)](https://travis-ci.org/sdv-dev/Copulas)
[![Coverage Status](https://codecov.io/gh/sdv-dev/Copulas/branch/master/graph/badge.svg)](https://codecov.io/gh/sdv-dev/Copulas)
[![Downloads](https://pepy.tech/badge/copulas)](https://pepy.tech/project/copulas)


# Copulas

* License: [MIT](https://github.com/sdv-dev/Copulas/blob/master/LICENSE)
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
