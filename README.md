<div align="center">
<br/>
<p align="center">
    <i>This repository is part of <a href="https://sdv.dev">The Synthetic Data Vault Project</a>, a project from <a href="https://datacebo.com">DataCebo</a>.</i>
</p>

[![Development Status](https://img.shields.io/badge/Development%20Status-2%20--%20Pre--Alpha-yellow)](https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha)
[![PyPi Shield](https://img.shields.io/pypi/v/copulas.svg)](https://pypi.python.org/pypi/copulas)
[![Downloads](https://pepy.tech/badge/copulas)](https://pepy.tech/project/copulas)
[![Unit Tests](https://github.com/sdv-dev/Copulas/actions/workflows/unit.yml/badge.svg)](https://github.com/sdv-dev/Copulas/actions/workflows/unit.yml)
[![Coverage Status](https://codecov.io/gh/sdv-dev/Copulas/branch/master/graph/badge.svg)](https://codecov.io/gh/sdv-dev/Copulas)

<div align="left">
<br/>
<p align="center">
<a href="https://github.com/sdv-dev/Copulas">
<img align="center" width=40% src="https://github.com/sdv-dev/SDV/blob/master/docs/images/Copulas-DataCebo.png"></img>
</a>
</p>
</div>

</div>

# Overview

**Copulas** is a Python library for modeling multivariate distributions and sampling from them
using [copula functions](https://en.wikipedia.org/wiki/Copula_%28probability_theory%29).
Given a table containing numerical data, we can use Copulas to learn the distribution and
later on generate new synthetic rows following the same statistical properties.

| Important Links                               |                                                                      |
| --------------------------------------------- | -------------------------------------------------------------------- |
| :computer: **[Website]**                      | Check out the SDV Website for more information about the project.    |
| :orange_book: **[SDV Blog]**                  | Regular publshing of useful content about Synthetic Data Generation. |
| :book: **[Documentation]**                    | Quickstarts, User and Development Guides, and API Reference.         |
| :octocat: **[Repository]**                    | The link to the Github Repository of this library.                   |
| :scroll: **[License]**                        | The entire ecosystem is published under the MIT License.             |
| :keyboard: **[Development Status]**           | This software is in its Pre-Alpha stage.                             |
| [![][Slack Logo] **Community**][Community]    | Join our Slack Workspace for announcements and discussions.          |
| [![][MyBinder Logo] **Tutorials**][Tutorials] | Run the SDV Tutorials in a Binder environment.                       |

[Website]: https://sdv.dev
[SDV Blog]: https://sdv.dev/blog
[Documentation]: https://sdv.dev/SDV
[Repository]: https://github.com/sdv-dev/Copulas
[License]: https://github.com/sdv-dev/Copulas/blob/master/LICENSE
[Development Status]: https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha
[Slack Logo]: https://github.com/sdv-dev/SDV/blob/master/docs/images/slack.png
[Community]: https://join.slack.com/t/sdv-space/shared_invite/zt-gdsfcb5w-0QQpFMVoyB2Yd6SRiMplcw
[MyBinder Logo]: https://github.com/sdv-dev/SDV/blob/master/docs/images/mybinder.png
[Tutorials]: https://mybinder.org/v2/gh/sdv-dev/SDV/master?filepath=tutorials

# Features

Some of the features provided by this library include:

* A variety of distributions for modeling univariate data.
* Multiple Archimedean copulas for modeling bivariate data.
* Gaussian and Vine copulas for modeling multivariate data.
* Automatic selection of univariate distributions and bivariate copulas.

## Supported Distributions

### Univariate

* Beta
* Gamma
* Gaussian
* Gaussian KDE
* Log-Laplace
* Student T
* Truncated Gaussian
* Uniform

### Archimedean Copulas (Bivariate)

* Clayton
* Frank
* Gumbel

### Multivariate

* Gaussian Copula
* D-Vine
* C-Vine
* R-Vine

# Install

**Copulas** is part of the **SDV** project and is automatically installed alongside it. For
details about this process please visit the [SDV Installation Guide](
https://sdv.dev/SDV/getting_started/install.html)

Optionally, **Copulas** can also be installed as a standalone library using the following commands:

**Using `pip`:**

```bash
pip install copulas
```

**Using `conda`:**

```bash
conda install -c conda-forge copulas
```

For more installation options please visit the [Copulas installation Guide](INSTALL.md)

# Quickstart

In this short quickstart, we show how to model a multivariate dataset and then generate
synthetic data that resembles it.

```python3
import warnings
warnings.filterwarnings('ignore')

from copulas.datasets import sample_trivariate_xyz
from copulas.multivariate import GaussianMultivariate
from copulas.visualization import compare_3d

# Load a dataset with 3 columns that are not independent
real_data = sample_trivariate_xyz()

# Fit a gaussian copula to the data
copula = GaussianMultivariate()
copula.fit(real_data)

# Sample synthetic data
synthetic_data = copula.sample(len(real_data))

# Plot the real and the synthetic data to compare
compare_3d(real_data, synthetic_data)
```

The output will be a figure with two plots, showing what both the real and the synthetic
data that you just generated look like:

![Quickstart](docs/images/quickstart.png)


# What's next?

For more details about **Copulas** and all its possibilities and features, please check the
[documentation site](https://sdv.dev/Copulas/).

There you can learn more about [how to contribute to Copulas](https://sdv.dev/Copulas/contributing.html)
in order to help us developing new features or cool ideas.

# Credits

Copulas is an open source project from the Data to AI Lab at MIT which has been built and
maintained over the years by the following team:

* Manuel Alvarez <manuel@pythiac.com>
* Carles Sala <csala@mit.edu>
* (Alicia) Yi Sun <yis@mit.edu>
* José David Pérez <jose@pythiac.com>
* Kevin Alex Zhang <kevz@mit.edu>
* Andrew Montanez <amontane@mit.edu>
* Gabriele Bonomi <gbonomib@gmail.com>
* Kalyan Veeramachaneni <kalyan@csail.mit.edu>
* Iván Ramírez <rollervan@gmail.com>
* Felipe Alex Hofmann <fealho@gmail.com>
* paulolimac <paulolimac@gmail.com>
* nazar-ivantsiv <nazar.ivantsiv@gmail.com>

---


<div align="center">
<a href="https://datacebo.com"><img align="center" width=40% src="https://github.com/sdv-dev/SDV/blob/master/docs/images/DataCebo.png"></img></a>
</div>
<br/>
<br/>

[The Synthetic Data Vault Project](https://sdv.dev) was first created at MIT's [Data to AI Lab](
https://dai.lids.mit.edu/) in 2016. After 4 years of research and traction with enterprise, we
created [DataCebo](https://datacebo.com) in 2020 with the goal of growing the project.
Today, DataCebo is the proud developer of SDV, the largest ecosystem for
synthetic data generation & evaluation. It is home to multiple libraries that support synthetic
data, including:

* 🔄 Data discovery & transformation. Reverse the transforms to reproduce realistic data.
* 🧠 Multiple machine learning models -- ranging from Copulas to Deep Learning -- to create tabular,
  multi table and time series data.
* 📊 Measuring quality and privacy of synthetic data, and comparing different synthetic data
  generation models.

[Get started using the SDV package](https://sdv.dev/SDV/getting_started/install.html) -- a fully
integrated solution and your one-stop shop for synthetic data. Or, use the standalone libraries
for specific needs.
