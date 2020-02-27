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
