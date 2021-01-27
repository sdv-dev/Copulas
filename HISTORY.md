# History

## v0.4.0 - 2021-01-27

This release introduces a few changes to optimize processing speed by re-implementing
the Gaussian KDE pdf to use vectorized root finding methods and also adding the option
to subsample the data during univariate selection.

### General Improvements

* Make `gaussian_kde` faster - Issue [#200](https://github.com/sdv-dev/Copulas/issues/200) by @k15z and @fealho
* Use sub-sampling in `select_univariate` - Issue [#183](https://github.com/sdv-dev/Copulas/issues/183) by @csala

## v0.3.3 - 2020-09-18

### General Improvements

* Use `corr` instead of `cov` in the GaussianMultivariate - Issue [#195](https://github.com/sdv-dev/Copulas/issues/195) by @rollervan
* Add arguments to GaussianKDE - Issue [#181](https://github.com/sdv-dev/Copulas/issues/181) by @rollervan

### New Features

* Log Laplace Distribution - Issue [#188](https://github.com/sdv-dev/Copulas/issues/188) by @rollervan

## v0.3.2 - 2020-08-08

### General Improvements

* Support Python 3.8 - Issue [#185](https://github.com/sdv-dev/Copulas/issues/185) by @csala
* Support scipy >1.3 - Issue [#180](https://github.com/sdv-dev/Copulas/issues/180) by @csala

### New Features

* Add Uniform Univariate - Issue [#179](https://github.com/sdv-dev/Copulas/issues/179) by @rollervan

## v0.3.1 - 2020-07-09

### General Improvements

* Raise numpy version upper bound to 2 - Issue [#178](https://github.com/sdv-dev/Copulas/issues/178) by @csala

### New Features

* Add Student T Univariate - Issue [#172](https://github.com/sdv-dev/Copulas/issues/172) by @gbonomib

### Bug Fixes

* Error in Quickstarts : Unknown projection '3d' - Issue [#174](https://github.com/sdv-dev/Copulas/issues/174) by @csala

## v0.3.0 - 2020-03-27

Important revamp of the internal implementation of the project, the testing
infrastructure and the documentation by Kevin Alex Zhang @k15z, Carles Sala
@csala and Kalyan Veeramachaneni @kveerama

### Enhancements

* Reimplementation of the existing Univariate distributions.
* Addition of new Beta and Gamma Univariates.
* New Univariate API with automatic selection of the optimal distribution.
* Several improvements and fixes on the Bivariate and Multivariate Copulas implementation.
* New visualization module with simple plotting patterns to visualize probability distributions.
* New datasets module with toy datasets sampling functions.
* New testing infrastructure with end-to-end, numerical and large scale testing.
* Improved tutorials and documentation.

## v0.2.5 - 2020-01-17

### General Improvements

* Convert import_object to get_instance - Issue [#114](https://github.com/sdv-dev/Copulas/issues/114) by @JDTheRipperPC

## v0.2.4 - 2019-12-23

### New Features

* Allow creating copula classes directly - Issue [#117](https://github.com/sdv-dev/Copulas/issues/117) by @csala

### General Improvements

* Remove `select_copula` from `Bivariate` - Issue [#118](https://github.com/sdv-dev/Copulas/issues/118) by @csala
* Rename TruncNorm to TruncGaussian and make it non standard - Issue [#102](https://github.com/sdv-dev/Copulas/issues/102) by @csala @JDTheRipperPC

### Bugs fixed

* Error on Frank and Gumble sampling - Issue [#112](https://github.com/sdv-dev/Copulas/issues/112) by @csala

## v0.2.3 - 2019-09-17

### New Features

* Add support to Python 3.7 - Issue [#53](https://github.com/sdv-dev/Copulas/issues/53) by @JDTheRipperPC

### General Improvements

* Document RELEASE workflow - Issue [#105](https://github.com/sdv-dev/Copulas/issues/105) by @JDTheRipperPC
* Improve serialization of univariate distributions - Issue [#99](https://github.com/sdv-dev/Copulas/issues/99) by @ManuelAlvarezC and @JDTheRipperPC

### Bugs fixed

* The method 'select_copula' of Bivariate return wrong CopulaType - Issue [#101](https://github.com/sdv-dev/Copulas/issues/101) by @JDTheRipperPC

## v0.2.2 - 2019-07-31

### New Features

* `truncnorm` distribution and a generic wrapper for `scipy.rv_continous` distributions - Issue [#27](https://github.com/sdv-dev/Copulas/issues/27) by @amontanez, @csala and @ManuelAlvarezC
* `Independence` bivariate copulas - Issue [#46](https://github.com/sdv-dev/Copulas/issues/46) by @aliciasun, @csala and @ManuelAlvarezC
* Option to select seed on random number generator - Issue [#63](https://github.com/sdv-dev/Copulas/issues/63) by @echo66 and @ManuelAlvarezC
* Option on Vine copulas to select number of rows to sample - Issue [#77](https://github.com/sdv-dev/Copulas/issues/77) by @ManuelAlvarezC
* Make copulas accept both scalars and arrays as arguments - Issues [#85](https://github.com/sdv-dev/Copulas/issues/85) and [#90](https://github.com/sdv-dev/Copulas/issues/90) by @ManuelAlvarezC

### General Improvements

* Ability to properly handle constant data - Issues [#57](https://github.com/sdv-dev/Copulas/issues/57) and [#82](https://github.com/sdv-dev/Copulas/issues/82) by @csala and @ManuelAlvarezC
* Tests for analytics properties of copulas - Issue [#61](https://github.com/sdv-dev/Copulas/issues/61) by @ManuelAlvarezC
* Improved documentation - Issue [#96](https://github.com/sdv-dev/Copulas/issues/96) by @ManuelAlvarezC

### Bugs fixed

* Fix bug on Vine copulas, that made it crash during the bivariate copula selection - Issue [#64](https://github.com/sdv-dev/Copulas/issues/64) by @echo66 and @ManuelAlvarezC

## v0.2.1 - Vine serialization

* Add serialization to Vine copulas.
* Add `distribution` as argument for the Gaussian Copula.
* Improve Bivariate Copulas code structure to remove code duplication.
* Fix bug in Vine Copulas sampling: 'Edge' object has no attribute 'index'
* Improve code documentation.
* Improve code style and linting tools configuration.

## v0.2.0 - Unified API

* New API for stats methods.
* Standarize input and output to `numpy.ndarray`.
* Increase unittest coverage to 90%.
* Add methods to load/save copulas.
* Improve Gaussian copula sampling accuracy.

## v0.1.1 - Minor Improvements

* Different Copula types separated in subclasses
* Extensive Unit Testing
* More pythonic names in the public API.
* Stop using third party elements that will be deprected soon.
* Add methods to sample new data on bivariate copulas.
* New KDE Univariate copula
* Improved examples with additional demo data.

## v0.1.0 - First Release

* First release on PyPI.
