# History

## 0.2.4 (2019-12-23)

### New Features

* Allow creating copula classes directly - Issue [#117](https://github.com/DAI-Lab/Copulas/issues/117) by @csala

### General Improvements

* Remove `select_copula` from `Bivariate` - Issue [#118](https://github.com/DAI-Lab/Copulas/issues/118) by @csala

* Rename TruncNorm to TruncGaussian and make it non standard - Issue [#102](https://github.com/DAI-Lab/Copulas/issues/102) by @csala @JDTheRipperPC

### Bugs fixed

* Error on Frank and Gumble sampling - Issue [#112](https://github.com/DAI-Lab/Copulas/issues/112) by @csala

## 0.2.3 (2019-09-17)

### New Features

* Add support to Python 3.7 - Issue [#53](https://github.com/DAI-Lab/Copulas/issues/53) by @JDTheRipperPC

### General Improvements

* Document RELEASE workflow - Issue [#105](https://github.com/DAI-Lab/Copulas/issues/105) by @JDTheRipperPC

* Improve serialization of univariate distributions - Issue [#99](https://github.com/DAI-Lab/Copulas/issues/99) by @ManuelAlvarezC and @JDTheRipperPC

### Bugs fixed

* The method 'select_copula' of Bivariate return wrong CopulaType - Issue [#101](https://github.com/DAI-Lab/Copulas/issues/101) by @JDTheRipperPC

## 0.2.2 (2019-07-31)

### New Features

* `truncnorm` distribution and a generic wrapper for `scipy.rv_continous` distributions - Issue [#27](https://github.com/DAI-Lab/Copulas/issues/27) by @amontanez, @csala and @ManuelAlvarezC
* `Independence` bivariate copulas - Issue [#46](https://github.com/DAI-Lab/Copulas/issues/46) by @aliciasun, @csala and @ManuelAlvarezC
* Option to select seed on random number generator - Issue [#63](https://github.com/DAI-Lab/Copulas/issues/63) by @echo66 and @ManuelAlvarezC
* Option on Vine copulas to select number of rows to sample - Issue [#77](https://github.com/DAI-Lab/Copulas/issues/77) by @ManuelAlvarezC
* Make copulas accept both scalars and arrays as arguments - Issues [#85](https://github.com/DAI-Lab/Copulas/issues/85) and [#90](https://github.com/DAI-Lab/Copulas/issues/90) by @ManuelAlvarezC

### General Improvements

* Ability to properly handle constant data - Issues [#57](https://github.com/DAI-Lab/Copulas/issues/57) and [#82](https://github.com/DAI-Lab/Copulas/issues/82) by @csala and @ManuelAlvarezC
* Tests for analytics properties of copulas - Issue [#61](https://github.com/DAI-Lab/Copulas/issues/61) by @ManuelAlvarezC
* Improved documentation - Issue [#96](https://github.com/DAI-Lab/Copulas/issues/96) by @ManuelAlvarezC

### Bugs fixed

* Fix bug on Vine copulas, that made it crash during the bivariate copula selection - Issue [#64](https://github.com/DAI-Lab/Copulas/issues/64) by @echo66 and @ManuelAlvarezC

## 0.2.1 - Vine serialization

* Add serialization to Vine copulas.
* Add `distribution` as argument for the Gaussian Copula.
* Improve Bivariate Copulas code structure to remove code duplication.
* Fix bug in Vine Copulas sampling: 'Edge' object has no attribute 'index'
* Improve code documentation.
* Improve code style and linting tools configuration.

## 0.2.0 - Unified API

* New API for stats methods.
* Standarize input and output to `numpy.ndarray`.
* Increase unittest coverage to 90%.
* Add methods to load/save copulas.
* Improve Gaussian copula sampling accuracy.

## 0.1.1 - Minor Improvements

* Different Copula types separated in subclasses
* Extensive Unit Testing
* More pythonic names in the public API.
* Stop using third party elements that will be deprected soon.
* Add methods to sample new data on bivariate copulas.
* New KDE Univariate copula
* Improved examples with additional demo data.

## 0.1.0 - First Release

* First release on PyPI.
