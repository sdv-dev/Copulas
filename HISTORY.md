# History

## 0.2.2 (2019-05-30)


### New Features

* Add `truncnorm` distribution and a generic wrapper for `scipy.rv_continous` distributions - 
  [Issue #27 ](https://github.com/DAI-Lab/Copulas/issues/27) by @amontanez, @csala and @ManuelAlvarezC
* Add `Independence` bivariate copulas - [Issue #46](https://github.com/DAI-Lab/Copulas/issues/46)
  by @aliciasun, @csala and @ManuelAlvarezC
* Add ability to handle constant data to univariate classes -
  [Issue #57](https://github.com/DAI-Lab/Copulas/issues/57) and
  [Issue #82](https://github.com/DAI-Lab/Copulas/issues/82) by @csala and @ManuelAlvarezC
* Improved documentation - [Issue #96](https://github.com/DAI-Lab/Copulas/issues/96)
  by @ManuelAlvarezC
* Add tests for analytics properties of copulas -
  [Issue #61](https://github.com/DAI-Lab/Copulas/issues/61) by @ManuelAlvarezC
* Add option to select seed on random number generator -
  [Issue #63](https://github.com/DAI-Lab/Copulas/issues/63) by @echo66 and @ManuelAlvarezC
* Add option on Vine copulas to select number of rows to sample -
  [Issue #77](https://github.com/DAI-Lab/Copulas/issues/77) by @ManuelAlvarezC

### Bugs fixed

* Fix bug on Vine copulas, that make it crash during the bivariate copula selection.
  [Issue #64](https://github.com/DAI-Lab/Copulas/issues/64) by @echo66 and @ManuelAlvarezC


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
