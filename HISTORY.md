# History

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
