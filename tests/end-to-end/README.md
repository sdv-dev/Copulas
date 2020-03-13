# End-to-end Tests

This folder contains a set of black-box tests designed to certify that all the functionalities
from the library work as expected from a user api perspective.

## Folder Structure

The structure of this folder and sub-folders should be the same as the main library, having one
sub-folder here for each sub-folder within the `copulas` directory.

For example, if the folders contained within `copulas` are `univariate`, `bivariate` and
`multivariate`, the three same folders should also exist within `test/end-to-end`.

Inside each folder, a `test_xyz.py` module should exist for each `xyz.py` module within the
`copulas` directory.

For example, if the modules `gaussian.py`, `tree.py` and `vine.py` exist within the folder
`copulas/multivariate`, the modules `test_gaussian.py`, `test_tree.py` and `test_vine.py`
should also exist within `tests/end-to-end/multivariate`.

## End-to-end Test Guidelines

1. End to end tests should be implemented using the final user API, having individual test
methods for each functionality that wants to be tested.

2. Test inputs should be as close as possible to the real data that the user will be using.
Whenever possible, the datasets from `copulas/datasets` should be used (real or simulated).

3. Mocks should be avoided as much as possible, letting the test use third party tools or
resources.

4. Some minimal validation of the outputs must be performed, but the main goal of these tests
is not to validate the outputs numerically, but only to certify that the software does not
crash when run. For this reason, types of the outputs and shapes must be checked, but exact
numerical matches must not.
