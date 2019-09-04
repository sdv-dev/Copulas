# Release workflow

The process of releasing a new version involves several steps:

1. [Install Copulas from source](#install-copulas-from-source)

2. [Linting and tests](#linting-and-tests)

3. [Documentation](#documentation)

4. [HISTORY.md](#history.md)

5. [Distribution](#distribution)

6. [Integration with SDV](#integration-with-sdv)

6.1. [Install SDV from source](#install-sdv-from-source)

6.2. [Install from distribution](#install-from-distribution)

6.3. [Run SDV tests and README.md examples](#run-sdv-tests-and-readme.md-examples)

## Install Copulas from source

Clone the project and install the development requirements before start the release process. Alternatively, with your virtualenv activated.

```bash
git clone https://github.com/DAI-Lab/Copulas.git
cd Copulas
git checkout master
make install-develop
```

## Linting and tests

Execute ALL the tests and linting, tests must end with no errors:

```bash
make test-all
```

This command will use tox to execute the unittests with different environments, see tox.ini configuration.

To be able to run this you will need the differents python versions used in the tox.ini file.

At the end, you will see an output like this:

```
_____________________________________________ summary ______________________________________________
  py35: commands succeeded
  py36: commands succeeded
  lint: commands succeeded
  docs: commands succeeded
```

To run the tests over your python version:

```bash
make test && make lint
```

And you will see something like this:

```
============================ 169 passed, 1 skipped, 3 warnings in 7.10s ============================
flake8 copulas tests examples
isort -c --recursive copulas tests examples
```

The execution has finished with no errors, 1 test skipped and 3 warnings.
		
## Documentation

The documentation must be up to dates and generated with:

```bash
make view-docs
```

Read the documentation to ensure all the changes are reflected in the documentation.

Alternatively, you can simple generate the documentation using the command:

```bash
make docs
```

## HISTORY.md

HISTORY.md is updated with the issues of the milestone:

```
# History
	
## X.Y.Z (YYYY-MM-DD)
	
### New Features
	
* <ISSUE TITLE> - [Issue #<issue>](https://github.com/DAI-Lab/Copulas/issues/<issue>) by @resolver
	
### General Improvements
	
* <ISSUE TITLE> - [Issue #<issue>](https://github.com/DAI-Lab/Copulas/issues/<issue>) by @resolver
	
### Bug Fixed
	
* <ISSUE TITLE> - [Issue #<issue>](https://github.com/DAI-Lab/Copulas/issues/<issue>) by @resolver
```

The issue list per milestone can be found [here][milestones].

[milestones]: https://github.com/DAI-Lab/Copulas/milestones

## Distribution

Generate the distribution executing:

```bash
make dist
```

This will create a `dist` and `build` directories. The `dist` directory contains the library installer.

```
dist/
├── copulas-<version>-py2.py3-none-any.whl
└── copulas-<version>.tar.gz
```

Now, create a new virtualenv with the distributed file generated and run the README.md examples:

1. Create the copulas-test directory (out of the Copulas directory):

```bash
mkdir copulas-test
cd copulas-test
```

2. Create a new virtuelenv and activate it:

```bash
virtualenv -p $(which python3.6) .venv
source .venv/bin/activate
```

3. Install the wheel distribution:

```bash
pip install /path/to/copulas/dist/<copulas-distribution-version-any>.whl
```

4. Now you are ready to execute the README.md examples.

## Integration with SDV

### Install SDV from source

Clone the project and install the development requirements. Alternatively, with your virtualenv activated.

```bash
git clone https://github.com/HDI-Project/SDV
cd SDV
git checkout master
make install-develop
```

### Install from distribution

Install the Copulas version from the generated distribution file.

```bash
pip install /path/to/copulas/dist/<copulas-distribution-version-any>.whl
```

### Run SDV tests and README.md examples

Execute the SDV tests to ensure that the new distribution version works.

```bash
make test
```

Also, execute the SDV README.md examples.
