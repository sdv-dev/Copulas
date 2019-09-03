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

        git clone https://github.com/DAI-Lab/Copulas.git

		cd Copulas

		git checkout master

		make install-develop

## Linting and tests

Execute ALL the tests and linting, tests must end with no errors:

		make test-all
		
## Documentation

The documentation must be up to dates and generated with:

        make view-docs

Read the documentation to ensure all the changes are reflected in the documentation.

## HISTORY.md

HISTORY.md is updated with the issues of the milestone:

		# History

		## X.Y.Z (YYYY-MM-DD)

		### New Features

		* <ISSUE TITLE> - [Issue #<issue>](https://github.com/DAI-Lab/Copulas/issues/<issue>) by @resolver

		### General Improvements

		* <ISSUE TITLE> - [Issue #<issue>](https://github.com/DAI-Lab/Copulas/issues/<issue>) by @resolver

		### Bug Fixed

		* <ISSUE TITLE> - [Issue #<issue>](https://github.com/DAI-Lab/Copulas/issues/<issue>) by @resolver

## Distribution

Generate the distribution executing:

        make dist

Now, create a new virtualenv with the distributed file generated and run the README.md examples.

## Integration with SDV

### Install SDV from source

Clone the project and install the development requirements. Alternatively, with your virtualenv activated.

        git clone https://github.com/HDI-Project/SDV

		cd SDV

        git checkout master

		make install-develop

### Install from distribution

Install the Copulas version from the generated distribution file.

        pip install /path/to/copulas/dist/<copulas-distribution-version-any>.whl

### Run SDV tests and README.md examples

Execute the SDV tests to ensure that the new distribution version works.

        make test

Also, execute the SDV README.md examples.
