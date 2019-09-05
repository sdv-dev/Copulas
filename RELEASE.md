# Release workflow

The process of releasing a new version involves several steps:

1. [Install Copulas from source](#install-copulas-from-source)

2. [Linting and tests](#linting-and-tests)

3. [Documentation](#documentation)

4. [Milestone](#milestone)

5. [HISTORY.md](#history.md)

6. [Distribution](#distribution)

7. [Integration with SDV](#integration-with-sdv)

7.1. [Install SDV from source](#install-sdv-from-source)

7.2. [Install from distribution](#install-from-distribution)

7.3. [Run SDV tests and README.md examples](#run-sdv-tests-and-readme.md-examples)

8. [Making the release](#making-the-release)

8.1. [Release](#release)

8.2. [Patch](#patch)

8.3. [Minnor](#minnor)

8.4. [Major](#major)



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

To be able to run this you will need the different python versions used in the tox.ini file.

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

The documentation must be up to date and generated with:

```bash
make view-docs
```

Read the documentation to ensure all the changes are reflected in the documentation.

Alternatively, you can simply generate the documentation using the command:

```bash
make docs
```

## Milestone

It's important check that the milestone exists.

Also, all the pull requests in the milestone has been closed and are related to one issue.

All the issues in the milestone must be closed and every issue assigned to one (or more) person.

If there are any issue in the milestone not closed should be reassigned in a new milestone.

## HISTORY.md

Make sure HISTORY.md is updated with the issues of the milestone:

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

## Making the release

At the end, we need to make the release, first check if the release can be made:

```bash
make check-release
```

Once we are sure that the release can be made we can make different releases:

### Release

This command will marge master into stable and bumpversion patch.

```bash
make bumpversion-release
```

### Patch

This command will merge stable to master and make a bumpversion patch.

```bash
make bumpversion-patch
```

### Minnor

This command will bump the version the next minnor skipping the release.

```bash
make bumpversion-minor
```

### Major

This command will bump the version the next major skipping the release.

```bash
make bumpversion-major
```

<br/>

Next, go to GitHub and edit the "tag" to add the release notes of the release.

And finaly, close the milestone.
