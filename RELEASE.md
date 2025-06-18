# Release workflow

The process of releasing a new version involves several steps:

1. [Install Copulas from source](#install-copulas-from-source)

2. [Linting and tests](#linting-and-tests)

3. [Documentation](#documentation)

4. [Make a release candidate](#make-a-release-candidate)

5. [Integration with SDV](#integration-with-sdv)

6. [Milestone](#milestone)

7. [HISTORY.md](#history.md)

8. [Distribution](#distribution)

9. [Making the release](#making-the-release)

9.1. [Tag and release to PyPI](#tag-and-release-to-pypi)

9.2. [Update the release on GitHub](#update-the-release-on-github)


## Install Copulas from source

Clone the project and install the development requirements before start the release process. Alternatively, with your virtualenv activated.

```bash
git clone https://github.com/sdv-dev/Copulas.git
cd Copulas
git checkout main
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
isort -c copulas tests examples
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

## Make a release candidate

1. On the Copulas GitHub page, navigate to the [Actions][actions] tab.
2. Select the `Release` action.
3. Run it on the main branch. Make sure `Release candidate` is checked and `Test PyPI` is not.
4. Check on [PyPI][copulas-pypi] to assure the release candidate was successfully uploaded.

[actions]: https://github.com/sdv-dev/Copulas/actions
[copulas-pypi]: https://pypi.org/project/copulas/#history

## Integration with SDV

### Create a branch on SDV to test the candidate

Before doing the actual release, we need to test that the candidate works with SDV. To do this, we can create a branch on SDV that points to the release candidate we just created using the following steps:

1. Create a new branch on the SDV repository.

```bash
git checkout -b test-copulas-X.Y.Z
```

2. Update the pyproject.toml to set the minimum version of Copulas to be the same as the version of the release. For example, 

```toml
'copulas>=X.Y.Z.dev0'
```

3. Push this branch. This should trigger all the tests to run.

```bash
git push --set-upstream origin test-copulas-X.Y.Z
```

4. Check the [Actions][sdv-actions] tab on SDV to make sure all the tests pass.

[sdv-actions]: https://github.com/sdv-dev/SDV/actions

## Milestone

It's important check that the git hub and milestone issues are up to date with the release.

You neet to check that:

- The milestone for the current release exists.
- All the issues closed since the latest release are associated to the milestone. If they are not, associate them
- All the issues associated to the milestone are closed. If there are open issues but the milestone needs to
  be released anyway, move them to the next milestone.
- All the issues in the milestone are assigned to at least one person.
- All the pull requests closed since the latest release are associated to an issue. If necessary, create issues
  and assign them to the milestone. Also assigne the person who opened the issue to them.

## HISTORY.md

Make sure HISTORY.md is updated with the issues of the milestone:

```
# History
	
## X.Y.Z (YYYY-MM-DD)
	
### New Features
	
* <ISSUE TITLE> - [Issue #<issue>](https://github.com/sdv-dev/Copulas/issues/<issue>) by @resolver
	
### General Improvements
	
* <ISSUE TITLE> - [Issue #<issue>](https://github.com/sdv-dev/Copulas/issues/<issue>) by @resolver
	
### Bug Fixed
	
* <ISSUE TITLE> - [Issue #<issue>](https://github.com/sdv-dev/Copulas/issues/<issue>) by @resolver
```

The issue list per milestone can be found [here][milestones].

[milestones]: https://github.com/sdv-dev/Copulas/milestones

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

## Making the release

At the end, we need to make the release. First, check if the release can be made:

```bash
make check-release
```

### Tag and release to PyPI

Once we are sure that the release can be made we can use different commands depending on
the type of release that we want to make:

* `make release`: This will relase a patch, which is the most common type of release. Use this
  when the changes are bugfixes or enhancements that do not modify the existing user API. Changes
  that modify the user API to add new features but that do not modify the usage of the previous
  features can also be released as a patch.
* `make release-minor`: This will release the next minor version. Use this if the changes modify
  the existing user API in any way, even if it is backwards compatible. Minor backwards incompatible
  changes can also be released as minor versions while the library is still in beta state.
  After the major version 1 has been released, minor version can only be used to add backwards
  compatible API changes.
* `make release-major`: This will release the next major version. Use this to if the changes modify
  the user API in a backwards incompatible way after the major version 1 has been released.


### Update the release on GitHub

Once the tag and the release to PyPI has been made, go to GitHub and edit the freshly created "tag" to
add the title and release notes, which should be exactly the same that we added to the HISTORY.md file.

Finaly, close the milestone and, if it does not exit, create the next one.
