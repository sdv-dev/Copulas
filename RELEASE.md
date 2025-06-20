# Release workflow

The process of releasing a new version involves several steps:

1. [Install Copulas from source](#install-copulas-from-source)

2. [Linting and tests](#linting-and-tests)

3. [Documentation](#documentation)

4. [Make a release candidate](#make-a-release-candidate)

5. [Integration with SDV](#integration-with-sdv)

6. [Milestone](#milestone)

7. [Update HISTORY](#update-history)

8. [Check the release](#check-the-release)

8. [Update stable branch and bump version](#update-stable-branch-and-bump-version)

10. [Create the Release on GitHub](#create-the-release-on-github)

11. [Close milestone and create new milestone](#close-milestone-and-create-new-milestone)

## Install Copulas from source

Clone the project and install the development requirements before start the release process. Alternatively, with your virtualenv activated.

```bash
git clone https://github.com/sdv-dev/Copulas.git
cd Copulas
git checkout main
make install-develop
```

## Linting and tests

Execute the tests and linting. The tests must end with no errors:

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
  - You should see X.Y.ZdevN PRE-RELEASE

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

It's important check that the GitHub and milestone issues are up to date with the release.

You neet to check that:

- The milestone for the current release exists.
- All the issues closed since the latest release are associated to the milestone. If they are not, associate them
- All the issues associated to the milestone are closed. If there are open issues but the milestone needs to
  be released anyway, move them to the next milestone.
- All the issues in the milestone are assigned to at least one person.
- All the pull requests closed since the latest release are associated to an issue. If necessary, create issues
  and assign them to the milestone. Also assigne the person who opened the issue to them.

## Update HISTORY
Run the [Release Prep](https://github.com/sdv-dev/Copulas/actions/workflows/prepare_release.yml) workflow. This workflow will create a pull request with updates to HISTORY.md

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

Put the pull request up for review and get 2 approvals to merge into `main`.

## Check the release
Check if the release can be made:

```bash
make check-release
```

## Update stable branch and bump version
The `stable` branch needs to updated with the changes from `main` and the verison needs to be bumped.
Depending on the type of release, run one of the following:

* `make release`: This will release a patch, which is the most common type of release. Use this when the changes are bugfixes or enhancements that do not modify the existing user API. Changes that modify the user API to add new features but that do not modify the usage of the previous features can also be released as a patch.
* `make release-minor`: This will release the next minor version. Use this if the changes modify the existing user API in any way, even if it is backwards compatible. Minor backwards incompatible changes can also be released as minor versions while the library is still in beta state. After the major version 1 has been released, minor version can only be used to add backwards compatible API changes.
* `make release-major`: This will release the next major version. Use this to if the changes modify the user API in a backwards incompatible way after the major version 1 has been released.

Running one of these will **push commits directly** to `main`.
At the end, you should see the 2 commits on `main` on (from oldest to newest):
- `make release-tag: Merge branch 'main' into stable`
- `Bump version: X.Y.Z.devN → X.Y.Z`

## Create the Release on GitHub

After the update to HISTORY.md is merged into `main` and the version is bumped, it is time to [create the release GitHub](https://github.com/sdv-dev/Copulas/releases/new).
- Create a new tag with the version number with a v prefix (e.g. v0.3.1)
- The target should be the `main` branch
- Release title is the same as the tag (e.g. v0.3.1)
- This is not a pre-release (`Set as a pre-release` should be unchecked)

Click `Publish release`, which will kickoff the release workflow and automatically upload the package to public PyPI.

The release workflow will create a pull request and auto-merge it into `main` that bumps to the next development release (e.g. 0.12.3 → 0.12.4.dev0).

## Close milestone and create new milestone

Finaly, **close the milestone** and, if it does not exist, **create the next milestone**.

