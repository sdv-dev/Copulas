.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

.PHONY: help
help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)


# CLEAN TARGETS

.PHONY: clean-build
clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

.PHONY: clean-pyc
clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

.PHONY: clean-docs
clean-docs: ## remove previously built docs
	rm -f docs/api/*.rst
	rm -rf docs/tutorials
	-$(MAKE) -C docs clean 2>/dev/null  # this fails if sphinx is not yet installed

.PHONY: clean-coverage
clean-coverage: ## remove coverage artifacts
	rm -f .coverage
	rm -f .coverage.*
	rm -fr htmlcov/

.PHONY: clean-test
clean-test: ## remove test artifacts
	rm -fr .tox/
	rm -fr .pytest_cache

.PHONY: clean
clean: clean-build clean-pyc clean-test clean-coverage clean-docs ## remove all build, test, coverage, docs and Python artifacts


# INSTALL TARGETS

.PHONY: install
install: clean-build clean-pyc ## install the package to the active Python's site-packages
	pip install .

.PHONY: install-test
install-test: clean-build clean-pyc ## install the package and test dependencies
	pip install .[test]

.PHONY: install-develop
install-develop: clean-build clean-pyc ## install the package in editable mode and dependencies for development
	pip install -e .[dev]


# LINT TARGETS

.PHONY: lint
lint: ## check style with flake8 and isort
	invoke lint

lint-docs: ## check docs formatting with doc8 and pydocstyle
	doc8 . docs/
	pydocstyle copulas/

.PHONY: fix-lint
fix-lint: ## fix lint issues using autoflake, autopep8, and isort
	find copulas tests -name '*.py' | xargs autoflake --in-place --remove-all-unused-imports --remove-unused-variables
	autopep8 --in-place --recursive --aggressive copulas tests
	isort --apply --atomic --recursive copulas tests


# TEST TARGETS

.PHONY: test-end-to-end
test-end-to-end: ## Tests to certify that all the functionalities from the library work as expected
	invoke end-to-end

.PHONY: test-numerical
test-numerical: ## Test that validate the functionality for the library from a numerical point of view
	invoke numerical

.PHONY: test-unit
test-unit: ## run the unit tests for copulas
	invoke unit

.PHONY: test-readme
test-readme: ## run the readme snippets
	invoke readme

.PHONY: test-tutorials
test-tutorials: ## run the tutorials notebooks
	invoke tutorials

.PHONY: test
test: test-unit test-numerical test-end-to-end test-tutorials test-readme ## run all the tests

.PHONY: test-all
test-all: ## test everything using tox
	tox -r

.PHONY: coverage
coverage: ## check code coverage quickly with the default Python
	coverage run --source copulas -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html


# DOCS TARGETS

.PHONY: docs
docs: clean-docs ## generate Sphinx HTML documentation, including API docs
	cp -r tutorials docs/tutorials
	sphinx-apidoc --separate -o docs/api/ copulas
	$(MAKE) -C docs html

.PHONY: view-docs
view-docs: ## view docs in browser
	$(BROWSER) docs/_build/html/index.html

.PHONY: serve-docs
serve-docs: ## compile the docs watching for changes
	watchmedo shell-command -W -R -D -p '*.rst;*.md' -c '$(MAKE) -C docs html' docs


# RELEASE TARGETS

.PHONY: dist
dist: clean ## builds source and wheel package
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

.PHONY: publish-confirm
publish-confirm:
	@echo "WARNING: This will irreversibly upload a new version to PyPI!"
	@echo -n "Please type 'confirm' to proceed: " \
		&& read answer \
		&& [ "$${answer}" = "confirm" ]

.PHONY: publish-test
publish-test: dist publish-confirm ## package and upload a release on TestPyPI
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

.PHONY: publish
publish: dist publish-confirm ## package and upload a release
	twine upload dist/*

.PHONY: git-merge-main-stable
git-merge-main-stable: ## Merge main into stable
	git checkout stable || git checkout -b stable
	git merge --no-ff main -m"make release-tag: Merge branch 'main' into stable"

.PHONY: git-merge-stable-main
git-merge-stable-main: ## Merge stable into main
	git checkout main
	git merge stable

.PHONY: git-push
git-push: ## Simply push the repository to github
	git push

.PHONY: git-push-tags-stable
git-push-tags-stable: ## Push tags and stable to github
	git push --tags origin stable

.PHONY: bumpversion-release
bumpversion-release: ## Bump the version to the next release
	bumpversion release

.PHONY: bumpversion-patch
bumpversion-patch: ## Bump the version to the next patch
	bumpversion --no-tag patch

.PHONY: bumpversion-candidate
bumpversion-candidate: ## Bump the version to the next candidate
	bumpversion candidate --no-tag

.PHONY: bumpversion-minor
bumpversion-minor: ## Bump the version the next minor skipping the release
	bumpversion --no-tag minor

.PHONY: bumpversion-major
bumpversion-major: ## Bump the version the next major skipping the release
	bumpversion --no-tag major

.PHONY: bumpversion-revert
bumpversion-revert: ## Undo a previous bumpversion-release
	git tag --delete $(shell git tag --points-at HEAD)
	git checkout main
	git branch -D stable

CLEAN_DIR := $(shell git status --short | grep -v ??)
CURRENT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD 2>/dev/null)
CURRENT_VERSION := $(shell grep "^current_version" setup.cfg | grep -o "dev[0-9]*")
CHANGELOG_LINES := $(shell git diff HEAD..origin/stable HISTORY.md 2>&1 | wc -l)

.PHONY: check-clean
check-clean: ## Check if the directory has uncommitted changes
ifneq ($(CLEAN_DIR),)
	$(error There are uncommitted changes)
endif

.PHONY: check-main
check-main: ## Check if we are in main branch
ifneq ($(CURRENT_BRANCH),main)
	$(error Please make the release from main branch\n)
endif

.PHONY: check-candidate
check-candidate: ## Check if a release candidate has been made
ifeq ($(CURRENT_VERSION),dev0)
	$(error Please make a release candidate and test it before atempting a release)
endif

.PHONY: check-history
check-history: ## Check if HISTORY.md has been modified
ifeq ($(CHANGELOG_LINES),0)
	$(error Please insert the release notes in HISTORY.md before releasing)
endif

.PHONY: check-release
check-release: check-clean check-candidate check-main check-history ## Check if the release can be made
	@echo "A new release can be made"

.PHONY: release
release: check-release git-merge-main-stable bumpversion-release git-push-tags-stable \
	publish git-merge-stable-main bumpversion-patch git-push

.PHONY: release-test
release-test: check-release git-merge-main-stable bumpversion-release bumpversion-revert

.PHONY: release-candidate
release-candidate: check-main publish bumpversion-candidate git-push

.PHONY: release-candidate-test
release-candidate-test: check-clean check-main publish-test
