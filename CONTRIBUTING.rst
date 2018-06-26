.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/DAI-Lab/copulas/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

Copulas could always use more documentation, whether as part of the
official Copulas docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/DAI-Lab/copulas/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `copulas` for local development.

1. Fork the `copulas` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/copulas.git

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed, this is how you set up your fork for local development::

    $ mkvirtualenv copulas
    $ cd copulas/
    $ python setup.py develop

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass flake8 and the
   tests, including testing other Python versions with tox::

    $ make lint
    $ pytest
    $ tox

   To get flake8 and tox, just pip install them into your virtualenv.

6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.
3. The pull request should work for Python3.4, 3.5 and 3.6, and for PyPy. Check
   https://travis-ci.org/DAI-Lab/copulas/pull_requests
   and make sure that the tests pass for all supported Python versions.

Tips
----

To run a subset of tests::

    $ pytest tests.test_copulas

Deploying
---------

The process of releasing a new version involves several steps combining both ``git`` and
``bumpversion`` which, briefly:

1. Merge what is in ``master`` branch into ``stable`` branch.
2. Update the version in ``setup.cfg``, ``copulas/__init__.py`` and ``HISTORY.md`` files.
3. Create a new TAG pointing at the correspoding commit in ``stable`` branch.
4. Merge the new commit from ``stable`` into ``master``.
5. Update the version in ``setup.cfg`` and ``copulas/__init__.py`` to open the next
   development interation.

**Note:** Before starting the process, make sure that ``HISTORY.md`` has a section titled
**Unreleased** with the list of changes that will be included in the new version, and that
these changes are committed and available in ``master`` branch.
Normally this is just a list of the Pull Requests that have been merged since the latest version.

Once this is done, just run the following commands::

    git checkout stable
    git merge --no-ff master    # This creates a merge commit
    bumpversion release   # This creates a new commit and a TAG
    git push --tags origin stable
    make release
    git checkout master
    git merge stable
    bumpversion --no-tag patch
    git push