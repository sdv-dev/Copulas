#.dev0!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md', encoding='utf-8') as history_file:
    history = history_file.read()

install_requires = [
    "matplotlib>=3.4.0,<4;python_version>='3.7' and python_version<'3.10'",
    "matplotlib>=3.6.0,<4;python_version>='3.10'",
    "numpy>=1.20.0,<2;python_version<'3.10'",
    "numpy>=1.23.3,<2;python_version>='3.10'",
    "pandas>=1.1.3,<2;python_version<'3.10'",
    "pandas>=1.3.4,<2;python_version>='3.10' and python_version<'3.11'",
    "pandas>=1.5.0,<2;python_version>='3.11'",
    "scipy>=1.5.4,<2;python_version<'3.10'",
    "scipy>=1.9.2,<2;python_version>='3.10'",
]

development_requires = [
    # general
    'pip>=9.0.1',
    'bumpversion>=0.5.3,<0.6',
    'watchdog>=0.8.3,<0.11',

    # docs
    'm2r>=0.2.0,<0.3',
    'nbsphinx>=0.5.0,<0.7',
    'Sphinx>=1.7.1,<3',
    'sphinx_rtd_theme>=0.2.4,<0.5',

    # Jinja2>=3 makes the sphinx theme fail
    'Jinja2>=2,<3',

    # style check
    'flake8>=3.7.7,<4',
    'isort>=4.3.4,<5',
    'flake8-debugger>=4.0.0,<4.1',
    'flake8-mock>=0.3,<0.4',
    'flake8-mutable>=1.2.0,<1.3',
    'flake8-fixme>=1.1.1,<1.2',
    'pep8-naming>=0.12.1,<0.13',
    'dlint>=0.11.0,<0.12',
    'flake8-docstrings>=1.5.0,<2',
    'pydocstyle>=6.1.1,<6.2',
    'flake8-pytest-style>=1.5.0,<2',
    'flake8-comprehensions>=3.6.1,<3.7',
    'flake8-print>=4.0.0,<4.1',
    'flake8-expression-complexity>=0.0.9,<0.1',
    'flake8-multiline-containers>=0.0.18,<0.1',
    'pandas-vet>=0.2.2,<0.3',
    'flake8-builtins>=1.5.3,<1.6',
    'flake8-eradicate>=1.1.0,<1.2',
    'flake8-quotes>=3.3.0,<4',
    'flake8-variables-names>=0.0.4,<0.1',
    'flake8-sfs>=0.0.3,<0.1',
    'flake8-absolute-import>=1.0,<2',

    # fix style issues
    'autoflake>=1.1,<2',
    'autopep8>=1.4.3,<1.6',

    # distribute on PyPI
    'twine>=1.10.0,<4',
    'wheel>=0.30.0',

    # Advanced testing
    'coverage>=4.5.1,<6',
    'tox>=2.9.1,<4',
    'invoke',

    # Documentation style
    'doc8>=0.8.0,<0.9',

    # Large scale evaluation
    'urllib3>=1.20,<1.26',
    'tabulate>=0.8.3,<0.9',
    'boto3>=1.7.47,<1.10',
    'docutils>=0.10,<0.15'
]

tutorials_require = [
    'markupsafe<=2.0.1',
    'scikit-learn>=0.24,<1.2',
    'jupyter>=1.0.0,<2',
]

tests_require = [
    'pytest>=6.2.5,<7',
    'pytest-cov>=2.6.0,<3',
    'pytest-rerunfailures>=9.0.0,<10',
    'rundoc>=0.4.3,<0.5',
]

setup_requires = [
    'pytest-runner>=2.11.1',
]

setup(
    author='DataCebo, Inc.',
    author_email='info@sdv.dev',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: Free for non-commercial use',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    description='Create tabular synthetic data using copulas-based modeling.',
    extras_require={
        'tutorials': tutorials_require,
        'test': tests_require + tutorials_require,
        'dev': tests_require + development_requires + tutorials_require,
    },
    install_requires=install_requires,
    license='BSL-1.1',
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='copulas',
    name='copulas',
    packages=find_packages(include=['copulas', 'copulas.*']),
    python_requires='>=3.7,<3.12',
    setup_requires=setup_requires,
    test_suite='tests',
    tests_require=tests_require,
    url='https://github.com/sdv-dev/Copulas',
    version='0.8.0.dev0',
    zip_safe=False,
)
