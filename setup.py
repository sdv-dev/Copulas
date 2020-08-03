#.dev0!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

install_requires = [
    'numpy>=1.13.1,<2',
    'pandas>=0.22.0,<2',
    'scipy>=1.2,<2',
    'matplotlib>=2.2.2,<3.2.2',
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

    # style check
    'flake8>=3.7.7,<4',
    'isort>=4.3.4,<5',

    # fix style issues
    'autoflake>=1.1,<2',
    'autopep8>=1.4.3,<2',

    # distribute on PyPI
    'twine>=1.10.0,<4',
    'wheel>=0.30.0',

    # Advanced testing
    'coverage>=4.5.1,<6',
    'tox>=2.9.1,<4',

    # Documentation style
    'doc8>=0.8.0,<0.9',
    'pydocstyle>=3.0.0,<4',

    # Large scale evaluation
    'tabulate>=0.8.3,<0.9',
    'boto3>=1.7.47,<1.10',
    'docutils>=0.10,<0.15'
]

tutorials_require = [
    'scikit-learn>=0.22,<0.23',
    'jupyter>=1.0.0,<2',
]

tests_require = [
    'pytest>=3.4.2,<6',
    'pytest-cov>=2.6.0,<3',
    'rundoc>=0.4.3,<0.5',
]

setup_requires = [
    'pytest-runner>=2.11.1',
]

setup(
    author="MIT Data To AI Lab",
    author_email='dailabmit@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="A python library for building different types of copulas and using them for sampling.",
    entry_points={
        'console_scripts': [
            'copulas=copulas.cli:main',
        ],
    },
    extras_require={
        'tutorials': tutorials_require,
        'test': tests_require + tutorials_require,
        'dev': tests_require + development_requires + tutorials_require,
    },
    install_requires=install_requires,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='copulas',
    name='copulas',
    packages=find_packages(include=['copulas', 'copulas.*']),
    python_requires='>=3.5,<3.8',
    setup_requires=setup_requires,
    test_suite='tests',
    tests_require=tests_require,
    url='https://github.com/sdv-dev/Copulas',
    version='0.3.2.dev0',
    zip_safe=False,
)
