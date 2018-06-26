#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

install_requires = [
    'numpy>=1.13.1',
    'pandas>=0.22.0',
    'scipy>=0.19.1',
    'exrex>=0.10.5',
    'matplotlib>=2.2.2'
]

tests_require = [
    'pytest>=3.4.2',
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
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    description="A python library for building different types of copulas and using them for sampling.",
    entry_points={
        'console_scripts': [
            'copulas=copulas.cli:main',
        ],
    },
    extras_require={
        'test': tests_require
    },
    install_requires=install_requires,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='copulas',
    name='copulas',
    packages=find_packages(include=['copulas', 'copulas.*']),
    setup_requires=setup_requires,
    test_suite='tests',
    tests_require=tests_require,
    url='https://github.com/DAI-Lab/Copulas',
    version='0.1.0',
    zip_safe=False,
)
