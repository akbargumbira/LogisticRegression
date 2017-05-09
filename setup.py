# coding=utf-8
"""Setup file for distutils / pypi."""
try:
    from ez_setup import use_setuptools
    use_setuptools()
except ImportError:
    pass

from setuptools import setup, find_packages

setup(
    name='log_regression',
    version='0.1',
    author='Akbar Gumbira',
    py_modules=['log_regression'],
    author_email='akbargumbira@gmail.com',
    description=('A simple logistic regression implementation for binary '
                 'classification.'),
    long_description=open('README.md').read(),
    install_requires=[
        "numpy"
    ]
)
