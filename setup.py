# coding=utf-8
"""Setup file for distutils / pypi."""
try:
    from ez_setup import use_setuptools
    use_setuptools()
except ImportError:
    pass

from setuptools import setup, find_packages

setup(
    name='logistic-regression',
    version='0.1',
    author='Akbar Gumbira',
    author_email='akbargumbira@gmail.com',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    scripts=[],
    url='http://pypi.python.org/pypi/logistic-regression/',
    license='LICENSE.txt',
    description=('A simple logistic regression implementation for binary '
                 'classification.'),
    long_description=open('README.md').read(),
    install_requires=[
        "numpy"
    ]
)
