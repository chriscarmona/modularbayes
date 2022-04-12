"""Install script for setuptools."""

import os
from setuptools import find_namespace_packages
from setuptools import setup

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def _get_version():
  with open(os.path.join(_CURRENT_DIR, 'modularbayes', '__init__.py')) as fp:
    for line in fp:
      if line.startswith('__version__') and '=' in line:
        version = line[line.find('=') + 1:].strip(' \'"\n')
        if version:
          return version
    raise ValueError('`__version__` not defined in `modularbayes/__init__.py`')


def _parse_requirements(path):

  with open(os.path.join(_CURRENT_DIR, path)) as f:
    return [
        line.rstrip()
        for line in f
        if not (line.isspace() or line.startswith('#'))
    ]


setup(
    name='modularbayes',
    version=_get_version(),
    url='https://github.com/chriscarmona/modularbayes',
    license="MIT",
    author='Chris Carmona',
    description=('Modular Bayesian Inference.'),
    long_description=open(os.path.join(_CURRENT_DIR, 'README.md')).read(),
    long_description_content_type='text/markdown',
    author_email='carmona@stats.ox.ac.uk',
    maintainer_email='carmona@stats.ox.ac.uk',
    keywords='modular bayesian inference cut smi posterior probability distribution',
    packages=find_namespace_packages(exclude=['*_test.py']),
    install_requires=_parse_requirements(
        os.path.join(_CURRENT_DIR, 'requirements', 'requirements.txt')),
    tests_require=_parse_requirements(
        os.path.join(_CURRENT_DIR, 'requirements', 'requirements-tests.txt')),
    zip_safe=False,  # Required for full installation.
    include_package_data=True,
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
