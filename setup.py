import sys

from setuptools import setup
from setuptools import find_packages

version = '0.1.2'

# Please update tox.ini when modifying dependency version requirements
install_requires = [
    'pycrypto>=2.6.1',
    'requests',
    'setuptools>=1.0',
    'six',
    'cmd2>=0.6.9',
    'psutil',
    'pid>=2.0.1',
    'blessed>=1.14.1',
    'future',
    'coloredlogs',
    'scipy',
    'numpy',
    'bitstring',
    'bitarray_ph4',
    'ufx',
     # 'matplotlib',  # apt-get install python-tk
    'filelock',
    'repoze.lru'
]

dev_extras = [
    'nose',
    'pep8',
    'tox',
]

docs_extras = [
    'Sphinx>=1.0',  # autodoc_member_order = 'bysource', autodoc_default_flags
    'sphinx_rtd_theme',
    'sphinxcontrib-programoutput',
]

try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')
    long_description = long_description.replace("\r", '')

except(IOError, ImportError):
    import io
    with io.open('README.md', encoding="utf-8") as f:
        long_description = f.read()

setup(
    name='booltest',
    version=version,
    description='Booltest: Polynomial randomness tester',
    long_description=long_description,
    url='https://github.com/ph4r05/polynomial-distinguishers',
    author='Dusan Klinec',
    author_email='dusan.klinec@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Topic :: Security',
    ],

    packages=find_packages(),
    include_package_data=True,
    install_requires=install_requires,
    extras_require={
        'dev': dev_extras,
        'docs': docs_extras,
    },
    entry_points={
        'console_scripts': [
            'booltest = booltest.booltest_main:main',
        ],
    }
)
