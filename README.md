# imputena: impute missing values using Python

[![Build Status](https://travis-ci.com/macarro/imputena.svg?branch=master)](https://travis-ci.com/macarro/imputena)
[![Coverage Status](https://coveralls.io/repos/github/macarro/imputena/badge.svg?branch=master)](https://coveralls.io/github/macarro/imputena?branch=master)

Package that allows both automated and customized treatment of missing values
in datasets using Python

## Installation

Clone this repository or download and unzip it. At the project root directory,
run:

```ShellSession
pip install .
```

## Tests

The tests for the implemented functions are located in the test directory and
use the unittest package.

To execute all tests, run the following command at the project root directory:

```ShellSession
python -m unittest
```

To execute only the tests contained in a particular test class, for example
test_delete_listwise.py, run the following command at the project root
directory:

```ShellSession
python -m unittest test.test_delete_listwise
```

## Documentation

The documentation is generated using sphinx using the docstrings. To generate
it, run either of the following commands at the `docs` directory:

```ShellSession
make html
make latexpdf
```

The generated documentation will be located in `docs/build`.