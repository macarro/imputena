# imputena

[![Build Status](https://travis-ci.com/macarro/imputena.svg?branch=master)](https://travis-ci.com/macarro/imputena)

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

