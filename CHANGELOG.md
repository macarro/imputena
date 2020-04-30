# Changelog

## 0.3 (unreleased)

### Added

* K-nearest neighbors imputation
* Perform linear regression imputation on all columns with missing values by
 calling linear_regression() without specifying a dependent column.

## [0.2](https://github.com/macarro/imputena/releases/tag/v0.2) (2020-04-23)

### Added

* Linear regression imputation
* Stochastic regression imputation
* Logistic regression imputation

### Changed

* `seasonal_interpolation` now raises a ValueError if the value of
 `dec_model` is not 'multiplicative' or 'additive'.
* Code refactoring and style improvements

## [0.1](https://github.com/macarro/imputena/releases/tag/v0.1) (2020-04-12)

### Added

* Listwise deletion
* Pairwise deletion
* Dropping variables
* Random sample imputation
* Random hot-deck imputation
* LOCF
* NOCB
* Most frequent substitution
* Mean and median substitution
* Constant value imputation
* Random value imputation
* Interpolation
* Interpolation with seasonal adjustment
