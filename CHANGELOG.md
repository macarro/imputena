# Changelog for imputena

## 0.3 (unreleased)

### Added

* K-nearest neighbors imputation
* Sequential regression multiple imputation
* Multiple imputation by chained equations
* Get information about the applicable and the recommended treatments for a
 given dataset
* Impute a dataset automatically with the best method using
 `impute_by_recommended`
* Perform linear regression imputation on all columns with missing values by
 calling linear_regression() without specifying a dependent column.
 
 ### Changed

* Code refactoring and style improvements

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
