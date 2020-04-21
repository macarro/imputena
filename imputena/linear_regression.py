import pandas as pd
from sklearn import linear_model
import logging


def linear_regression(data=None, dependent=None, predictors=[], inplace=False):
    """Performs simple or multiple linear regression imputation on the data.
    First, the regression equation for the dependent variable given the
    predictor variables is computed. For this step, all rows that contain a
    missing value in either the dependent variable or any of the predictor
    variable is ignored via pairwise deletion. Then, missing valued in the
    dependent column in imputed using the regression equation. If, in the same
    row as a missing value in the dependent variable the value for any
    predictor variable is missing, that row does not get imputed.

    :param data: The data on which to perform the linear regression imputation.
    :type data: pandas.DataFrame
    :param dependent: The dependent variable in which the missing values
        should be imputed.
    :type dependent: String
    :param predictors: The predictor variables on which the dependent variable
        is dependent.
    :type predictors: array-like
    :param inplace: If True, do operation inplace and return None.
    :type inplace: bool, default False
    :return: The dataframe with linear regression imputation performed for the
        incomplete variable or None if inplace=True.
    :rtype: pandas.DataFrame o None
    :raises: TypeError, ValueError
    """
    # Check if data is a dataframe:
    if not isinstance(data, pd.DataFrame):
        raise TypeError('The data has to be a DataFrame.')
    # Check if the dependent variable is actually a column of the dataframe:
    if dependent not in data.columns:
        raise ValueError(
            '\'' + dependent + '\' is not a column of the data.')
    # Check if each of the predictor variables is actually a column of the
    # dataframe:
    for column in predictors:
        if column not in data.columns:
            raise ValueError(
                '\'' + column + '\' is not a column of the data.')
    # Perform the operation:
    if inplace:
        data.loc[:, :] = linear_regression_iter(data, dependent, predictors)
    else:
        res = linear_regression_iter(data, dependent, predictors)
    if not inplace:
        return res


def linear_regression_iter(
        data=None, dependent=None, predictors=[]):
    """Auxiliary function that performs (simple or multiple) linear
    regression on the data, for the dependent column only. In rows that
    contain a missing value for any predictor variable, the value of the
    dependent variable does not get imputed. The operation is always
    performed on a copy of the data, which is returned.

    :param data: The data on which to perform the linear regression imputation.
    :type data: pandas.DataFrame
    :param dependent: The dependent variable in which the missing values
        should be imputed.
    :type dependent: String
    :param predictors: The predictor variables on which the dependent variable
        is dependent.
    :type predictors: array-like
    :return: A copy of the dataframe with linear regression imputation
        performed for the incomplete variable.
    :rtype: pandas.DataFrame o None
    """
    # Perform pairwise deletion before calculating the regression
    data_pairwise_deleted = data.copy()
    variables = predictors.copy()
    variables.append(dependent)
    data_pairwise_deleted.dropna(subset=variables, inplace=True)
    # Calculate the regression:
    x = data_pairwise_deleted[predictors]
    y = data_pairwise_deleted[dependent]
    model = linear_model.LinearRegression()
    model.fit(x, y)
    # Extract the regression parameters from the model
    intercept = model.intercept_
    coefs = model.coef_
    # Log regression equation:
    eq = str(dependent) + ' = ' + str(intercept)
    for idx, coef in enumerate(coefs):
        eq += ' + ' + str(coef) + '*' + predictors[idx]
    logging.info('Regression equation: ' + eq)
    # Implementation using apply:
    return data.apply(
        lambda row: get_imputed_row(
            row, dependent, predictors, intercept, coefs),
        axis=1, result_type='broadcast')


def get_imputed_row(row, dependent, predictors, intercept, coefs):
    """Auxiliary function that receives a row of a DataFrame and returns the
    same row. If the row contains a missing value for the dependent variable,
    it gets imputed according to the regression equation specified by
    predictors, intercept and coefs.

    :param row: The row for which the missing value should be imputed
    :type row: pandas.Series
    :param dependent: The dependent variable for which the row might contain a
        missing value
    :type dependent: String
    :param predictors: The predictor variables on which the dependent variable
        is dependent.
    :type predictors: array-like
    :param intercept: The y-intercept of the regression equation.
    :type intercept: scalar
    :param coefs:  The coefficients of the regression equation, in the same
        order as the predictors.
    :type coefs: array-like
    :return: The row, with the missing value imputed if it contains one.
    :rtype: pandas.Series
    """
    res = row.copy()
    if pd.isnull(res[dependent]):
        value = intercept
        for idx, coef in enumerate(coefs):
            value += coef * row[predictors[idx]]
        res[dependent] = value
    return res
