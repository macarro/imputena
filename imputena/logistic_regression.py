import pandas as pd
from sklearn import linear_model


def logistic_regression(
        data=None, dependent=None, predictors=[], inplace=False):
    """Performs logistic regression imputation on the data. First, the
    regression equation for the dependent variable given the predictor
    variables is computed. For this step, all rows that contain a missing
    value in either the dependent variable or any of the predictor variable
    is ignored via pairwise deletion. Then, missing valued in the dependent
    column in imputed using the regression equation. If, in the same row as
    a missing value in the dependent variable the value for any predictor
    variable is missing, that row does not get imputed.

    :param data: The data on which to perform the logistic regression
        imputation.
    :type data: pandas.DataFrame
    :param dependent: The dependent variable in which the missing values
        should be imputed.
    :type dependent: String
    :param predictors: The predictor variables on which the dependent variable
        is dependent.
    :type predictors: array-like
    :param inplace: If True, do operation inplace and return None.
    :type inplace: bool, default False
    :return: The dataframe with logistic regression imputation performed for
        the incomplete variable or None if inplace=True.
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
    # Perform pairwise deletion before calculating the regression
    data_pairwise_deleted = data.copy()
    variables = predictors.copy()
    variables.append(dependent)
    data_pairwise_deleted.dropna(subset=variables, inplace=True)
    # Calculate the regression:
    x = data_pairwise_deleted[predictors]
    y = data_pairwise_deleted[dependent]
    model = linear_model.LogisticRegression()
    model.fit(x, y)
    print(type(model))
    # Implementation using apply:
    if inplace:
        data.loc[:, :] = data.apply(
            lambda row: get_imputed_row(
                row, dependent, predictors, model),
            axis=1, result_type='broadcast')
    else:
        res = data.apply(
            lambda row: get_imputed_row(
                row, dependent, predictors, model),
            axis=1, result_type='broadcast')
        # Return the imputed data, or None if inplace:
        if inplace:
            return None
        else:
            return res


def get_imputed_row(row, dependent, predictors, model):
    """Auxiliary function that receives a row of a DataFrame and returns the
    same row. If the row contains a missing value for the dependent variable,
    it gets imputed according to the regression model.

    :param row: The row for which the missing value should be imputed
    :type row: pandas.Series
    :param dependent: The dependent variable for which the row might contain a
        missing value
    :type dependent: String
    :param predictors: The predictor variables on which the dependent variable
        is dependent.
    :type predictors: array-like
    :param model: The logistic regression model.
    :type model: sklearn.linear_model._logistic.LogisticRegression
    :return: The row, with the missing value imputed if it contains one.
    :rtype: pandas.Series
    """
    res = row.copy()
    if pd.isnull(res[dependent]) and not row[predictors].isnull().any():
        value = model.predict(row[predictors].values.reshape(1, -1))
        res[dependent] = value[0]
    return res
