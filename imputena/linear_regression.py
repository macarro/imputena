import pandas as pd
import numpy as np
from sklearn import linear_model
import logging


def linear_regression(
        data=None, dependent=None, predictors=None, regressions='available',
        noise=False, inplace=False):
    """Performs simple or multiple linear regression imputation on the data.
    First, the regression equation for the dependent variable given the
    predictor variables is computed. For this step, all rows that contain a
    missing value in either the dependent variable or any of the predictor
    variable is ignored via pairwise deletion. Then, missing valued in the
    dependent column in imputed using the regression equation. If, in the same
    row as a missing value in the dependent variable the value for any
    predictor variable is missing, a regression model based on all available
    predictors in calculated just to impute those values where the
    predictor(s) are missing. This behavior can be changed by assigning to
    the parameter regressions the value 'complete'. In this case, rows in
    which a predictor variable is missing do not get imputed. If stochastic
    regression imputation should be performed, set noise=True. In this
    case, a random value is chosen from a normal distribution with the width
    of the standard error of the regression model and added to the imputed
    value. If the parameter predictors is omitted, all variables other than
    the dependent are used as predictors. If the parameter dependent is
    omitted, the operation is performed on all columns that contain missing
    values.

    :param data: The data on which to perform the linear regression imputation.
    :type data: pandas.DataFrame
    :param dependent: The dependent variable in which the missing values
        should be imputed.
    :type dependent: String, optional
    :param predictors: The predictor variables on which the dependent variable
        is dependent.
    :type predictors: array-like, optional
    :param regressions: If 'available': Impute missing values by modeling a
        regression based on all available predictors if some predictors have
        missing values themselves. If 'complete': Only impute with a
        regression model based on all predictors and leave missing values in
        rows in which some predictor value is missing itself unimputed.
    :type regressions: {'available', 'complete'}, default 'available'
    :param noise: Whether to add noise to the imputed values (stochastic
        regression imputation)
    :type noise: bool, default False
    :param inplace: If True, do operation inplace and return None.
    :type inplace: bool, default False
    :return: The dataframe with linear regression imputation performed for the
        incomplete variable(s) or None if inplace=True.
    :rtype: pandas.DataFrame or None
    :raises: TypeError, ValueError
    """
    # Check if data is a dataframe:
    if not isinstance(data, pd.DataFrame):
        raise TypeError('The data has to be a DataFrame.')
    # Check if the dependent variable is actually a column of the dataframe:
    if dependent is not None and dependent not in data.columns:
        raise ValueError(
            '\'' + dependent + '\' is not a column of the data.')
    # Check if each of the predictor variables is actually a column of the
    # dataframe:
    if predictors is not None:
        for column in predictors:
            if column not in data.columns:
                raise ValueError(
                    '\'' + column + '\' is not a column of the data.')
    # Assign value to do_available_regressions
    if regressions == 'available':
        do_available_regressions = True
    elif regressions == 'complete':
        do_available_regressions = False
    else:
        raise ValueError(regressions + 'could not be understood')
    # Assign a reference or copy to res, depending on inplace:
    if inplace:
        res = data
    else:
        res = data.copy()
    # If dependent is not set, apply the operation to each column that contains
    # missing data:
    if dependent is None:
        for column in data.columns:
            if data[column].isna().any():
                res.loc[:, :] = linear_regression_one_dependent(
                    res, column, predictors, do_available_regressions,
                    noise)
    # Otherwise apply the operation to the dependent column only:
    else:
        res.loc[:, :] = linear_regression_one_dependent(
            data, dependent, predictors, do_available_regressions, noise)
    # Return dataframe if the operation is not to be performed inplace:
    if not inplace:
        return res


def linear_regression_one_dependent(
        data, dependent, predictors, do_available_regressions, noise):
    """Auxiliary function that performs linear regression imputation for the
    dependent column. The difference with linear_regression() is that in
    that function dependent can be None, in which case this function is
    called for each column containing missing values,

    :param data: The data on which to perform the linear regression imputation.
    :type data: pandas.DataFrame
    :param dependent: The dependent variable in which the missing values
        should be imputed.
    :type dependent: String
    :param predictors: The predictor variables on which the dependent variable
        is dependent.
    :type predictors: array-like
    :param do_available_regressions: Whether to do regressions for all
        available predictor combinations or only on complete ones
    :type do_available_regressions: bool
    :param noise: Whether to add noise to the imputed values (stochastic
        regression imputation)
    :type noise: bool
    :return: The dataframe with linear regression imputation performed for the
        incomplete variable.
    :rtype: pandas.DataFrame
    """
    # This auxiliary function always returns a copy:
    res = data.copy()
    # If predictors is None, all variables except for the dependent one are
    # considered predictors:
    if predictors is None:
        predictors = list(data.columns)
        predictors.remove(dependent)
    # Predictor combination sets and lists
    limited_predictors_combs = set()
    predictors_combs_done = []
    predictors_combs_todo = [tuple(predictors)]
    # Perform the operation:
    while len(predictors_combs_todo) > 0:
        # Select iteration predictors
        it_predictors = predictors_combs_todo.pop(0)
        # Log iteration beginning:
        logging.info('Applying regression imputation with predictors: ' + str(
            it_predictors))
        # Perform iteration:
        res.loc[:, :] = linear_regression_iter(
            res, dependent, list(it_predictors), noise,
            limited_predictors_combs)
        # Update predictor combinations done and to do
        predictors_combs_done.append(it_predictors)
        if do_available_regressions:
            predictors_combs_todo = list(
                set(limited_predictors_combs) - set(predictors_combs_done))
        # Log iteration end:
        logging.info('Predictor combinations done: ' + str(
            predictors_combs_done))
        logging.info('Predictor combinations to do: ' + str(
            predictors_combs_todo))
    return res


def linear_regression_iter(
        data, dependent, predictors, noise, limited_predictors_combs):
    """Auxiliary function that performs (simple or multiple) linear
    regression imputation on the data, for the dependent column only. In rows
    that contain a missing value for any predictor variable, the value of the
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
    :param noise: Whether to add noise to the imputed value (stochastic
        regression imputation)
    :type noise: bool
    :param limited_predictors_combs: Reference to the set which contains all
        limited predictor combinations that are necessary to use because
        some predictor had a missing value in some row.
    :type limited_predictors_combs: set
    :return: A copy of the dataframe with linear regression imputation
        performed for the incomplete variable.
    :rtype: pandas.DataFrame
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
    # Calculate standard error:
    std_error = (model.predict(x) - y).std()
    logging.info('Standard error: ' + str(std_error))
    # Implementation using apply:
    return data.apply(
        lambda row: get_imputed_row(
            row, dependent, predictors, intercept, coefs, noise, std_error,
            limited_predictors_combs),
        axis=1, result_type='broadcast')


def get_imputed_row(
        row, dependent, predictors, intercept, coefs, noise, std_error,
        limited_predictors_combs):
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
    :type coefs: array-like,
    :param noise: Whether to add noise to the imputed value (stochastic
        regression imputation)
    :type noise: bool
    :param std_error: The standard error of the regression model. Required
        if noise=True
    :type std_error: scalar
    :param limited_predictors_combs: Reference to the set which contains all
        limited predictor combinations that are necessary to use because
        some predictor had a missing value in some row.
    :type limited_predictors_combs: set
    :return: The row, with the missing value imputed if it contains one.
    :rtype: pandas.Series
    """
    res = row.copy()
    if pd.isnull(res[dependent]):
        # Check whether there are predictors for which the value is NA
        na_predictors = tuple(
            row[predictors][row[predictors].isnull()].index.to_list())
        # If the row contains NA values for one or several predictors,
        # add the combination of predictors to na_predictor_combs, in order
        # to perform regression without them:
        if na_predictors != ():
            limited_predictors = tuple(set(predictors) - set(na_predictors))
            # Add the limited_predictors to the set only if the combination
            # isn't empty:
            if limited_predictors != ():
                limited_predictors_combs.add(limited_predictors)
        # If the row doesn't contain missing values for any predictor, impute:
        else:
            value = intercept
            for idx, coef in enumerate(coefs):
                value += coef * row[predictors[idx]]
            # If noise == True, add noise (stochastic regression imputation)
            if noise:
                value += std_error * np.random.randn()
            res[dependent] = value
    return res
