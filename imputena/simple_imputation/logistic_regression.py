import pandas as pd
from sklearn import linear_model
from sklearn.exceptions import ConvergenceWarning
import logging
import warnings


def logistic_regression(
        data=None, dependent=None, predictors=None, regressions='available',
        inplace=False):
    """Performs logistic regression imputation on the data. First, the
    regression equation for the dependent variable given the predictor
    variables is computed. For this step, all rows that contain a missing
    value in either the dependent variable or any of the predictor variable
    is ignored via pairwise deletion. Then, missing valued in the dependent
    column in imputed using the regression equation.  If, in the same
    row as a missing value in the dependent variable the value for any
    predictor variable is missing, a regression model based on all available
    predictors in calculated just to impute those values where the
    predictor(s) are missing. This behavior can be changed by assigning to
    the parameter regressions the value 'complete'. In this case, rows in
    which a predictor variable is missing do not get imputed. If the parameter
    predictors is omitted, all variables other than the dependent are used as
    predictors.

    :param data: The data on which to perform the logistic regression
        imputation.
    :type data: pandas.DataFrame
    :param dependent: The dependent variable in which the missing values
        should be imputed.
    :type dependent: String
    :param predictors: The predictor variables on which the dependent variable
        is dependent.
    :type predictors: array-like, optional
    :param regressions: If 'available': Impute missing values by modeling a
        regression based on all available predictors if some predictors have
        missing values themselves. If 'complete': Only impute with a
        regression model based on all predictors and leave missing values in
        rows in which some predictor value is missing itself unimputed.
    :type regressions: {'available', 'complete'}, default 'available'
    :param inplace: If True, do operation inplace and return None.
    :type inplace: bool, default False
    :return: The dataframe with logistic regression imputation performed for
        the incomplete variable or None if inplace=True.
    :rtype: pandas.DataFrame or None
    :raises: TypeError, ValueError
    """
    # Check if data is a dataframe:
    if not isinstance(data, pd.DataFrame):
        raise TypeError('The data has to be a DataFrame.')
    # Check if the dependent variable is actually a column of the dataframe:
    if dependent not in data.columns:
        raise ValueError(
            '\'' + dependent + '\' is not a column of the data.')
    # If predictors is None, all variables except for the dependent one are
    # considered predictors:
    if predictors is None:
        predictors = list(data.columns)
        predictors.remove(dependent)
    # Check if each of the predictor variables is actually a column of the
    # dataframe:
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
        res.loc[:, :] = logistic_regression_iter(
            res, dependent, list(it_predictors),
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
    # Return dataframe is the operation is not to be performed inplace:
    if not inplace:
        return res


def logistic_regression_iter(
        data, dependent, predictors, limited_predictors_combs):
    """Auxiliary function that performs (simple or multiple) logistic
    regression imputation on the data, for the dependent column only. In rows
    that contain a missing value for any predictor variable, the value of the
    dependent variable does not get imputed. The operation is always
    performed on a copy of the data, which is returned.

    :param data: The data on which to perform the logistic regression
        imputation.
    :type data: pandas.DataFrame
    :param dependent: The dependent variable in which the missing values
        should be imputed.
    :type dependent: String
    :param predictors: The predictor variables on which the dependent variable
        is dependent.
    :type predictors: array-like
    :param limited_predictors_combs: Reference to the set which contains all
        limited predictor combinations that are necessary to use because
        some predictor had a missing value in some row.
    :type limited_predictors_combs: set
    :return: A copy of the dataframe with logistic regression imputation
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
    model = linear_model.LogisticRegression()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        model.fit(x, y)
    # Implementation using apply:
    return data.apply(
        lambda row: get_imputed_row(
            row, dependent, predictors, model, limited_predictors_combs),
        axis=1, result_type='broadcast')


def get_imputed_row(
        row, dependent, predictors, model, limited_predictors_combs):
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
            value = model.predict(row[predictors].values.reshape(1, -1))
            res[dependent] = value[0]
    return res
