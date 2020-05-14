import pandas as pd
from sklearn import linear_model
from imputena.simple_imputation.linear_regression import get_imputed_row
import logging


def srmi(data=None, sample_size=10, imputations=3, regressions='available'):
    """Performs sequential regression multiple imputation on the data.
    Several (parameter imputations) imputations are performed and the
    resulting dataframes returned as a list. For each one, a regression
    model is created based on a sample of the available rows. The size of
    these samples is fixed by the parameter sample_size. If, in the same
    row as a missing value in the dependent variable the value for any
    predictor variable is missing, a regression model based on all available
    predictors in calculated just to impute those values where the
    predictor(s) are missing. This behavior can be changed by assigning to
    the parameter regressions the value 'complete'. In this case, rows in
    which a predictor variable is missing do not get imputed.

    :param data: The data on which to perform the SRMI.
    :type data: pandas.DataFrame
    :param sample_size: Maximum size of the set of rows used to compute the
        regression model. Has to be at least 2.
    :type sample_size: scalar, default 10
    :param imputations: Number of imputations to perform
    :type imputations: scalar, default 3
    :param regressions: If 'available': Impute missing values by modeling a
        regression based on all available predictors if some predictors have
        missing values themselves. If 'complete': Only impute with a
        regression model based on all predictors and leave missing values in
        rows in which some predictor value is missing itself unimputed.
    :type regressions: {'available', 'complete'}, default 'available'
    :return: A list of linear regression imputations performed based on
        regression models calculated from different samples.
    :rtype: list of pandas.DataFrame
    :raises: TypeError, ValueError
    """
    # Check if data is a dataframe:
    if not isinstance(data, pd.DataFrame):
        raise TypeError('The data has to be a DataFrame.')
    # Assign value to do_available_regressions
    if regressions == 'available':
        do_available_regressions = True
    elif regressions == 'complete':
        do_available_regressions = False
    else:
        raise ValueError(regressions + 'could not be understood')
    # Create the list that will be returned
    imputed_datasets = []
    # Impute several times and add the results to the list:
    for _ in range(imputations):
        imputed_datasets.append(
            srmi_one_imputation(data, sample_size, do_available_regressions))
    # Return the imputed datasets:
    return imputed_datasets


def srmi_one_imputation(data, sample_size, do_available_regressions):
    """Auxiliary function that performs one linear regression imputation,
    creating the regression model based on a sample.

    :param data: The data on which to perform the linear regression imputation.
    :type data: pandas.DataFrame
    :param sample_size: Maximum size of the set of rows used to compute the
        regression model.
    :type sample_size: scalar
    :param do_available_regressions: Whether to do regressions for all
        available predictor combinations or only on complete ones
    :type do_available_regressions: bool
    :return: The dataframe with one linear regression imputation performed
        for all columns with missing values, based on a model created from a
        sample.
    :rtype: pandas.DataFrame
    """
    # This auxiliary function always returns a copy:
    res = data.copy()
    # Impute each column that contains missing values:
    for column in data.columns:
        if data[column].isna().any():
            res.loc[:, :] = srmi_one_dependent(
                res, column, None, do_available_regressions, sample_size)
    # Return the result:
    return res


def srmi_one_dependent(
        data, dependent, predictors, do_available_regressions, sample_size):
    """Auxiliary function that performs linear regression imputation for the
    dependent column. The difference with srmi_step() is that in
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
    :param sample_size: Maximum size of the set of rows used to compute the
        regression model.
    :type sample_size: scalar
    :return: The dataframe with linear regression imputation performed for the
        incomplete variable.
    :rtype: pandas.DataFrame
    """
    # THis auxiliary function always returns a copy:
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
        res.loc[:, :] = srmi_iter(
            res, dependent, list(it_predictors), sample_size,
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
    # Return the result:
    return res


def srmi_iter(
        data, dependent, predictors, sample_size, limited_predictors_combs):
    """Auxiliary function that performs (simple or multiple) linear
    regression imputation on the data, for the dependent column only. The
    regression model is based on a subset of all available rows that has the
    maximum size sample_size. In rows that contain a missing value for any
    predictor variable, the value of the dependent variable does not get
    imputed. The operation is always performed on a copy of the data,
    which is returned.

    :param data: The data on which to perform the linear regression imputation.
    :type data: pandas.DataFrame
    :param dependent: The dependent variable in which the missing values
        should be imputed.
    :type dependent: String
    :param predictors: The predictor variables on which the dependent variable
        is dependent.
    :type predictors: array-like
    :param sample_size: Maximum size of the set of rows used to compute the
        regression model.
    :type sample_size: scalar
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
    # Select sample_size random values from data_pairwise_deleted:
    data_sampled = data_pairwise_deleted
    if len(data_pairwise_deleted) > sample_size:
        data_sampled = data_pairwise_deleted.sample(sample_size)
    # Calculate the regression:
    x = data_sampled[predictors]
    y = data_sampled[dependent]
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
            row, dependent, predictors, intercept, coefs, False, std_error,
            limited_predictors_combs),
        axis=1, result_type='broadcast')
