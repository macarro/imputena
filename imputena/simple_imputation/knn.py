import pandas as pd
from sklearn.impute import KNNImputer


def knn(data=None, columns=None, k=3, inplace=False):
    """Performs k-nearest neighbors imputation on the data. The k nearest
    neighbors or each subject with missing data are chosen and the average
    of their values is used to impute the missing value. The operation can be
    applied to all columns, by leaving the parameter columns empty, or to
    selected columns, passed as an array of strings.

    :param data: The data on which to perform the k-nearest neighbors
        imputation.
    :type data: pandas.DataFrame
    :param columns: Columns on which to apply the operation.
    :type columns: array-like, optional
    :param k: The number of neighbors to which the subject with missing
        values should be compared
    :type k: int, default 3
    :param inplace: If True, do operation inplace and return None.
    :type inplace: bool, default False
    :return: The series or dataframe with NA values imputed, or
        None if inplace=True.
    :rtype: pandas.DataFrame or None
    :raises: TypeError, ValueError
    """
    # Check if data is a dataframe:
    if not isinstance(data, pd.DataFrame):
        raise TypeError('The data has to be a DataFrame.')
    # Assign a reference or copy to res, depending on inplace:
    if inplace:
        res = data
    else:
        res = data.copy()
    # The KNNImputer removes all columns that contain only empty values.
    # Therefore, we save those values in order to add them later (otherwise
    # problems would occur with dataframes that contain such columns:
    empty_column_names = res.columns[res.isna().all()]
    empty_column_indices = [
        res.columns.get_loc(column_name) for column_name in empty_column_names]
    empty_column_values = res.loc[:, res.isna().all()]
    # Perform KNN:
    knn_out_array = KNNImputer(n_neighbors=k).fit_transform(data)
    knn_out = pd.DataFrame(knn_out_array)
    # Add empty columns back and set indices of knn_out:
    for i, empty_column_name in enumerate(empty_column_names):
        knn_out.insert(
            empty_column_indices[i], empty_column_name,
            empty_column_values.iloc[:, i])
    knn_out.columns = res.columns
    knn_out.index = res.index
    # Treatment for a whole dataframe:
    if columns is None:
        res.loc[:, :] = knn_out
    # Treatment for selected columns of a dataframe:
    else:
        for column in columns:
            if column not in data.columns:
                raise ValueError(
                    '\'' + column + '\' is not a column of the data.')
            col_loc = data.columns.get_loc(column)
            res[column] = knn_out.iloc[:, col_loc]
    # Return the imputed data, or None if inplace:
    if inplace:
        return None
    else:
        return res
