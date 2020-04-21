import unittest

from imputena import logistic_regression

from .example_data import *


class TestLogisticRegression(unittest.TestCase):

    # Positive tests ----------------------------------------------------------

    def test_logistic_regression_returning(self):
        """
        Positive test

        data: Correct data frame (example_df_categorical)

        The data frame (example_df_categorical) contains 3 NA values.
        logistic_regression() should impute one of them.

        Checks that the original series remains unmodified and that the
        returned series contains 2 NA values.
        """
        # 1. Arrange
        df = generate_example_df_categorical()
        # 2. Act
        df2 = logistic_regression(df, 'healthy', ['age'])
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 3)
        self.assertEqual(df2.isna().sum().sum(), 2)

    def test_logistic_regression_inplace(self):
        """
        Positive test

        data: Correct data frame (example_df_categorical)

        The data frame (example_df_categorical) contains 3 NA values.
        logistic_regression() should impute one of them.

        Checks that the data frame contains 2 NA values after the operation.
        """
        # 1. Arrange
        df = generate_example_df_categorical()
        # 2. Act
        logistic_regression(df, 'healthy', ['age'], inplace=True)
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 2)

    # Negative tests ----------------------------------------------------------

    def test_logistic_regression_wrong_type(self):
        """
        Negative test

        data: array (unsupported type)

        Checks that the function raises a TypeError if the data is passed as
        an array.
        """
        # 1. Arrange
        data = [2, 4, np.nan, 1]
        # 2. Act & 3. Assert
        with self.assertRaises(TypeError):
            logistic_regression(data)

    def test_logistic_regression_wrong_dependent(self):
        """
        Negative test

        data: Correct data frame (example_df_categorical)
        dependent: 'z' (not a column of example_df_categorical)

        Checks that the function raises a ValueError if the column specified as
        the dependent variable doesn't exist in the data.
        """
        # 1. Arrange
        df = generate_example_df_categorical()
        # 2. Act & 3. Assert
        with self.assertRaises(ValueError):
            logistic_regression(df, 'z', ['age'])

    def test_logistic_regression_wrong_predictor(self):
        """
        Negative test

        data: Correct data frame (example_df_categorical)
        predictors: ['age', 'z'] ('z' is not a column of
        example_df_categorical)

        Checks that the function raises a ValueError if one of the column s
        specified as the predictor variables doesn't exist in the data.
        """
        # 1. Arrange
        df = generate_example_df_categorical()
        # 2. Act & 3. Assert
        with self.assertRaises(ValueError):
            logistic_regression(df, 'healthy', ['age', 'z'])
