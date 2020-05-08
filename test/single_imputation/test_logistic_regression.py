import unittest

from imputena import logistic_regression

from test.example_data import *


class TestLogisticRegression(unittest.TestCase):

    # Positive tests ----------------------------------------------------------

    def test_logistic_regression_returning(self):
        """
        Positive test

        data: Correct data frame (df_breast_cancer)

        The data frame (df_breast_cancer) contains 15 NA values.
        logistic_regression() should impute 7 of them.

        Checks that the original series remains unmodified and that the
        returned series contains 8 NA values.
        """
        # 1. Arrange
        df = generate_df_breast_cancer()
        # 2. Act
        df2 = logistic_regression(df, 'class', ['thickness', 'uniformity'])
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 15)
        self.assertEqual(df2.isna().sum().sum(), 8)

    def test_logistic_regression_inplace(self):
        """
        Positive test

        data: Correct data frame (df_breast_cancer)

        The data frame (df_breast_cancer) contains 15 NA values.
        logistic_regression() should impute 7 of them.

        Checks that the data frame contains 8 NA values after the operation.
        """
        # 1. Arrange
        df = generate_df_breast_cancer()
        # 2. Act
        logistic_regression(
            df, 'class', ['thickness', 'uniformity'], inplace=True)
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 8)

    def test_logistic_regression_implicit_predictors(self):
        """
        Positive test

        data: Correct data frame (df_breast_cancer)
        predictors: None

        The data frame (df_breast_cancer) contains 15 NA values.
        logistic_regression() should impute 7 of them.

        Checks that the original series remains unmodified and that the
        returned series contains 8 NA values.
        """
        # 1. Arrange
        df = generate_df_breast_cancer()
        # 2. Act
        df2 = logistic_regression(df, 'class')
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 15)
        self.assertEqual(df2.isna().sum().sum(), 8)

    def test_logistic_regression_complete(self):
        """
        Positive test

        data: Correct data frame (df_breast_cancer)
        regressions: 'complete'

        The data frame (df_breast_cancer) contains 15 NA values.
        logistic_regression() should impute 3 of them.

        Checks that the original series remains unmodified and that the
        returned series contains 12 NA values.
        """
        # 1. Arrange
        df = generate_df_breast_cancer()
        # 2. Act
        df2 = logistic_regression(
            df, 'class', ['thickness', 'uniformity'], regressions='complete')
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 15)
        self.assertEqual(df2.isna().sum().sum(), 12)

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

        data: Correct data frame (df_breast_cancer)
        dependent: 'z' (not a column of df_breast_cancer)

        Checks that the function raises a ValueError if the column specified as
        the dependent variable doesn't exist in the data.
        """
        # 1. Arrange
        df = generate_df_breast_cancer()
        # 2. Act & 3. Assert
        with self.assertRaises(ValueError):
            logistic_regression(df, 'z', ['thickness', 'uniformity'])

    def test_logistic_regression_wrong_predictor(self):
        """
        Negative test

        data: Correct data frame (df_breast_cancer)
        predictors: ['thickness', 'z'] ('z' is not a column of
        df_breast_cancer)

        Checks that the function raises a ValueError if one of the column s
        specified as the predictor variables doesn't exist in the data.
        """
        # 1. Arrange
        df = generate_df_breast_cancer()
        # 2. Act & 3. Assert
        with self.assertRaises(ValueError):
            logistic_regression(df, 'class', ['thickness', 'z'])

    def test_logistic_regression_wrong_regressions(self):
        """
        Negative test

        data: Correct data frame (df_breast_cancer)
        regressions: 'z' (not a valid value)

        Checks that the function raises a ValueError if the value passed for
        the parameter regressions is not valid.
        """
        # 1. Arrange
        df = generate_df_breast_cancer()
        # 2. Act & 3. Assert
        with self.assertRaises(ValueError):
            logistic_regression(
                df, 'class', ['thickness', 'uniformity'], regressions='z')
