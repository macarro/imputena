import unittest

from imputena import linear_regression

from .example_data import *


class TestLinearRegression(unittest.TestCase):

    # Positive tests ----------------------------------------------------------

    def test_LR_returning(self):
        """
        Positive test

        data: Correct data frame (example_df_reg)

        The data frame (example_df_reg) contains 3 NA values.
        linear_regression() should impute one of them.

        Checks that the original series remains unmodified and that the
        returned series contains 2 NA values.
        """
        # 1. Arrange
        df = generate_example_df_reg()
        # 2. Act
        df2 = linear_regression(df, 'dep', ['pred1', 'pred2'])
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 3)
        self.assertEqual(df2.isna().sum().sum(), 2)

    def test_LR_inplace(self):
        """
        Positive test

        data: Correct data frame (example_df_reg)

        The data frame (example_df_reg) contains 3 NA values.
        linear_regression() should impute one of them.

        Checks that the data frame contains 2 NA values after the operation.
        """
        # 1. Arrange
        df = generate_example_df_reg()
        # 2. Act
        linear_regression(df, 'dep', ['pred1', 'pred2'], inplace=True)
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 2)

    # Negative tests ----------------------------------------------------------

    def test_LR_wrong_type(self):
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
            df = linear_regression(data)

    def test_LR_df_wrong_dependent(self):
        """
        Negative test

        data: Correct data frame (example_df_reg)
        dependent: 'z' (not a column of example_df_reg)

        Checks that the function raises a ValueError if the column specified as
        the dependent variable doesn't exist in the data.
        """
        # 1. Arrange
        df = generate_example_df_reg()
        # 2. Act & 3. Assert
        with self.assertRaises(ValueError):
            df2 = linear_regression(df, 'z', ['pred1', 'pred2'])

    def test_LR_df_wrong_predictor(self):
        """
        Negative test

        data: Correct data frame (example_df_reg)
        predictors: ['pred1', 'z'] ('z' is not a column of example_df_reg)

        Checks that the function raises a ValueError if one of the column s
        specified as the predictor variables doesn't exist in the data.
        """
        # 1. Arrange
        df = generate_example_df_reg()
        # 2. Act & 3. Assert
        with self.assertRaises(ValueError):
            df2 = linear_regression(df, 'dep', ['pred1', 'z'])
