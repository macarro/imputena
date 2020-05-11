import unittest
import logging

from imputena import linear_regression

from test.example_data import *


class TestLinearRegression(unittest.TestCase):

    def setUp(self):
        # Uncomment the following line to show debug and info logs
        # logging.getLogger().setLevel(logging.DEBUG)
        logging.basicConfig(format='%(message)s')
        logging.info("\n" + self._testMethodName)

    # Positive tests ----------------------------------------------------------

    def test_LR_returning(self):
        """
        Positive test

        data: Correct data frame (sales)

        The data frame sales contains 4 NA values in the column 'sales'.
        linear_regression() should impute 3 of them.

        Checks that the original data frame remains unmodified and that the
        returned series contains 1 NA value in the column 'sales'.
        """
        # 1. Arrange
        df = generate_df_sales()
        # 2. Act
        df2 = linear_regression(df, 'sales', ['advertising', 'year'])
        # 3. Assert
        self.assertEqual(df['sales'].isna().sum(), 4)
        self.assertEqual(df2['sales'].isna().sum(), 1)

    def test_LR_inplace(self):
        """
        Positive test

        data: Correct data frame (sales)

        The data frame sales contains 4 NA values in the column 'sales'.
        linear_regression() should impute 3 of them.

        Checks that the data frame contains 1 NA value in the column 'sales'
        after the operation.
        """
        # 1. Arrange
        df = generate_df_sales()
        # 2. Act
        linear_regression(df, 'sales', ['advertising', 'year'], inplace=True)
        # 3. Assert
        self.assertEqual(df['sales'].isna().sum(), 1)

    def test_LR_implicit_predictors(self):
        """
        Positive test

        data: Correct data frame (sales)
        predictors: None

        The data frame sales contains 4 NA values in the column 'sales'.
        linear_regression() should impute 3 of them.

        Checks that the original data frame remains unmodified and that the
        returned series contains 1 NA value in the column 'sales'.
        """
        # 1. Arrange
        df = generate_df_sales()
        # 2. Act
        df2 = linear_regression(df, 'sales')
        # 3. Assert
        self.assertEqual(df['sales'].isna().sum(), 4)
        self.assertEqual(df2['sales'].isna().sum(), 1)

    def test_LR_complete(self):
        """
        Positive test

        data: Correct data frame (sales)
        regressions: 'complete'

        The data frame sales contains 4 NA values in the column 'sales'.
        linear_regression() should impute 1 of them.

        Checks that the original data frame remains unmodified and that the
        returned series contains 3 NA value in the column 'sales'.
        """
        # 1. Arrange
        df = generate_df_sales()
        # 2. Act
        df2 = linear_regression(
            df, 'sales', ['advertising', 'year'], 'complete')
        # 3. Assert
        self.assertEqual(df['sales'].isna().sum(), 4)
        self.assertEqual(df2['sales'].isna().sum(), 3)

    def test_LR_noise(self):
        """
        Positive test

        data: Correct data frame (sales)
        noise: True

        The data frame sales contains 4 NA values in the column 'sales'.
        linear_regression() should impute 3 of them.

        Checks that the original data frame remains unmodified and that the
        returned series contains 1 NA value in the column 'sales'.
        """
        # 1. Arrange
        df = generate_df_sales()
        # 2. Act
        df2 = linear_regression(
            df, 'sales', ['advertising', 'year'], noise=True)
        # 3. Assert
        self.assertEqual(df['sales'].isna().sum(), 4)
        self.assertEqual(df2['sales'].isna().sum(), 1)

    def test_LR_all_columns(self):
        """
        Positive test

        data: Correct data frame (sales)
        dependent: None

        The data frame sales contains 8 NA values.
        linear_regression() should impute 5 of them.

        Checks that the original data frame remains unmodified and that the
        returned series contains 3 NA values.
        """
        # 1. Arrange
        df = generate_df_sales()
        # 2. Act
        df2 = linear_regression(df)
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 8)
        self.assertEqual(df2.isna().sum().sum(), 3)

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
            linear_regression(data)

    def test_LR_wrong_dependent(self):
        """
        Negative test

        data: Correct data frame (sales)
        dependent: 'z' (not a column of sales)

        Checks that the function raises a ValueError if the column specified as
        the dependent variable doesn't exist in the data.
        """
        # 1. Arrange
        df = generate_df_sales()
        # 2. Act & 3. Assert
        with self.assertRaises(ValueError):
            linear_regression(df, 'z', ['advertising', 'year'])

    def test_LR_wrong_predictor(self):
        """
        Negative test

        data: Correct data frame (sales)
        predictors: ['advertising', 'z'] ('z' is not a column of sales)

        Checks that the function raises a ValueError if one of the column
        specified as the predictor variables doesn't exist in the data.
        """
        # 1. Arrange
        df = generate_df_sales()
        # 2. Act & 3. Assert
        with self.assertRaises(ValueError):
            linear_regression(df, 'sales', ['advertising', 'z'])

    def test_LR_wrong_regressions(self):
        """
        Negative test

        data: Correct data frame (sales)
        regressions: 'z' (not a valid value)

        Checks that the function raises a ValueError if the value passed for
        the parameter regressions is not valid.
        """
        # 1. Arrange
        df = generate_df_sales()
        # 2. Act & 3. Assert
        with self.assertRaises(ValueError):
            linear_regression(df, 'sales', ['advertising', 'year'], 'z')
