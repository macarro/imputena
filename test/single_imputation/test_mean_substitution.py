import unittest

from imputena import mean_substitution

from test.example_data import *


class TestMeanSubstitution(unittest.TestCase):

    # Positive tests for data as a dataframe ----------------------------------

    def test_MS_df_mean_returning(self):
        """
        Positive test

        data: Correct dataframe (divcols)

        Checks that the original dataframe remains unmodified and that the
        returned dataframe contains 10 NA values, 8 less than the original.
        """
        # 1. Arrange
        df = generate_example_df_divcols()
        # 2. Act
        df2 = mean_substitution(df)
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 18)
        self.assertEqual(df2.isna().sum().sum(), 10)

    def test_MS_df_median_returning(self):
        """
        Positive test

        data: Correct dataframe (divcols)
        method: 'median'

        Checks that the original dataframe remains unmodified and that the
        returned dataframe contains 10 NA values, 8 less than the original.
        """
        # 1. Arrange
        df = generate_example_df_divcols()
        # 2. Act
        df2 = mean_substitution(df, method='median')
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 18)
        self.assertEqual(df2.isna().sum().sum(), 10)

    def test_MS_df_mean_inplace(self):
        """
        Positive test

        data: Correct dataframe (divcols)

        Checks that mean_substitution removes 8 NA values from the dataframe.
        """
        # 1. Arrange
        df = generate_example_df_divcols()
        # 2. Act
        mean_substitution(df, inplace=True)
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 10)

    def test_MS_df_median_inplace(self):
        """
        Positive test

        data: Correct dataframe (divcols)
        method: 'median'

        Checks that mean_substitution removes 8 NA values from the dataframe.
        """
        # 1. Arrange
        df = generate_example_df_divcols()
        # 2. Act
        mean_substitution(df, method='median', inplace=True)
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 10)

    def test_MS_df_mean_returning_columns(self):
        """
        Positive test

        data: Correct dataframe (divcols)
        columns: ['f', 'g']

        Checks that the original dataframe remains unmodified and that the
        returned dataframe contains 14 NA values, 4 less than the original.
        """
        # 1. Arrange
        df = generate_example_df_divcols()
        # 2. Act
        df2 = mean_substitution(df, columns=['f', 'g'])
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 18)
        self.assertEqual(df2.isna().sum().sum(), 14)

    def test_MS_df_median_returning_columns(self):
        """
        Positive test

        data: Correct dataframe (divcols)
        columns: ['f', 'g']
        method: 'median'

        Checks that the original dataframe remains unmodified and that the
        returned dataframe contains 14 NA values, 4 less than the original.
        """
        # 1. Arrange
        df = generate_example_df_divcols()
        # 2. Act
        df2 = mean_substitution(df, method='median', columns=['f', 'g'])
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 18)
        self.assertEqual(df2.isna().sum().sum(), 14)

    def test_MS_df_mean_inplace_columns(self):
        """
        Positive test

        data: Correct dataframe (divcols)
        columns: ['f', 'g']

        Checks that mean_substitution removes 4 NA values from the specified
        columns.
        """
        # 1. Arrange
        df = generate_example_df_divcols()
        # 2. Act
        mean_substitution(df, columns=['f', 'g'], inplace=True)
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 14)

    def test_MS_df_median_inplace_columns(self):
        """
        Positive test

        data: Correct dataframe (divcols)
        columns: ['f', 'g']
        method: 'median'

        Checks that mean_substitution removes 4 NA values from the specified
        columns.
        """
        # 1. Arrange
        df = generate_example_df_divcols()
        # 2. Act
        mean_substitution(
            df, columns=['f', 'g'], method='median', inplace=True)
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 14)

    # Positive tests for data as a series -------------------------------------

    def test_MS_series_mean_returning(self):
        """
        Positive test

        data: Correct Series (example series)

        Checks that the original series remains unmodified and that the
        returned series contains no NA values, 3 less than the original.
        """
        # 1. Arrange
        ser = generate_example_series()
        # 2. Act
        ser2 = mean_substitution(ser)
        # 3. Assert
        self.assertEqual(ser.isna().sum(), 3)
        self.assertEqual(ser2.isna().sum(), 0)

    def test_MS_series_median_returning(self):
        """
        Positive test

        data: Correct Series (example series)
        method: 'median'

        Checks that the original series remains unmodified and that the
        returned series contains no NA values, 3 less than the original.
        """
        # 1. Arrange
        ser = generate_example_series()
        # 2. Act
        ser2 = mean_substitution(ser, method='median')
        # 3. Assert
        self.assertEqual(ser.isna().sum(), 3)
        self.assertEqual(ser2.isna().sum(), 0)

    def test_MS_series_mean_inplace(self):
        """
        Positive test

        data: Correct Series (example series)

        Checks that mean_substitution removes 3 NA values from the series.
        """
        # 1. Arrange
        ser = generate_example_series()
        # 2. Act
        mean_substitution(ser, inplace=True)
        # 3. Assert
        self.assertEqual(ser.isna().sum(), 0)

    def test_MS_series_median_inplace(self):
        """
        Positive test

        data: Correct Series (example series)
        method: 'median'

        Checks that mean_substitution removes 3 NA values from the series.
        """
        # 1. Arrange
        ser = generate_example_series()
        # 2. Act
        mean_substitution(ser, method='median', inplace=True)
        # 3. Assert
        self.assertEqual(ser.isna().sum(), 0)

    # Negative tests ----------------------------------------------------------

    def test_MS_wrong_type(self):
        """
        Negative test

        data: array (unsupported type)

        Checks that the mean_substitution raises a TypeError if the data is
        passed as an array.
        """
        # 1. Arrange
        data = [2, 4, np.nan, 1]
        # 2. Act & 3. Assert
        with self.assertRaises(TypeError):
            mean_substitution(data)

    def test_MS_df_mean_returning_wrong_column(self):
        """
        Negative test

        data: Correct dataframe (divcols)
        columns: ['f', 'g', 'z'] ('z' doesn't exist in the data)

        Checks that mean_substitution raises a ValueError if the data is
        passed as an array.
        """
        # 1. Arrange
        df = generate_example_df_divcols()
        # 2. Act & 3. Assert
        with self.assertRaises(ValueError):
            mean_substitution(df, columns=['f', 'g', 'z'])

    def test_MS_df_mean_inplace_wrong_column(self):
        """
        Negative test

        data: Correct dataframe (divcols)
        columns: ['f', 'g', 'z'] ('z' doesn't exist in the data)

        Checks that mean_substitution raises a ValueError if the data is
        passed as an array.
        """
        # 1. Arrange
        df = generate_example_df_divcols()
        # 2. Act & 3. Assert
        with self.assertRaises(ValueError):
            mean_substitution(df, columns=['f', 'g', 'z'], inplace=True)
