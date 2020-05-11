import unittest

from imputena import interpolation

from test.example_data import *


class TestInterpolation(unittest.TestCase):

    # Positive tests for data as a dataframe ----------------------------------

    def test_interpolation_df_returning(self):
        """
        Positive test

        data: Correct dataframe (divcols)

        Checks that the original dataframe remains unmodified and that the
        returned dataframe contains 10 NA values, 8 less than the original.
        """
        # 1. Arrange
        df = generate_example_df_divcols()
        # 2. Act
        df2 = interpolation(df)
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 18)
        self.assertEqual(df2.isna().sum().sum(), 10)

    def test_interpolation_df_inplace(self):
        """
        Positive test

        data: Correct dataframe (divcols)

        Checks that interpolation removes 8 NA values from the dataframe.
        """
        # 1. Arrange
        df = generate_example_df_divcols()
        # 2. Act
        interpolation(df, inplace=True)
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 10)

    def test_interpolation_df_returning_columns(self):
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
        df2 = interpolation(df, columns=['f', 'g'])
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 18)
        self.assertEqual(df2.isna().sum().sum(), 14)

    def test_interpolation_df_inplace_columns(self):
        """
        Positive test

        data: Correct dataframe (divcols)
        columns: ['f', 'g']

        Checks that interpolation removes 4 NA values from the dataframe.
        """
        # 1. Arrange
        df = generate_example_df_divcols()
        # 2. Act
        interpolation(df, columns=['f', 'g'], inplace=True)
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 14)

    # Positive tests for data as a series -------------------------------------

    def test_interpolation_series_returning(self):
        """
        Positive test

        data: Correct Series (example series)

        Checks that the original series remains unmodified and that the
        returned series contains 0 NA values, 3 less than the original.
        """
        # 1. Arrange
        ser = generate_example_series()
        # 2. Act
        ser2 = interpolation(ser)
        # 3. Assert
        self.assertEqual(ser.isna().sum(), 3)
        self.assertEqual(ser2.isna().sum(), 0)

    def test_interpolation_series_inplace(self):
        """
        Positive test

        data: Correct Series (example series)

        Checks that interpolation removes 3 NA values from the series.
        """
        # 1. Arrange
        ser = generate_example_series()
        # 2. Act
        interpolation(ser, inplace=True)
        # 3. Assert
        self.assertEqual(ser.isna().sum(), 0)

    # Negative tests ----------------------------------------------------------

    def test_interpolation_wrong_type(self):
        """
        Negative test

        data: array (unsupported type)

        Checks that the interpolation raises a TypeError if the data is
        passed as an array.
        """
        # 1. Arrange
        data = [2, 4, np.nan, 1]
        # 2. Act & 3. Assert
        with self.assertRaises(TypeError):
            interpolation(data)

    def test_interpolation_df_returning_wrong_column(self):
        """
        Negative test

        data: Correct dataframe (divcols)
        columns: ['f', 'g', 'z'] ('z' doesn't exist in the data)

        Checks that the interpolation raises a ValueError if one of the
        specified columns doesn't exist in the data.
        """
        # 1. Arrange
        df = generate_example_df_divcols()
        # 2. Act & 3. Assert
        with self.assertRaises(ValueError):
            interpolation(df, columns=['f', 'g', 'z'])

    def test_interpolation_df_inplace_wrong_column(self):
        """
        Negative test

        data: Correct dataframe (divcols)
        columns: ['f', 'g', 'z'] ('z' doesn't exist in the data)

        Checks that the interpolation raises a ValueError if one of the
        specified columns doesn't exist in the data.
        """
        # 1. Arrange
        df = generate_example_df_divcols()
        # 2. Act & 3. Assert
        with self.assertRaises(ValueError):
            interpolation(df, columns=['f', 'g', 'z'], inplace=True)
