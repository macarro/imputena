import unittest

from imputena import locf

from test.example_data import *


class TestLOCF(unittest.TestCase):

    # Positive tests for data as a dataframe ----------------------------------

    def test_LOCF_df_returning(self):
        """
        Positive test

        data: Correct dataframe (divcols)

        Checks that the original dataframe remains unmodified and that the
        returned dataframe contains 13 NA values, 5 less than the original.
        """
        # 1. Arrange
        df = generate_example_df_divcols()
        # 2. Act
        df2 = locf(df)
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 18)
        self.assertEqual(df2.isna().sum().sum(), 13)

    def test_LOCF_df_inplace(self):
        """
        Positive test

        data: Correct dataframe (divcols)

        Checks that locf removes 5 NA values from the dataframe.
        """
        # 1. Arrange
        df = generate_example_df_divcols()
        # 2. Act
        locf(df, inplace=True)
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 13)

    def test_LOCF_df_returning_fill_leading(self):
        """
        Positive test

        data: Correct dataframe (divcols)
        fill_leading: True

        Checks that the original dataframe remains unmodified and that the
        returned dataframe contains 10 NA values, 8 less than the original.
        """
        # 1. Arrange
        df = generate_example_df_divcols()
        # 2. Act
        df2 = locf(df, fill_leading=True)
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 18)
        self.assertEqual(df2.isna().sum().sum(), 10)

    def test_LOCF_df_inplace_fill_leading(self):
        """
        Positive test

        data: Correct dataframe (divcols)
        fill_leading: True

        Checks that locf removes 8 NA values from the dataframe.
        """
        # 1. Arrange
        df = generate_example_df_divcols()
        # 2. Act
        locf(df, fill_leading=True, inplace=True)
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 10)

    def test_LOCF_df_returning_columns(self):
        """
        Positive test

        data: Correct dataframe (divcols)
        columns: ['f', 'g']

        Checks that the original dataframe remains unmodified and that the
        returned dataframe contains 16 NA values, 2 less than the original.
        """
        # 1. Arrange
        df = generate_example_df_divcols()
        # 2. Act
        df2 = locf(df, columns=['f', 'g'])
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 18)
        self.assertEqual(df2.isna().sum().sum(), 16)

    def test_LOCF_df_inplace_columns(self):
        """
        Positive test

        data: Correct dataframe (divcols)
        columns: ['f', 'g']

        Checks that locf removes 2 NA values from the specified columns.
        """
        # 1. Arrange
        df = generate_example_df_divcols()
        # 2. Act
        locf(df, columns=['f', 'g'], inplace=True)
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 16)

    def test_LOCF_df_returning_columns_fill_leading(self):
        """
        Positive test

        data: Correct dataframe (divcols)
        columns: ['f', 'g']
        fill_leading: True

        Checks that the original dataframe remains unmodified and that the
        returned dataframe contains 14 NA values, 4 less than the original.
        """
        # 1. Arrange
        df = generate_example_df_divcols()
        # 2. Act
        df2 = locf(df, columns=['f', 'g'], fill_leading=True)
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 18)
        self.assertEqual(df2.isna().sum().sum(), 14)

    def test_LOCF_df_inplace_columns_fill_leading(self):
        """
        Positive test

        data: Correct dataframe (divcols)
        columns: ['f', 'g']
        fill_leading: True

        Checks that locf removes 4 NA values from the specified columns.
        """
        # 1. Arrange
        df = generate_example_df_divcols()
        # 2. Act
        locf(df, columns=['f', 'g'], fill_leading=True, inplace=True)
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 14)

    # Positive tests for data as a series -------------------------------------

    def test_LOCF_series_returning(self):
        """
        Positive test

        data: Correct Series (example series)

        Checks that the original series remains unmodified and that the
        returned series contains 1 NA value, 2 less than the original.
        """
        # 1. Arrange
        ser = generate_example_series()
        # 2. Act
        ser2 = locf(ser)
        # 3. Assert
        self.assertEqual(ser.isna().sum(), 3)
        self.assertEqual(ser2.isna().sum(), 1)

    def test_LOCF_series_inplace(self):
        """
        Positive test

        data: Correct Series (example series)

        Checks that locf removes 2 NA values from the series.
        """
        # 1. Arrange
        ser = generate_example_series()
        # 2. Act
        locf(ser, inplace=True)
        # 3. Assert
        self.assertEqual(ser.isna().sum(), 1)

    def test_LOCF_series_returning_fill_leading(self):
        """
        Positive test

        data: Correct Series (example series)
        fill_leading: True

        Checks that the original series remains unmodified and that the
        returned series contains 0 NA values.
        """
        # 1. Arrange
        ser = generate_example_series()
        # 2. Act
        ser2 = locf(ser, fill_leading=True)
        # 3. Assert
        self.assertEqual(ser.isna().sum(), 3)
        self.assertEqual(ser2.isna().sum(), 0)

    def test_LOCF_series_inplace_fill_leading(self):
        """
        Positive test

        data: Correct Series (example series)
        fill_leading: True

        Checks that locf removes all NA values from the series.
        """
        # 1. Arrange
        ser = generate_example_series()
        # 2. Act
        locf(ser, fill_leading=True, inplace=True)
        # 3. Assert
        self.assertEqual(ser.isna().sum(), 0)

    # Negative tests ----------------------------------------------------------

    def test_LOCF_wrong_type(self):
        """
        Negative test

        data: array (unsupported type)

        Checks that the locf raises a TypeError if the data is passed as an
        array.
        """
        # 1. Arrange
        data = [2, 4, np.nan, 1]
        # 2. Act & 3. Assert
        with self.assertRaises(TypeError):
            locf(data)

    def test_LOCF_df_returning_wrong_column(self):
        """
        Negative test

        data: Correct dataframe (divcols)
        columns: ['f', 'g', 'z'] ('z' doesn't exist in the data)

        Checks that the locf raises a ValueError if the data is passed as an
        array.
        """
        # 1. Arrange
        df = generate_example_df_divcols()
        # 2. Act & 3. Assert
        with self.assertRaises(ValueError):
            locf(df, columns=['f', 'g', 'z'])

    def test_LOCF_df_inplace_wrong_column(self):
        """
        Negative test

        data: Correct dataframe (divcols)
        columns: ['f', 'g', 'z'] ('z' doesn't exist in the data)

        Checks that the locf raises a ValueError if the data is passed as an
        array.
        """
        # 1. Arrange
        df = generate_example_df_divcols()
        # 2. Act & 3. Assert
        with self.assertRaises(ValueError):
            locf(df, columns=['f', 'g', 'z'], inplace=True)
