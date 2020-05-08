import unittest

from imputena import nocb

from test.example_data import *


class TestNOCB(unittest.TestCase):

    # Positive tests for data as a dataframe ----------------------------------

    def test_NOCB_df_returning(self):
        """
        Positive test

        data: Correct dataframe (divcols)

        Checks that the original dataframe remains unmodified and that the
        returned dataframe contains 11 NA values, 7 less than the original.
        """
        # 1. Arrange
        df = generate_example_df_divcols()
        # 2. Act
        df2 = nocb(df)
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 18)
        self.assertEqual(df2.isna().sum().sum(), 11)

    def test_NOCB_df_inplace(self):
        """
        Positive test

        data: Correct dataframe (divcols)

        Checks that nocb removes 7 NA values from the dataframe.
        """
        # 1. Arrange
        df = generate_example_df_divcols()
        # 2. Act
        nocb(df, inplace=True)
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 11)

    def test_NOCB_df_returning_fill_trailing(self):
        """
        Positive test

        data: Correct dataframe (divcols)
        fill_trailing: True

        Checks that the original dataframe remains unmodified and that the
        returned dataframe contains 10 NA values, 8 less than the original.
        """
        # 1. Arrange
        df = generate_example_df_divcols()
        # 2. Act
        df2 = nocb(df, fill_trailing=True)
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 18)
        self.assertEqual(df2.isna().sum().sum(), 10)

    def test_NOCB_df_inplace_fill_trailing(self):
        """
        Positive test

        data: Correct dataframe (divcols)
        fill_trailing: True

        Checks that nocb removes 8 NA values from the dataframe.
        """
        # 1. Arrange
        df = generate_example_df_divcols()
        # 2. Act
        nocb(df, fill_trailing=True, inplace=True)
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 10)

    def test_NOCB_df_returning_columns(self):
        """
        Positive test

        data: Correct dataframe (divcols)
        columns: ['f', 'g']

        Checks that the original dataframe remains unmodified and that the
        returned dataframe contains 15 NA values, 3 less than the original.
        """
        # 1. Arrange
        df = generate_example_df_divcols()
        # 2. Act
        df2 = nocb(df, columns=['f', 'g'])
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 18)
        self.assertEqual(df2.isna().sum().sum(), 15)

    def test_NOCB_df_inplace_columns(self):
        """
        Positive test

        data: Correct dataframe (divcols)
        columns: ['f', 'g']

        Checks that nocb removes 3 NA values from the specified columns.
        """
        # 1. Arrange
        df = generate_example_df_divcols()
        # 2. Act
        nocb(df, columns=['f', 'g'], inplace=True)
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 15)

    def test_NOCB_df_returning_columns_fill_trailing(self):
        """
        Positive test

        data: Correct dataframe (divcols)
        columns: ['f', 'g']
        fill_trailing: True

        Checks that the original dataframe remains unmodified and that the
        returned dataframe contains 14 NA values, 4 less than the original.
        """
        # 1. Arrange
        df = generate_example_df_divcols()
        # 2. Act
        df2 = nocb(df, columns=['f', 'g'], fill_trailing=True)
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 18)
        self.assertEqual(df2.isna().sum().sum(), 14)

    def test_NOCB_df_inplace_columns_fill_trailing(self):
        """
        Positive test

        data: Correct dataframe (divcols)
        columns: ['f', 'g']
        fill_trailing: True

        Checks that nocb removes 4 NA values from the specified columns.
        """
        # 1. Arrange
        df = generate_example_df_divcols()
        # 2. Act
        nocb(df, columns=['f', 'g'], fill_trailing=True, inplace=True)
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 14)

    # Positive tests for data as a series -------------------------------------

    def test_NOCB_series_returning(self):
        """
        Positive test

        data: Correct Series (example series)

        Checks that the original series remains unmodified and that the
        returned series contains 1 NA value, 2 less than the original.
        """
        # 1. Arrange
        ser = generate_example_series()
        # 2. Act
        ser2 = nocb(ser)
        # 3. Assert
        self.assertEqual(ser.isna().sum(), 3)
        self.assertEqual(ser2.isna().sum(), 1)

    def test_NOCB_series_inplace(self):
        """
        Positive test

        data: Correct Series (example series)

        Checks that nocb removes 2 NA values from the series.
        """
        # 1. Arrange
        ser = generate_example_series()
        # 2. Act
        nocb(ser, inplace=True)
        # 3. Assert
        self.assertEqual(ser.isna().sum(), 1)

    def test_NOCB_series_returning_fill_trailing(self):
        """
        Positive test

        data: Correct Series (example series)
        fill_trailing: True

        Checks that the original series remains unmodified and that the
        returned series contains 0 NA values.
        """
        # 1. Arrange
        ser = generate_example_series()
        # 2. Act
        ser2 = nocb(ser, fill_trailing=True)
        # 3. Assert
        self.assertEqual(ser.isna().sum(), 3)
        self.assertEqual(ser2.isna().sum(), 0)

    def test_NOCB_series_inplace_fill_trailing(self):
        """
        Positive test

        data: Correct Series (example series)
        fill_trailing: True

        Checks that nocb removes all NA values from the series.
        """
        # 1. Arrange
        ser = generate_example_series()
        # 2. Act
        nocb(ser, fill_trailing=True, inplace=True)
        # 3. Assert
        self.assertEqual(ser.isna().sum(), 0)

    # Negative tests ----------------------------------------------------------

    def test_NOCB_wrong_type(self):
        """
        Negative test

        data: array (unsupported type)

        Checks that the nocb raises a TypeError if the data is passed as an
        array.
        """
        # 1. Arrange
        data = [2, 4, np.nan, 1]
        # 2. Act & 3. Assert
        with self.assertRaises(TypeError):
            nocb(data)

    def test_NOCB_df_returning_wrong_column(self):
        """
        Negative test

        data: Correct dataframe (divcols)
        columns: ['f', 'g', 'z'] ('z' doesn't exist in the data)

        Checks that the nocb raises a ValueError if the data is passed as an
        array.
        """
        # 1. Arrange
        df = generate_example_df_divcols()
        # 2. Act & 3. Assert
        with self.assertRaises(ValueError):
            nocb(df, columns=['f', 'g', 'z'])

    def test_NOCB_df_inplace_wrong_column(self):
        """
        Negative test

        data: Correct dataframe (divcols)
        columns: ['f', 'g', 'z'] ('z' doesn't exist in the data)

        Checks that the nocb raises a ValueError if the data is passed as an
        array.
        """
        # 1. Arrange
        df = generate_example_df_divcols()
        # 2. Act & 3. Assert
        with self.assertRaises(ValueError):
            nocb(df, columns=['f', 'g', 'z'], inplace=True)
