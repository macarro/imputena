import unittest

from imputena import delete_columns

from test.example_data import *


class TestDeleteColumns(unittest.TestCase):

    # Positive tests ----------------------------------------------------------

    def test_delete_columns_returning(self):
        """
        Positive test

        Checks that the original dataframe still has 10 columns and
        therefore has not been modified.
        Checks that the returned dataframe has 3 columns.
        """
        # 1. Arrange
        df = generate_example_df_divcols()
        # 2. Act
        df2 = delete_columns(df)
        # 3. Assert
        self.assertTrue(len(df.columns) == 8)
        self.assertTrue(len(df2.columns) == 3)

    def test_delete_columns_inplace(self):
        """
        Positive test

        Checks that the dataframe has 3 columns after applying
        delete_columns on it.
        """
        # 1. Arrange
        df = generate_example_df_divcols()
        # 2. Act
        delete_columns(df, inplace=True)
        # 3. Assert
        self.assertTrue(len(df.columns) == 3)

    def test_delete_columns_columns_specified(self):
        """
        Positive test

        columns: ['d', 'e']

        Checks that the dataframe has 7 columns after applying
        delete_columns on it. The column d doesn't have any NA values and
        should therefore not be deleted.
        """
        # 1. Arrange
        df = generate_example_df_divcols()
        # 2. Act
        delete_columns(df, columns=['d', 'e'], inplace=True)
        # 3. Assert
        self.assertTrue(len(df.columns) == 7)

    def test_delete_columns_threshold(self):
        """
        Positive test

        threshold: 8

        Checks that the dataframe has 6 columns after applying
        delete_columns on it. Columns c and h contain less than 8 non-NA
        values.
        """
        # 1. Arrange
        df = generate_example_df_divcols()
        # 2. Act
        delete_columns(df, threshold=8, inplace=True)
        # 3. Assert
        self.assertTrue(len(df.columns) == 6)

    def test_delete_columns_threshold_and_columns_specified(self):
        """
        Positive test

        columns: ['h']
        threshold: 8

        Checks that the dataframe has 7 columns after applying
        delete_columns on it. Columns c and h contain less than 8 non-NA
        values but only column h is being considered.
        """
        # 1. Arrange
        df = generate_example_df_divcols()
        # 2. Act
        delete_columns(df, columns=['h'], threshold=8, inplace=True)
        # 3. Assert
        self.assertTrue(len(df.columns) == 7)

    # Negative tests ----------------------------------------------------------

    def test_delete_columns_wrong_datatype(self):
        """
        Negative test

        data: Series (unsupported datatype)

        Checks that delete_columns raises a TypeError when the data is passed
        as a series.
        """
        # 1. Arrange
        ser = generate_example_series()
        # 2. Act & 3. Assert
        with self.assertRaises(TypeError):
            delete_columns(ser, inplace=True)

    def test_delete_columns_no_data(self):
        """
        Negative test

        data: None

        Checks that delete_columns raises a TypeError when no data is passed.
        """
        # 1. Arrange
        # 2. Act & 3. Assert
        with self.assertRaises(TypeError):
            delete_columns()

    def test_delete_columns_wrong_column(self):
        """
        Negative test

        columns: ['d', 'e', 'z'] (z doesn't exist as a column in the data)

        Checks that delete_columns raises a ValueError when a column name is
        passed that doesn't exist in the dataframe.
        """
        # 1. Arrange
        df = generate_example_df_divcols()
        # 2. Act & 3. Assert
        with self.assertRaises(ValueError):
            delete_columns(df, columns=['d', 'e', 'z'], inplace=True)
