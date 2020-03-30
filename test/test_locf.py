import unittest

from imputena import locf

from .example_data import *


class TestLOCF(unittest.TestCase):

    def test_LOCF_df_returning(self):
        """
        Positive test

        data: Correct dataframe (divcols)

        Checks that the original dataframe remains unmodified and that the
        returned dataframa contains 13 NA values, 5 less than the original.
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
        returned dataframa contains 10 NA values, 8 less than the original.
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
        returned dataframa contains 16 NA values, 2 less than the original.
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

        Checks that locf returns 2 NA values from the specified columns.
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
        returned dataframa contains 14 NA values, 2 less than the original.
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

        Checks that locf returns 4 NA values from the specified columns.
        """
        # 1. Arrange
        df = generate_example_df_divcols()
        # 2. Act
        locf(df, columns=['f', 'g'], fill_leading=True, inplace=True)
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 14)

    # Tests for data as series ------------------------------------------------

    def test_LOCF_series_returning(self):
        """
        Positive test

        data: Correct Series (example series)

        Checks that the original series remains unmodified and that the
        returned series contains 1 NA values, 2 less than the original.
        """
        # 1. Arrange
        es = generate_example_series()
        # 2. Act
        es2 = locf(es)
        # 3. Assert
        self.assertEqual(es.isna().sum(), 3)
        self.assertEqual(es2.isna().sum(), 1)

    def test_LOCF_series_inplace(self):
        """
        Positive test

        data: Correct Series (example series)

        Checks that locf removes 2 NA values from the series.
        """
        # 1. Arrange
        es = generate_example_series()
        # 2. Act
        locf(es, inplace=True)
        # 3. Assert
        self.assertEqual(es.isna().sum(), 1)

    def test_LOCF_series_returning_fill_leading(self):
        """
        Positive test

        data: Correct Series (example series)
        fill_leading: True

        Checks that the original series remains unmodified and that the
        returned series contains 0 NA values.
        """
        # 1. Arrange
        es = generate_example_series()
        # 2. Act
        es2 = locf(es, fill_leading=True)
        # 3. Assert
        self.assertEqual(es.isna().sum(), 3)
        self.assertEqual(es2.isna().sum(), 0)

    def test_LOCF_series_inplace_fill_leading(self):
        """
        Positive test

        data: Correct Series (example series)
        fill_leading: True

        Checks that locf removes all NA values from the series.
        """
        # 1. Arrange
        es = generate_example_series()
        # 2. Act
        locf(es, fill_leading=True, inplace=True)
        # 3. Assert
        self.assertEqual(es.isna().sum(), 0)
