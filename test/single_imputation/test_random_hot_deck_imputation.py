import unittest

from imputena import random_hot_deck_imputation

from test.example_data import *


class TestRandomHotDeckImputation(unittest.TestCase):

    # Positive tests ----------------------------------------------------------

    def test_RHDI_returning(self):
        """
        Positive test

        data: Correct dataframe (hotdeck)

        Checks that the original dataframe remains unmodified and that the
        returned dataframe contains 1 NA value, 2 less than the original.
        """
        # 1. Arrange
        df = generate_example_df_hotdeck()
        # 2. Act
        df2 = random_hot_deck_imputation(
            df, incomplete_variable='a', deck_variables=['b'], inplace=False)
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 3)
        self.assertEqual(df2.isna().sum().sum(), 1)

    def test_RHDI_inplace(self):
        """
        Positive test

        data: Correct dataframe (hotdeck)

        Checks that random_hot_deck_imputation removes 2 NA values from the
        dataframe.
        """
        # 1. Arrange
        df = generate_example_df_hotdeck()
        # 2. Act
        random_hot_deck_imputation(
            df, incomplete_variable='a', deck_variables=['b'], inplace=True)
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 1)

    # Negative tests ----------------------------------------------------------

    def test_RHDI_no_donors(self):
        """
        Negative test

        data: Correct dataframe (hotdeck)
        incomplete_variable: a
        deck_variables: ['b', 'c']

        Checks that no NA value gets removed because no row coincides in
        value for the variable c.
        """
        # 1. Arrange
        df = generate_example_df_hotdeck()
        # 2. Act
        df2 = random_hot_deck_imputation(
            df, incomplete_variable='a', deck_variables=['b', 'c'],
            inplace=False)
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 3)
        self.assertEqual(df2.isna().sum().sum(), 3)

    def test_RHDI_wrong_type(self):
        """
        Negative test

        data: Series (unsupported type)

        Checks that the random_hot_deck_imputation raises a TypeError if the
        data is passed as a series.
        """
        # 1. Arrange
        data = generate_example_series()
        # 2. Act & 3. Assert
        with self.assertRaises(TypeError):
            random_hot_deck_imputation(
                data, incomplete_variable='a', deck_variables=['b'],
                inplace=True)

    def test_RHDI_wrong_incomplete_variable(self):
        """
        Negative test

        data: Correct dataframe (hotdeck)
        incomplete_variable: 'z' (doesn't exist in the data)

        Checks that random_hot_deck_imputation raises a ValueError if the
        given incomplete variable doesn't exist in the data.
        """
        # 1. Arrange
        data = generate_example_df_hotdeck()
        # 2. Act & 3. Assert
        with self.assertRaises(ValueError):
            random_hot_deck_imputation(
                data, incomplete_variable='z', deck_variables=['b'],
                inplace=True)

    def test_RHDI_wrong_deck_variable(self):
        """
        Negative test

        data: Correct dataframe (hotdeck)
        deck_variables: ['b', 'z'] ('z' doesn't exist in the data)

        Checks that random_hot_deck_imputation raises a ValueError if one of
        the given deck variables doesn't exist in the data.
        """
        # 1. Arrange
        data = generate_example_df_hotdeck()
        # 2. Act & 3. Assert
        with self.assertRaises(ValueError):
            random_hot_deck_imputation(
                data, incomplete_variable='a', deck_variables=['b', 'z'],
                inplace=True)
