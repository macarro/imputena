import unittest
import logging

from imputena import mice

from test.example_data import *


class TestMICE(unittest.TestCase):

    def setUp(self):
        # Uncomment the following line to show debug and info logs
        # logging.getLogger().setLevel(logging.DEBUG)
        logging.basicConfig(format='%(message)s')
        logging.info("\n" + self._testMethodName)

    # Positive tests ----------------------------------------------------------

    def test_MICE_available(self):
        """
        Positive test

        data: Correct data frame (sales)

        The data frame sales contains 8 NA values.
        mice() should impute all of them.

        Checks that the original dataframe remains unmodified, that mice
        returns 3 dataframes and that each of those contains no NA values.
        """
        # 1. Arrange
        df = generate_df_sales()
        # 2. Act
        dfs = mice(df)
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 8)
        self.assertEqual(len(dfs), 3)
        self.assertEqual(dfs[0].isna().sum().sum(), 0)
        self.assertEqual(dfs[1].isna().sum().sum(), 0)
        self.assertEqual(dfs[2].isna().sum().sum(), 0)

    def test_MICE_complete(self):
        """
        Positive test

        data: Correct data frame (sales)
        regressions: 'complete'

        The data frame sales contains 8 NA values.
        mice() should impute all of them.

        Checks that the original dataframe remains unmodified, that mice
        returns 3 dataframes and that each of those contains 7 NA values.
        """
        # 1. Arrange
        df = generate_df_sales()
        # 2. Act
        dfs = mice(df, regressions='complete')
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 8)
        self.assertEqual(len(dfs), 3)
        self.assertEqual(dfs[0].isna().sum().sum(), 0)
        self.assertEqual(dfs[1].isna().sum().sum(), 0)
        self.assertEqual(dfs[2].isna().sum().sum(), 0)

    # Negative tests ----------------------------------------------------------

    def test_MICE_wrong_type(self):
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
            mice(data)

    def test_MICE_wrong_regressions(self):
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
            mice(df, regressions='z')
