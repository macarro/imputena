import unittest
import logging

from imputena import srmi

from test.example_data import *


class TestSRMI(unittest.TestCase):

    def setUp(self):
        # Uncomment the following line to show debug and info logs
        # logging.getLogger().setLevel(logging.DEBUG)
        logging.basicConfig(format='%(message)s')
        logging.info("\n" + self._testMethodName)

    # Positive tests ----------------------------------------------------------

    def test_SRMI_available(self):
        """
        Positive test

        data: Correct data frame (sales)

        The data frame sales contains 8 NA values.
        srmi() should impute 5 of them.

        Checks that the original dataframe remains unmodified, that srmi
        returns 3 dataframes and that each of those contains 3 NA values.
        """
        # 1. Arrange
        df = generate_df_sales()
        # 2. Act
        dfs = srmi(df)
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 8)
        self.assertEqual(len(dfs), 3)
        self.assertEqual(dfs[0].isna().sum().sum(), 3)
        self.assertEqual(dfs[1].isna().sum().sum(), 3)
        self.assertEqual(dfs[2].isna().sum().sum(), 3)

    def test_SRMI_complete(self):
        """
        Positive test

        data: Correct data frame (sales)
        regressions: 'complete'

        The data frame sales contains 8 NA values.
        srmi() should impute one of them.

        Checks that the original dataframe remains unmodified, that srmi
        returns 3 dataframes and that each of those contains 7 NA values.
        """
        # 1. Arrange
        df = generate_df_sales()
        # 2. Act
        dfs = srmi(df, regressions='complete')
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 8)
        self.assertEqual(len(dfs), 3)
        self.assertEqual(dfs[0].isna().sum().sum(), 7)
        self.assertEqual(dfs[1].isna().sum().sum(), 7)
        self.assertEqual(dfs[2].isna().sum().sum(), 7)

    def test_SRMI_sample_size_1(self):
        """
        Positive test

        data: Correct data frame (sales)
        sample_size: 1

        The data frame sales contains 8 NA values.
        srmi() should impute 5 of them.

        Checks that the original dataframe remains unmodified, that srmi
        returns 3 dataframes and that each of those contains 3 NA values.
        """
        # 1. Arrange
        df = generate_df_sales()
        # 2. Act
        dfs = srmi(df, sample_size=1)
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 8)
        self.assertEqual(len(dfs), 3)
        self.assertEqual(dfs[0].isna().sum().sum(), 3)
        self.assertEqual(dfs[1].isna().sum().sum(), 3)
        self.assertEqual(dfs[2].isna().sum().sum(), 3)

    # Negative tests ----------------------------------------------------------

    def test_SRMI_wrong_type(self):
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
            srmi(data)

    def test_SRMI_wrong_regressions(self):
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
            srmi(df, regressions='z')

    def test_SRMI_sample_size_0(self):
        """
        Positive test

        data: Correct data frame (sales)
        sample_size: 0 (invalid)

        Checks that the function raises a ValueError if sample_size is set to 0
        """
        # 1. Arrange
        df = generate_df_sales()
        # 2. Act & 3. Assert
        with self.assertRaises(ValueError):
            srmi(df, sample_size=0)

    def test_SRMI_sample_size_neg(self):
        """
        Positive test

        data: Correct data frame (sales)
        sample_size: -1 (invalid)

        Checks that the function raises a ValueError if sample_size is set
        to a negative value
        """
        # 1. Arrange
        df = generate_df_sales()
        # 2. Act & 3. Assert
        with self.assertRaises(ValueError):
            srmi(df, sample_size=-1)
