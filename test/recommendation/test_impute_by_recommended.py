import unittest

from imputena import impute_by_recommended

from test.example_data import *


class TestImputeByRecommended(unittest.TestCase):

    # Positive tests for data as a data frame ---------------------------------

    def test_IBR_df_cat(self):
        """
        Positive test

        data: Correct dataframe (df_breast_cancer)

        The data frame contains categorical values.
        Therefore, most-frequent substitution should be used.

        Checks that the original dataframe remains unmodified and that the
        returned dataframe contains 0 NA values, 15 less than the original.
        """
        # 1. Arrange
        df = generate_df_breast_cancer()
        # 2. Act
        df2 = impute_by_recommended(df)
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 15)
        self.assertEqual(df2.isna().sum().sum(), 0)

    def test_IBR_df_num(self):
        """
        Positive test

        data: Correct dataframe (df_sales)

        The data frame contains no categorical values.
        Therefore, imputation using k-NN should be used.

        Checks that the original dataframe remains unmodified and that the
        returned dataframe contains 0 NA values, 8 less than the original.
        """
        # 1. Arrange
        df = generate_df_sales()
        # 2. Act
        df2 = impute_by_recommended(df)
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 8)
        self.assertEqual(df2.isna().sum().sum(), 0)

    def test_IBR_df_col_cat(self):
        """
        Positive test

        data: Correct dataframe (df_breast_cancer)
        column: 'class'

        The column contains categorical values.
        Therefore, logistic regression imputation should be used.

        Checks that the original dataframe remains unmodified and that the
        returned dataframe contains 8 NA values, 7 less than the original.
        """
        # 1. Arrange
        df = generate_df_breast_cancer()
        # 2. Act
        df2 = impute_by_recommended(df, 'class')
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 15)
        self.assertEqual(df2.isna().sum().sum(), 8)

    def test_IBR_df_col_ts(self):
        """
        Positive test

        data: Correct dataframe (example_df_ts)
        column: 'airgap'

        The column contains numerical values and the dataframe has a
        datetime index.
        Therefore, interpolation with seasonal adjustment should be
        used.

        Checks that the original series remains unmodified and that the
        returned series contains 144 NA values.
        """
        # 1. Arrange
        df = generate_example_df_ts()
        # 2. Act
        df2 = impute_by_recommended(df, 'airgap')
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 144 + 13)
        self.assertEqual(df2.isna().sum().sum(), 144 + 0)

    def test_IBR_df_col_little_na(self):
        """
        Positive test

        data: Correct dataframe (example_df_divcols)
        column: 'd'

        The column contains numerical values and the dataframe does not have a
        datetime index. Less than 10% of the data is missing.
        Therefore, mean substitution should be used.

        Since the column doesn't contain NA values, it stays the same.
        """
        # 1. Arrange
        df = generate_example_df_divcols()
        # 2. Act
        df2 = impute_by_recommended(df, 'd')
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 18)
        self.assertEqual(df2.isna().sum().sum(), 18)

    def test_IBR_df_col_low_corr(self):
        """
        Positive test

        data: Correct dataframe (example_df_divcols)
        column: 'h'

        The column contains numerical values and the dataframe does not have a
        datetime index. More than 10% of the data is missing. The column has
        los (<= 0.8) correlation with another column.
        Therefore, linear regression imputation should be used.

        Checks that the original series remains unmodified and that the
        returned series contains 15 NA values.
        """
        # 1. Arrange
        df = generate_example_df_divcols()
        # 2. Act
        df2 = impute_by_recommended(df, 'h')
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 18)
        self.assertEqual(df2.isna().sum().sum(), 15)

    def test_IBR_df_col_high_corr(self):
        """
        Positive test

        data: Correct dataframe (example_df_high_corr)
        column: 'h'

        The column contains numerical values and the dataframe does not have a
        datetime index. More than 10% of the data is missing. The column has
        los (<= 0.8) correlation with another column.
        Therefore, linear regression imputation should be used.

        Checks that the original series remains unmodified and that the
        returned series contains no NA values.
        """
        # 1. Arrange
        df = generate_example_df_high_corr()
        # 2. Act
        df2 = impute_by_recommended(df, 'b')
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 1)
        self.assertEqual(df2.isna().sum().sum(), 0)

    def test_IBR_df_cat_inplace(self):
        """
        Positive test

        data: Correct dataframe (df_breast_cancer)

        The data frame contains categorical values.
        Therefore, most-frequent substitution should be used.

        Checks that the dataframe contains 0 NA values after the operation.
        """
        # 1. Arrange
        df = generate_df_breast_cancer()
        # 2. Act
        impute_by_recommended(df, inplace=True)
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 0)

    # Positive tests for data as a series -------------------------------------

    def test_IBR_series_cat(self):
        """
        Positive test

        data: Correct series (ts_cat)

        The series contains categorical values.
        Therefore, random sample imputation should be used.

        Checks that the original series remains unmodified and that the
        returned series contains no NA values.
        """
        # 1. Arrange
        ts = generate_ts_cat()
        # 2. Act
        ts2 = impute_by_recommended(ts)
        # 3. Assert
        self.assertEqual(ts.isna().sum(), 4)
        self.assertEqual(ts2.isna().sum(), 0)

    def test_IBR_series_num_no_ts(self):
        """
        Positive test

        data: Correct series (example_series)

        The series contains numerical values but not a datetime index.
        Therefore, mean substitution should be used.

        Checks that the original series remains unmodified and that the
        returned series contains no NA values.
        """
        # 1. Arrange
        ser = generate_example_series()
        # 2. Act
        ser2 = impute_by_recommended(ser)
        # 3. Assert
        self.assertEqual(ser.isna().sum(), 3)
        self.assertEqual(ser2.isna().sum(), 0)

    def test_IBR_series_timeseries(self):
        """
        Positive test

        data: Correct series (ts_airgap)

        The series contains numerical values and has a datetime index.
        Therefore, interpolation with seasonal adjustment should be
        recommended.
        """
        # 1. Arrange
        ts = generate_ts_airgap()
        # 2. Act
        ts2 = impute_by_recommended(ts)
        # 3. Assert
        self.assertEqual(ts.isna().sum(), 13)
        self.assertEqual(ts2.isna().sum(), 0)

    # Negative tests ----------------------------------------------------------

    def test_IBR_wrong_type(self):
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
            impute_by_recommended(data)

    def test_IBR_col_for_series(self):
        """
        Negative test

        data: Correct series (ts_cat)
        column: 'a' (series can't have columns=

        Checks that the function raises a ValueError if a column is passed
        for a series.
        """
        # 1. Arrange
        ser = generate_ts_cat()
        # 2. Act & 3. Assert
        with self.assertRaises(ValueError):
            impute_by_recommended(ser, 'a')
