import unittest

from imputena import recommend_method

from test.example_data import *


class TestRecommendMethod(unittest.TestCase):

    # Positive tests for data as a data frame ---------------------------------

    def test_recommend_method_df_cat(self):
        """
        Positive test

        data: Correct dataframe (df_breast_cancer)

        The data frame contains categorical values.
        Therefore, most-frequent substitution should be recommended.
        """
        # 1. Arrange
        ser = generate_df_breast_cancer()
        # 2. Act
        method = recommend_method(ser, title_only=True)
        # 3. Assert
        self.assertEqual(method, 'most-frequent substitution')

    def test_recommend_method_df_num(self):
        """
        Positive test

        data: Correct dataframe (df_sales)

        The data frame contains no categorical values.
        Therefore, imputation using k-NN should be recommended.
        """
        # 1. Arrange
        ser = generate_df_sales()
        # 2. Act
        method = recommend_method(ser, title_only=True)
        # 3. Assert
        self.assertEqual(method, 'imputation using k-NN')

    def test_recommend_method_df_col_cat(self):
        """
        Positive test

        data: Correct dataframe (df_breast_cancer)
        column: 'class'

        The column contains categorical values.
        Therefore, logistic regression imputation should be recommended.
        """
        # 1. Arrange
        ser = generate_df_breast_cancer()
        # 2. Act
        method = recommend_method(ser, 'class', title_only=True)
        # 3. Assert
        self.assertEqual(method, 'logistic regression imputation')

    def test_recommend_method_df_col_ts(self):
        """
        Positive test

        data: Correct dataframe (example_df_ts)
        column: 'airgap'

        The column contains numerical values and the dataframe has a
        datetime index.
        Therefore, interpolation with seasonal adjustment should be
        recommended.
        """
        # 1. Arrange
        ser = generate_example_df_ts()
        # 2. Act
        method = recommend_method(ser, 'airgap', title_only=True)
        # 3. Assert
        self.assertEqual(method, 'interpolation with seasonal adjustment')

    def test_recommend_method_df_col_little_na(self):
        """
        Positive test

        data: Correct dataframe (example_df_divcols)
        column: 'd'

        The column contains numerical values and the dataframe does not have a
        datetime index. Less than 10% of the data is missing.
        Therefore, mean substitution should be recommended.
        """
        # 1. Arrange
        ser = generate_example_df_divcols()
        # 2. Act
        method = recommend_method(ser, 'd', title_only=True)
        # 3. Assert
        self.assertEqual(method, 'mean substitution')

    def test_recommend_method_df_col_low_corr(self):
        """
        Positive test

        data: Correct dataframe (example_df_divcols)
        column: 'h'

        The column contains numerical values and the dataframe does not have a
        datetime index. More than 10% of the data is missing. The column has
        los (<= 0.8) correlation with another column.
        Therefore, linear regression imputation should be recommended.
        """
        # 1. Arrange
        ser = generate_example_df_divcols()
        # 2. Act
        method = recommend_method(ser, 'h', title_only=True)
        # 3. Assert
        self.assertEqual(method, 'imputation using k-NN')

    def test_recommend_method_df_col_high_corr(self):
        """
        Positive test

        data: Correct dataframe (example_df_high_corr)
        column: 'h'

        The column contains numerical values and the dataframe does not have a
        datetime index. More than 10% of the data is missing. The column has
        los (<= 0.8) correlation with another column.
        Therefore, linear regression imputation should be recommended.
        """
        # 1. Arrange
        ser = generate_example_df_high_corr()
        # 2. Act
        method = recommend_method(ser, 'b', title_only=True)
        # 3. Assert
        self.assertEqual(method, 'linear regression imputation')

    def test_recommend_method_df_process_description(self):
        """
        Positive test

        data: Correct dataframe (df_breast_cancer)
        title_only: False

        Since title_only is false, the full message should be shown, in this
        case, 4 lines
        """
        # 1. Arrange
        ser = generate_df_breast_cancer()
        # 2. Act
        message = recommend_method(ser)
        # 3. Assert
        self.assertEqual(message.count('\n')+1, 4)

    # Positive tests for data as a series -------------------------------------

    def test_recommend_method_series_cat(self):
        """
        Positive test

        data: Correct series (ts_cat)

        The series contains categorical values.
        Therefore, random sample imputation should be recommended.
        """
        # 1. Arrange
        ser = generate_ts_cat()
        # 2. Act
        method = recommend_method(ser, title_only=True)
        # 3. Assert
        self.assertEqual(method, 'random sample imputation')

    def test_recommend_method_series_num_no_ts(self):
        """
        Positive test

        data: Correct series (example_series)

        The series contains numerical values but not a datetime index.
        Therefore, mean substitution should be recommended.
        """
        # 1. Arrange
        ser = generate_example_series()
        # 2. Act
        method = recommend_method(ser, title_only=True)
        # 3. Assert
        self.assertEqual(method, 'mean substitution')

    def test_recommend_method_series_timeseries(self):
        """
        Positive test

        data: Correct series (ts_airgap)

        The series contains numerical values and has a datetime index.
        Therefore, interpolation with seasonal adjustment should be
        recommended.
        """
        # 1. Arrange
        ser = generate_ts_airgap()
        # 2. Act
        method = recommend_method(ser, title_only=True)
        # 3. Assert
        self.assertEqual(method, 'interpolation with seasonal adjustment')

    # Negative tests ----------------------------------------------------------

    def test_recommend_method_wrong_type(self):
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
            recommend_method(data)

    def test_recommend_method_col_for_series(self):
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
            recommend_method(ser, 'a')
