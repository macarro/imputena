import unittest

from imputena import seasonal_interpolation

from test.example_data import *


class TestSeasonalInterpolation(unittest.TestCase):

    # Positive tests for data as a dataframe ----------------------------------

    def test_SI_df_returning(self):
        """
        Positive test

        data: Correct data frame (example_df_ts)

        The data frame (example_df_ts) contains 144+13 NA values.
        seasonal_interpolation() should impute 13 of them.

        Checks that the original data frame remains unmodified and that the
        returned data frame contains 144 NA values.
        """
        # 1. Arrange
        df = generate_example_df_ts()
        # 2. Act
        df2 = seasonal_interpolation(df)
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 144+13)
        self.assertEqual(df2.isna().sum().sum(), 144+0)

    def test_SI_df_inplace(self):
        """
        Positive test

        data: Correct data frame (example_df_ts)
        inplace: True

        The data frame (example_df_ts) contains 144+13 NA values.
        seasonal_interpolation() should impute 13 of them.

        Checks that the data frame contains 144 NA values after the operation.
        """
        # 1. Arrange
        df = generate_example_df_ts()
        # 2. Act
        seasonal_interpolation(df, inplace=True)
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 144+0)

    def test_SI_df_columns(self):
        """
        Positive test

        data: Correct data frame (example_df_ts)
        columns: ['airgap']

        The data frame (example_df_ts) contains 144+13 NA values.
        seasonal_interpolation() should impute 13 of them.

        Checks that the original series remains unmodified and that the
        returned series contains 144 NA values.
        """
        # 1. Arrange
        df = generate_example_df_ts()
        # 2. Act
        df2 = seasonal_interpolation(df, columns=['airgap'])
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 144+13)
        self.assertEqual(df2.isna().sum().sum(), 144+0)

    # Positive tests for data as a series -------------------------------------

    def test_SI_series_returning(self):
        """
        Positive test

        data: Correct series (ts_airgap)

        The series (ts_airgap) contains 13 NA values.
        seasonal_interpolation() should impute all of them.

        Checks that the original series remains unmodified and that the
        returned series contains no NA values.
        """
        # 1. Arrange
        ts = generate_ts_airgap()
        # 2. Act
        ts2 = seasonal_interpolation(ts)
        # 3. Assert
        self.assertEqual(ts.isna().sum().sum(), 13)
        self.assertEqual(ts2.isna().sum().sum(), 0)

    def test_SI_series_inplace(self):
        """
        Positive test

        data: Correct series (ts_airgap)
        inplace: True

        The series (ts_airgap) contains 13 NA values.
        seasonal_interpolation() should impute all of them.

        Checks that the series contains no NA values after the operation.
        """
        # 1. Arrange
        ts = generate_ts_airgap()
        # 2. Act
        seasonal_interpolation(ts, inplace=True)
        # 3. Assert
        self.assertEqual(ts.isna().sum().sum(), 0)

    def test_SI_series_forward(self):
        """
        Positive test

        data: Correct series (ts_airgap)
        int_direction='forward'

        The original series (ts_airgap) contains 13 NA values.
        seasonal_interpolation() should impute all but one of them, because
        there is a NA in the fifth position of the timeseries, and 5 is less
        then half of the window size (6).

        Checks that the original series remains unmodified and that the
        returned series contains 1 NA value.
        """
        # 1. Arrange
        ts = generate_ts_airgap()
        # 2. Act
        ts2 = seasonal_interpolation(ts, int_direction='forward')
        # 3. Assert
        self.assertEqual(ts.isna().sum().sum(), 13)
        self.assertEqual(ts2.isna().sum().sum(), 1)

    def test_SI_series_quadratic_interpolation(self):
        """
        Positive test

        data: Correct series (ts_airgap)
        int_method: 'quadratic'

        The series (ts_airgap) contains 13 NA values.
        seasonal_interpolation() should impute all but one of them, because
        there is a NA in the fifth position of the timeseries, and 5 is less
        then half of the window size (6).

        Checks that the original series remains unmodified and that the
        returned series contains no NA values.
        """
        # 1. Arrange
        ts = generate_ts_airgap()
        # 2. Act
        ts2 = seasonal_interpolation(ts, int_method='quadratic')
        # 3. Assert
        self.assertEqual(ts.isna().sum().sum(), 13)
        self.assertEqual(ts2.isna().sum().sum(), 1)

    def test_SI_series_cubic_interpolation(self):
        """
        Positive test

        data: Correct series (ts_airgap)
        int_method: 'cubic'

        The series (ts_airgap) contains 13 NA values.
        seasonal_interpolation() should impute all but one of them, because
        there is a NA in the fifth position of the timeseries, and 5 is less
        then half of the window size (6).

        Checks that the original series remains unmodified and that the
        returned series contains no NA values.
        """
        # 1. Arrange
        ts = generate_ts_airgap()
        # 2. Act
        ts2 = seasonal_interpolation(ts, int_method='cubic')
        # 3. Assert
        self.assertEqual(ts.isna().sum().sum(), 13)
        self.assertEqual(ts2.isna().sum().sum(), 1)

    def test_SI_series_additive_model(self):
        """
        Positive test

        data: Correct series (ts_ausbeer)
        dec_model='additive'

        The series (ts_ausbeer) contains 9 NA values.
        seasonal_interpolation() should impute all of them.

        Checks that the original series remains unmodified and that the
        returned series contains no NA values.
        """
        # 1. Arrange
        ts = generate_ts_ausbeer()
        # 2. Act
        ts2 = seasonal_interpolation(ts, dec_model='additive')
        # 3. Assert
        self.assertEqual(ts.isna().sum().sum(), 9)
        self.assertEqual(ts2.isna().sum().sum(), 0)

    # Negative tests ----------------------------------------------------------

    def test_SI_wrong_type(self):
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
            seasonal_interpolation(data)

    def test_SI_df_wrong_column(self):
        """
        Negative test

        data: Correct data frame (example_df_ts)
        columns: ['z'] ('z' is not a column of example_df_ts)

        Checks that the function raises a ValueError if one of the specified
        columns doesn't exist in the data.
        """
        # 1. Arrange
        df = generate_example_df_ts()
        # 2. Act & 3. Assert
        with self.assertRaises(ValueError):
            seasonal_interpolation(df, columns=['z'])

    def test_SI_wrong_dec_model(self):
        """
        Negative test

        data: Correct data frame (example_df_ts)
        dec_model='z' (not a valid decomposition model)

        Checks that the function raises a ValueError if the value of
        dec_model is not valid.
        """
        # 1. Arrange
        df = generate_example_df_ts()
        # 2. Act & 3. Assert
        with self.assertRaises(ValueError):
            seasonal_interpolation(df, dec_model='z')
