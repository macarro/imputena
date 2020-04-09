import unittest

from imputena import seasonal_interpolation

from .example_data import *


class TestSeasonalInterpolation(unittest.TestCase):

    # Positive tests for data as a series -------------------------------------

    def test_SI_series_returning(self):
        """
        Positive test

        data: Correct series (ts_airgap)

        The original series (ts_airgap) contains 13 NA values.
        seasonal_interpolation() should imputate all of them.

        Checks that the original series remains unmodified and that the
        returned series contains no NA values.
        """
        # 1. Arrange
        df = ts_airgap()
        # 2. Act
        df2 = seasonal_interpolation(df)
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 13)
        self.assertEqual(df2.isna().sum().sum(), 0)

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
        df = ts_airgap()
        # 2. Act
        df2 = seasonal_interpolation(df, int_direction='forward')
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 13)
        self.assertEqual(df2.isna().sum().sum(), 1)
