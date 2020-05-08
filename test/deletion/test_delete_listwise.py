import unittest

from imputena import delete_listwise

from test.example_data import *


class TestDeleteListwise(unittest.TestCase):

    # Positive tests for data as a dataframe ----------------------------------

    def test_delete_listwise_df_returning(self):
        """
        Positive test

        data: Correct dataframe

        Checks that the original dataframe still has 4 rows and therefore has
        not been modified.
        Checks that the returned dataframe has 2 rows.
        """
        # 1. Arrange
        df = generate_example_df()
        # 2. Act
        df2 = delete_listwise(df)
        # 3. Assert
        self.assertTrue(len(df.index) == 4)
        self.assertTrue(len(df2.index) == 2)

    def test_delete_listwise_df_inplace(self):
        """
        Positive test

        data: Correct dataframe
        inplace: True

        Checks that the dataframe has 2 rows after applying delete_listwise on
        it.
        """
        # 1. Arrange
        df = generate_example_df()
        # 2. Act
        delete_listwise(df, inplace=True)
        # 3. Assert
        self.assertTrue(len(df.index) == 2)

    def test_delete_listwise_threshold(self):
        """
        Positive test

        data: Correct dataframe
        threshold: 2

        Checks that the dataframe has 4 rows after applying delete_listwise
        with threshold 2 on it.
        """
        # 1. Arrange
        df = generate_example_df()
        # 2. Act
        delete_listwise(df, threshold=2, inplace=True)
        # 3. Assert
        self.assertTrue(len(df.index) == 4)

    # Positive tests for data as a series -------------------------------------

    def test_delete_listwise_series_returning(self):
        """
        Positive test

        data: Correct series

        Checks that the original series still has 6 rows and therefore has
        not been modified.
        Checks that the returned series has 3 rows.
        """
        # 1. Arrange
        ser = generate_example_series()
        # 2. Act
        ser2 = delete_listwise(ser)
        # 3. Assert
        self.assertTrue(len(ser.index) == 6)
        self.assertTrue(len(ser2.index) == 3)

    def test_delete_listwise_series_inplace(self):
        """
        Positive test

        data: Correct series
        inplace: True

        Checks that the series has 3 rows after applying delete_listwise on
        it.
        """
        # 1. Arrange
        ser = generate_example_series()
        # 2. Act
        delete_listwise(ser, inplace=True)
        # 3. Assert
        self.assertTrue(len(ser.index) == 3)

    # Negative tests ----------------------------------------------------------

    def test_delete_listwise_wrong_datatype(self):
        """
        Negative test

        data: Array (unsupported datatype)

        Checks that delete_listwise raises a TypeError when the data is passed
        as an array.
        """
        # 1. Arrange
        array = [1, 2, 3, 4]
        # 2. Act & 3. Assert
        with self.assertRaises(TypeError):
            delete_listwise(array)

    def test_delete_listwise_no_data(self):
        """
        Negative test

        data: None

        Checks that delete_listwise raises a TypeError when no data is passed.
        """
        # 1. Arrange
        # 2. Act & 3. Assert
        with self.assertRaises(TypeError):
            delete_listwise()

    def test_delete_listwise_threshold_series(self):
        """
        Negative test

        data: Series
        threshold: 2, even though delete_listwise shouldn't receive a threshold
        if the data is a series

        Checks that the delete_listwise raises a ValueError if the data is a
        series and a threshold is passed.
        """
        # 1. Arrange
        ser = generate_example_series()
        # 2. Act & 3. Assert
        with self.assertRaises(ValueError):
            delete_listwise(ser, threshold=2, inplace=True)
