import unittest

from imputena import delete_pairwise

from test.example_data import *


class TestDeletePairwise(unittest.TestCase):

    # Positive tests ----------------------------------------------------------

    def test_delete_pairwise_returning(self):
        """
        Positive test

        data: Correct dataframe

        Checks that the original dataframe still has 4 rows and therefore has
        not been modified.
        Checks that the returned dataframe has 3 rows.
        """
        # 1. Arrange
        df = generate_example_df()
        # 2. Act
        df2 = delete_pairwise(df, ['x', 'z'])
        # 3. Assert
        self.assertTrue(len(df.index) == 4)
        self.assertTrue(len(df2.index) == 3)

    def test_delete_pairwise_inplace(self):
        """
        Positive test

        data: Correct dataframe
        inplace: True

        Checks that the dataframe has 3 rows after applying delete_pairwise for
        the columns c and z on it.
        """
        # 1. Arrange
        df = generate_example_df()
        # 2. Act
        delete_pairwise(df, ['x', 'z'], inplace=True)
        # 3. Assert
        self.assertTrue(len(df.index) == 3)

    def test_delete_pairwise_threshold(self):
        """
        Positive test

        data: Correct dataframe
        threshold: 2

        Checks that the dataframe has 3 rows after applying delete_pairwise
        with threshold 2 on it.
        """
        # 1. Arrange
        df = generate_example_df()
        # 2. Act
        delete_pairwise(df, ['x', 'z'], threshold=2, inplace=True)
        # 3. Assert
        self.assertTrue(len(df.index) == 3)

    # Negative tests ----------------------------------------------------------

    def test_delete_pairwise_wrong_datatype(self):
        """
        Negative test

        data: Series (unsupported datatype)

        Checks that delete_pairwise raises a TypeError when the data is passed
        as a series.
        """
        # 1. Arrange
        ser = generate_example_series()
        # 2. Act & 3. Assert
        with self.assertRaises(TypeError):
            delete_pairwise(ser, ['x', 'z'])

    def test_delete_pairwise_no_data(self):
        """
        Negative test

        data: None

        Checks that delete_pairwise raises a TypeError when no data is passed.
        """
        # 1. Arrange
        # 2. Act & 3. Assert
        with self.assertRaises(TypeError):
            delete_pairwise(columns=['x', 'z'])

    def test_delete_pairwise_wrong_column(self):
        """
        Negative test

        data: Correct dataframe
        columns: ['x', 'a'], a is not a column name of the dataframe

        Checks that the delete_pairwise raises a
        """
        # 1. Arrange
        df = generate_example_df()
        # 2. Act & 3. Assert
        with self.assertRaises(ValueError):
            delete_pairwise(df, ['x', 'a'], inplace=True)
