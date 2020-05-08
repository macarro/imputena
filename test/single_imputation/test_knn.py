import unittest

from imputena import knn

from test.example_data import *


class TestKNN(unittest.TestCase):

    # Positive tests ----------------------------------------------------------

    def test_KNN_returning(self):
        """
        Positive test

        data: Correct data frame (example_df)

        The data frame (example_df) contains 2 NA values.
        knn() should impute all of them.

        Checks that the original data frame remains unmodified and that the
        returned data frame contains no NA values.
        """
        # 1. Arrange
        df = generate_example_df()
        # 2. Act
        df2 = knn(df)
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 2)
        self.assertEqual(df2.isna().sum().sum(), 0)

    def test_KNN_inplace(self):
        """
        Positive test

        data: Correct data frame (example_df)

        The data frame (example_df) contains 2 NA values.
        knn() should impute both of them.

        Checks that the data frame contains no NA values after the operation.
        """
        # 1. Arrange
        df = generate_example_df()
        # 2. Act
        knn(df, inplace=True)
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 0)

    def test_KNN_columns(self):
        """
        Positive test

        data: Correct data frame (example_df)
        columns: ['x']

        The data frame (example_df) contains 2 NA values.
        knn() should impute 1 of them.

        Checks that the original data frame remains unmodified and that the
        returned data frame contains 1 NA value.
        """
        # 1. Arrange
        df = generate_example_df()
        # 2. Act
        df2 = knn(df, columns=['x'])
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 2)
        self.assertEqual(df2.isna().sum().sum(), 1)

    def test_KNN_k(self):
        """
        Positive test

        data: Correct data frame (example_df)
        k: 1

        The data frame (example_df) contains 2 NA values.
        knn() should impute both of them.

        Checks that the original data frame remains unmodified and that the
        returned data frame contains no NA values.
        """
        # 1. Arrange
        df = generate_example_df()
        # 2. Act
        df2 = knn(df, k=1)
        # 3. Assert
        self.assertEqual(df.isna().sum().sum(), 2)
        self.assertEqual(df2.isna().sum().sum(), 0)

    # Negative tests ----------------------------------------------------------

    def test_KNN_wrong_type(self):
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
            knn(data)

    def test_KNN_wrong_column(self):
        """
        Negative test

        data: Correct data frame (example_df)
        columns: ['x', 'a'] ('a' is not a column of example_df)

        Checks that the function raises a ValueError if one of the specified
        columns doesn't exist in the data.
        """
        # 1. Arrange
        df = generate_example_df()
        # 2. Act
        df2 = knn(df, columns=['x'])
        # 2. Act & 3. Assert
        with self.assertRaises(ValueError):
            knn(df, columns=['x', 'a'])
