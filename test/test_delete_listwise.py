import unittest

from imputena import delete_listwise

import pandas as pd
import numpy as np


class TestDeleteListwise(unittest.TestCase):

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

    def test_delete_listwise_series_returning(self):
        """
        Positive test

        data: Correct series

        Checks that the original series still has 6 rows and therefore has
        not been modified.
        Checks that the returned series has 3 rows.
        """
        # 1. Arrange
        es = generate_example_series()
        # 2. Act
        es2 = delete_listwise(es)
        # 3. Assert
        self.assertTrue(len(es.index) == 6)
        self.assertTrue(len(es2.index) == 3)

    def test_delete_listwise_series_inplace(self):
        """
        Positive test

        data: Correct series
        inplace: True

        Checks that the series has 3 rows after applying delete_listwise on
        it.
        """
        # 1. Arrange
        es = generate_example_series()
        # 2. Act
        delete_listwise(es, inplace=True)
        # 3. Assert
        self.assertTrue(len(es.index) == 3)

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
        with self.assertRaises(TypeError) as context:
            delete_listwise(array)

    def test_delete_listwise_no_data(self):
        """
        Negative test

        data: None

        Checks that delete_listwise raises a TypeError when no data is passed.
        """
        # 1. Arrange
        # 2. Act & 3. Assert
        with self.assertRaises(TypeError) as context:
            delete_listwise()

    if __name__ == '__main__':
        unittest.main()


# Auxiliary functions ---------------------------------------------------------

def generate_example_df():
    return pd.DataFrame(
        data={
            'x': np.array([18, np.nan, 27, 22]),
            'y': np.array([np.nan, 1, 5, -3.0]),
            'z': np.array([9, 4, 2, 7])
        },
        index=list([x for x in range(1, 5)])
    )


def generate_example_series():
    return pd.Series(
        data=np.array([np.nan, 4, -3, np.nan, 24, np.nan]),
        index=list([x for x in range(1, 7)])
    )
