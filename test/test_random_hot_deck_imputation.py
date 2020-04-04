import unittest

from imputena import random_hot_deck_imputation

from .example_data import *


class TestRandomHotDeckImputation(unittest.TestCase):

    def test_rhdi_returning(self):
        df = pd.DataFrame(
            data={
                'x': ['A', 'A', 'A', 'B'],
                'y': ['i', 'i', 'j', 'j'],
                'z': np.array([5, np.nan, 27, 22])
            },
            index=list([x for x in range(1, 5)])
        )
        print("\n\nBEFORE: =========")
        print(df)
        df2 = random_hot_deck_imputation(df, incomplete_variable='z',
                                  deck_variables=['x'], inplace=False)
        print("\n\nAFTER: =========")
        print(df2)


