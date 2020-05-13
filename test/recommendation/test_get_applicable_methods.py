import unittest

from imputena import get_applicable_methods

from test.example_data import *


class TestGetApplicableMethods(unittest.TestCase):

    # Positive tests for data as a data frame ---------------------------------

    def test_GAM_df_cat_no_ts(self):
        """
        Positive test

        data: Correct dataframe (example_df_hotdeck[['b', 'c', 'd']])

        The data frame contains only categorical values and is not temporal.

        Checks that get_applicable_methods returns all applicable methods
        and nothing more.
        """
        # 1. Arrange
        df = generate_example_df_hotdeck()[['b', 'c', 'd']]
        # 2. Act
        methods = get_applicable_methods(df)
        # 3. Assert
        self.assertSetEqual(methods, {
            'listwise deletion',
            'pairwise deletion',
            'variable deletion',
            'random sample imputation',
            'random hot-deck imputation',
            'most-frequent substitution',
            'constant value substitution',
            'srmi',
            'mice',
            'logistic regression imputation'
        })

    def test_GAM_df_cat_ts(self):
        """
        Positive test

        data: Correct dataframe (example_df_ts_cat)

        The data frame contains only categorical values and is temporal.

        Checks that get_applicable_methods returns all applicable methods
        and nothing more.
        """
        # 1. Arrange
        df = generate_example_df_ts_cat()
        # 2. Act
        methods = get_applicable_methods(df)
        # 3. Assert
        self.assertSetEqual(methods, {
            'listwise deletion',
            'pairwise deletion',
            'variable deletion',
            'random sample imputation',
            'random hot-deck imputation',
            'most-frequent substitution',
            'constant value substitution',
            'srmi',
            'mice',
            'LOCF',
            'NOCB',
            'logistic regression imputation'
        })

    def test_GAM_df_cat_and_num_no_ts(self):
        """
        Positive test

        data: Correct dataframe (example_df_hotdeck)

        The data frame contains categorical and numerical values and is not
        temporal.

        Checks that get_applicable_methods returns all applicable methods
        and nothing more.
        """
        # 1. Arrange
        df = generate_example_df_hotdeck()
        # 2. Act
        methods = get_applicable_methods(df)
        # 3. Assert
        self.assertSetEqual(methods, {
            'listwise deletion',
            'pairwise deletion',
            'variable deletion',
            'random sample imputation',
            'random hot-deck imputation',
            'most-frequent substitution',
            'constant value substitution',
            'srmi',
            'mice',
        })

    def test_GAM_df_cat_and_num_ts(self):
        """
        Positive test

        data: Correct dataframe (example_df_ts_cat_and_num)

        The data frame contains categorical and numerical values and is
        temporal.

        Checks that get_applicable_methods returns all applicable methods
        and nothing more.
        """
        # 1. Arrange
        df = generate_example_df_ts_cat_and_num()
        # 2. Act
        methods = get_applicable_methods(df)
        # 3. Assert
        self.assertSetEqual(methods, {
            'listwise deletion',
            'pairwise deletion',
            'variable deletion',
            'random sample imputation',
            'random hot-deck imputation',
            'most-frequent substitution',
            'constant value substitution',
            'srmi',
            'mice',
            'LOCF',
            'NOCB',
        })

    def test_GAM_df_num_no_ts(self):
        """
        Positive test

        data: Correct dataframe (df_sales)

        The data frame contains only numerical values and is not temporal.

        Checks that get_applicable_methods returns all applicable methods
        and nothing more.
        """
        # 1. Arrange
        df = generate_df_sales()
        # 2. Act
        methods = get_applicable_methods(df)
        # 3. Assert
        self.assertSetEqual(methods, {
            'listwise deletion',
            'pairwise deletion',
            'variable deletion',
            'random sample imputation',
            'random hot-deck imputation',
            'most-frequent substitution',
            'constant value substitution',
            'srmi',
            'mice',
            'mean substitution',
            'median substitution',
            'random value imputation',
            'linear regression',
            'stochastic regression',
            'imputation using k-NN'
        })

    def test_GAM_df_num_ts(self):
        """
        Positive test

        data: Correct dataframe (example_df_ts)

        The data frame contains only numerical values and is temporal.

        Checks that get_applicable_methods returns all applicable methods
        and nothing more.
        """
        # 1. Arrange
        df = generate_example_df_ts()
        # 2. Act
        methods = get_applicable_methods(df)
        # 3. Assert
        self.assertSetEqual(methods, {
            'listwise deletion',
            'pairwise deletion',
            'variable deletion',
            'random sample imputation',
            'random hot-deck imputation',
            'most-frequent substitution',
            'constant value substitution',
            'srmi',
            'mice',
            'LOCF',
            'NOCB',
            'mean substitution',
            'median substitution',
            'random value imputation',
            'linear regression',
            'stochastic regression',
            'imputation using k-NN',
            'interpolation',
            'interpolation with seasonal adjustment'
        })

    # Positive tests for data as a series -------------------------------------

    def test_GAM_ser_cat_no_ts(self):
        """
        Positive test

        data: Correct series (df_breast_cancer['class'])

        The series contains only categorical values and is not temporal.

        Checks that get_applicable_methods returns all applicable methods
        and nothing more.
        """
        # 1. Arrange
        ser = generate_df_breast_cancer()['class']
        # 2. Act
        methods = get_applicable_methods(ser)
        # 3. Assert
        self.assertSetEqual(methods, {
            'listwise deletion',
            'random sample imputation',
            'most-frequent substitution',
            'constant value substitution',
        })

    def test_GAM_ser_cat_ts(self):
        """
        Positive test

        data: Correct series (ts_cat)

        The series contains only categorical values and is temporal.

        Checks that get_applicable_methods returns all applicable methods
        and nothing more.
        """
        # 1. Arrange
        ser = generate_ts_cat()
        # 2. Act
        methods = get_applicable_methods(ser)
        # 3. Assert
        self.assertSetEqual(methods, {
            'listwise deletion',
            'random sample imputation',
            'most-frequent substitution',
            'constant value substitution',
            'LOCF',
            'NOCB'
        })

    def test_GAM_ser_num_no_ts(self):
        """
        Positive test

        data: Correct series (example_series)

        The series contains only numerical values and is not temporal.

        Checks that get_applicable_methods returns all applicable methods
        and nothing more.
        """
        # 1. Arrange
        ser = generate_example_series()
        # 2. Act
        methods = get_applicable_methods(ser)
        # 3. Assert
        self.assertSetEqual(methods, {
            'listwise deletion',
            'random sample imputation',
            'most-frequent substitution',
            'constant value substitution',
            'mean substitution',
            'median substitution',
            'random value imputation'
        })

    def test_GAM_ser_num_ts(self):
        """
        Positive test

        data: Correct series (ts_airgap)

        The series contains only numerical values and is temporal.

        Checks that get_applicable_methods returns all applicable methods
        and nothing more.
        """
        # 1. Arrange
        df = generate_ts_airgap()
        # 2. Act
        methods = get_applicable_methods(df)
        # 3. Assert
        self.assertSetEqual(methods, {
            'listwise deletion',
            'random sample imputation',
            'most-frequent substitution',
            'constant value substitution',
            'LOCF',
            'NOCB',
            'mean substitution',
            'median substitution',
            'random value imputation',
            'interpolation',
            'interpolation with seasonal adjustment'
        })

    # Negative tests ----------------------------------------------------------

    def test_GAM_wrong_type(self):
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
            get_applicable_methods(data)
