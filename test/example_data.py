import pandas as pd
import numpy as np


def generate_example_df():
    return pd.DataFrame(
        data={
            'x': np.array([18, np.nan, 27, 22]),
            'y': np.array([np.nan, 1, 5, -3.0]),
            'z': np.array([9, 4, 2, 7])
        },
        index=list([x for x in range(1, 5)])
    )


def generate_example_df_divcols():
    """
    Example data frame with 10 rows for 8 columns. Has diverse columns.

    By columns:
    3 columns (a, b and d) contain only non-NA values.
    1 column (e) contains 1 NA value.
    2 columns (f and g) contain 2 NA values.
    1 column (h) contains 3 NA values.
    1 column (c) contains only NA values.

    By rows:
    5 rows (5 - 9) contain 1 NA value.
    3 rows (2, 4 and 10) contain 2 NA values.
    1 row (3) contains 3 NA values.
    1 row (1) contains 4 NA values.

    Representation:

         a  b   c     d     e     f    g    h
    1    1  0 NaN   7.4   NaN   NaN  NaN  5.0
    2    2  0 NaN   5.2  -6.2  -3.0  2.0  NaN
    3    3  0 NaN   7.0  34.5   NaN  2.0  NaN
    4    4  0 NaN   8.0   8.0   7.0  0.0  NaN
    5    5  0 NaN   8.0   5.0   4.3  9.0 -5.5
    6    6  0 NaN   6.0   3.0   1.3  0.5 -0.1
    7    7  0 NaN   2.0   2.0   8.0  7.0  2.0
    8    8  0 NaN  15.3   5.0  12.0  5.0  3.0
    9    9  0 NaN   2.0   6.0  -3.0  4.3  4.0
    10  10  0 NaN   4.0   6.0   5.0  NaN  5.0
    """
    return pd.DataFrame(
        data={
            'a': np.array([
                1, 2, 3, 4, 5,
                6, 7, 8, 9, 10]),
            'b': np.array([
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0]),
            'c': np.array([
                np.nan, np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan, np.nan, np.nan]),
            'd': np.array([
                7.4, 5.2, 7, 8, 8,
                6, 2, 15.3, 2, 4]),
            'e': np.array([
                np.nan, -6.2, 34.5, 8, 5,
                3, 2, 5, 6, 6]),
            'f': np.array([
                np.nan, -3.0, np.nan, 7, 4.3,
                1.3, 8, 12, -3, 5]),
            'g': np.array([
                np.nan, 2, 2, 0, 9,
                0.5, 7, 5, 4.3, np.nan]),
            'h': np.array([
                5, np.nan, np.nan, np.nan, -5.5,
                -0.1, 2, 3, 4, 5])
        },
        index=list([x for x in range(1, 11)])
    )


def generate_example_df_hotdeck():
    """
    Example data frame with 6 rows for 4 columns. Prepared specifically for
    testing random hot deck imputation.

    Representation:

         a  b     c  d
    1  3.1  x  None  i
    2  NaN  x     a  j
    3  NaN  y     b  k
    4  5.7  y     c  l
    5  8.0  x     d  m
    6  1.2  y     e  n
    """
    return pd.DataFrame(
        data={
            'a': np.array([
                3.1, np.nan, np.nan, 5.7, 8.0, 1.2]),
            'b': np.array([
                'x', 'x', 'y', 'y', 'x', 'y']),
            'c': np.array([
                None, 'a', 'b', 'c', 'd', 'e']),
            'd': np.array([
                'i', 'j', 'k', 'l', 'm', 'n']),

        },
        index=list([x for x in range(1, 7)])
    )


def generate_example_series():
    return pd.Series(
        data=np.array([np.nan, 4, -3, np.nan, 24, np.nan]),
        index=list([x for x in range(1, 7)])
    )


def ts_airgap():
    """
    Timeseries describing monthly totals of international airline passengers
    (presumably in millions) and containing 13 missing values.

    Adapted from the imputeTS R package, with data originating from:
    George E. P. Box, Gwilym M. Jenkins, Gregory C. Reinsel, Greta M. Ljung
        (2015). Time Series Analysis: Forecasting and Control. Fifth Edition.
        (John Wiley & Sons)
    """
    return pd.Series(
        np.array([
            112, 118, 132, 129, np.nan, 135, 148, 148, np.nan, 119, 104, 118,
            115, 126, 141, 135, 125, 149, 170, 170, np.nan, 133, np.nan, 140,
            145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
            171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
            196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
            204, 188, 235, 227, 234, np.nan, 302, 293, 259, 229, 203, 229,
            242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
            284, 277, np.nan, np.nan, np.nan, 374, 413, 405, 355, 306, 271, 306,
            315, 301, 356, 348, 355, np.nan, 465, 467, 404, 347, np.nan, 336,
            340, 318, np.nan, 348, 363, 435, 491, 505, 404, 359, 310, 337,
            360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, np.nan,
            417, 391, 419, 461, np.nan, 535, 622, 606, 508, 461, 390, 432
        ]),
        index=pd.date_range(start='1949', end='1961', freq='M')
    )
