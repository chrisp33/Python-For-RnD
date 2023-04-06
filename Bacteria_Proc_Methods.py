from scipy.signal import find_peaks as fp
from scipy.signal import savgol_filter
from peakutils import baseline
import pandas as pd

def SmoothData(df, **kwargs):
    """Applies a Savitzky-Golay filter to each column of an input Pandas DataFrame to smooth the data.

    Parameters:
    df (pandas.DataFrame): Input DataFrame with data to smooth.
    **kwargs: Optional keyword arguments to be passed to the Savitzky-Golay filter.

    Returns:
    pandas.DataFrame: DataFrame with smoothed values for each input column.

    Notes:
    This function creates a new DataFrame with the same index as the input DataFrame and iterates 
    over each column. For each column, it applies a Savitzky-Golay filter using the `savgol_filter`
    function from the SciPy library. The keyword arguments specified in the function call are passed on to the 
    `savgol_filter` function. The resulting smoothed values are stored in the new DataFrame with a column name that 
    includes '_ftrd' as a suffix.

    Examples:
    >>> data = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8], 'C': [10, 11, 12, 13]})
    >>> smoothed_data = SmoothData(data, window_length=3, polyorder=1)
    >>> print(smoothed_data)
          A_ftrd  B_ftrd  C_ftrd
    0  1.000000     5.0    10.0
    1  1.666667     6.0    11.0
    2  2.333333     7.0    12.0
    3  3.000000     8.0    13.0
    """
    new_df = pd.DataFrame(index=df.index)
    for i,vals in df.items():
        new_df[i+'_ftrd'] = savgol_filter(vals, **kwargs)
    return new_df

def BaselineCorrect(df):
    """Corrects the baseline of each column in an input Pandas DataFrame 
    by subtracting a polynomial fit of degree 3.

    Parameters:
    df (pandas.DataFrame): Input DataFrame with data to correct.

    Returns:
    pandas.DataFrame: DataFrame with baseline-corrected values for each input column.

    Notes:
    This function creates a new DataFrame with the same index as the input DataFrame 
    and iterates over each column. For each column, it calculates a baseline using a 
    polynomial fit of degree 3 and subtracts this baseline from the original values. 
    The baseline is calculated using the `baseline` function, which is assumed to take 
    a time series as input and return a polynomial fit of the specified degree.

    Examples:
    >>> data = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8], 'C': [10, 11, 12, 13]})
    >>> corrected_data = BaselineCorrect(data)
    >>> print(corrected_data)
             A    B    C
    0 -0.59375 -0.5 -0.5
    1  0.09375  0.0  0.0
    2  0.78125  0.5  0.5
    3  1.40625  1.0  1.0
    """
    new_df = pd.DataFrame(index = df.index)
    for i,vals in df.items():
        new_df[i+'_bc'] = vals - baseline(vals, deg = 3)
    return new_df

def GetPeaks(df, **kwargs):
    """Identifies peaks in each column of an input Pandas DataFrame and 
    returns a filtered DataFrame with rows that contain at least one peak.

    Parameters:
    df (pandas.DataFrame): Input DataFrame with time series data to analyze.
    **kwargs: Optional keyword arguments to be passed to the peak-finding function.

    Returns:
    pandas.DataFrame: Filtered DataFrame with only the rows that contain at least one peak.

    Notes:
    This function relies on the `fp` function, which is assumed to take a time series
    as input and return a tuple of two values: the indices of the peaks and any additional 
    metadata. The indices of the peaks are used to create a new DataFrame that indicates 
    which rows contain at least one peak.

    Examples:
    >>> data = pd.DataFrame({'A': [0, 1, 2, 3, 4], 'B': [5, 6, 7, 8, 9], 'C': [10, 11, 12, 13, 14]})
    >>> filtered_data = GetPeaks(data, threshold=0.5)
    >>> print(filtered_data)
           A  B   C
    1      1  6  11
    2      2  7  12
    3      3  8  13
    4      4  9  14
    """
    new_df = pd.DataFrame(index = df.index)
    for i,vals in df.items():
        peaks, _ = fp(vals, **kwargs)
        new_df[i+'_pks'] = new_df.index.isin(peaks+df.index.min())
        peak_filter = new_df.sum(axis =1)>=1
    return df.loc[peak_filter]

def GetLabels(wide_df):
    """Extracts labels from the index of an input Pandas DataFrame.

    Parameters:
    wide_df (pandas.DataFrame): Input DataFrame with index containing labels.

    Returns:
    pandas.Series: Series of labels extracted from the index of the input DataFrame.

    Notes:
    This function assumes that the index of the input DataFrame contains labels followed 
    by an underscore and a number, and extracts the labels by splitting the index on 
    this pattern using the `str.split` method. The resulting labels are returned as a
    pandas Series.

    Examples:
    >>> data = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8], 'C': [10, 11, 12, 13]},
                            index=['Label_1', 'Label_2', 'Label_3', 'Label_4'])
    >>> labels = GetLabels(data)
    >>> print(labels)
    0    Label
    1    Label
    2    Label
    3    Label
    dtype: object
    """
    labels = (wide_df.index.astype(str).str.split('_\d+', n = 1, regex = True, expand = True)
              .get_level_values(0))
    return labels
