import numpy as np
from autots import AutoTS
import pandas as pd  # I had to use it somewhere... right...


###############################################################################
###############################################################################
###############################################################################
def create_dataframe(time_array : np.array,
                     data_dict  : dict) -> pd.DataFrame :
    '''
    Create a pandas DataFrame using a time array and a dictionary of arrays.

    Parameters
    ----------
    time_array : np.array
        DESCRIPTION. numpy array of datetime64[ns] objects to be used as the index.
    data_dict : dict
        DESCRIPTION. dictionary where keys are column names and values are numpy arrays for each column.

    Returns
    -------
    df : TYPE
        DESCRIPTION. DataFrame with datetime index and columns as specified in the dictionary.

    '''
    # Convert the numpy datetime array to a pandas DatetimeIndex
    datetime_index = pd.DatetimeIndex(time_array)

    # Create the DataFrame using the datetime index and the dictionary
    df = pd.DataFrame(data_dict, index=datetime_index)

    return df

###############################################################################
###############################################################################
###############################################################################
