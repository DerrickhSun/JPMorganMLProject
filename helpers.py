import pandas as pd
import numpy as np

def load_census_data(filename='data/census-bureau.data'):
    """
    Loads census bureau data from census-bureau.data and census-bureau.columns files
    and returns a pandas DataFrame.
    
    Returns:
        pandas.DataFrame: DataFrame containing the census data with proper column names
    """
    # Read column names from the columns file
    with open(filename, 'r') as f:
        columns = [line.strip() for line in f.readlines()]
    
    # Read the data file
    # The data appears to be comma-separated with 42 columns
    df = pd.read_csv(filename, header=None, names=columns)
    
    return df

def get_census_data_info():
    """
    Returns basic information about the loaded census data.
    
    Returns:
        dict: Dictionary containing basic information about the dataset
    """
    df = load_census_data()
    
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'sample_data': df.head(3).to_dict('records')
    }
    
    return info
