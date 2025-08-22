import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import helpers

# TODO: check data for awkward values that may be misinterpreted

def preprocess_census_data(df):
    """
    Preprocess the census data for machine learning analysis.
    Handles categorical variables and scales only numerical data.
    
    Args:
        df (pandas.DataFrame): Raw census data
        
    Returns:
        tuple: (df_scaled, label_encoders, scaler, preprocessing_info)
            - df_scaled: pandas DataFrame with scaled numerical columns and encoded categorical columns
            - label_encoders: dictionary of label encoders for categorical columns
            - scaler: fitted StandardScaler (only for numerical columns)
            - preprocessing_info: dictionary with preprocessing details
    """
    df_processed = df.copy()
    
    # Separate numerical and categorical columns
    numerical_cols = []
    categorical_cols = []
    
    for col in df.columns:
        if df[col].dtype == 'object':
            categorical_cols.append(col)
        else:
            numerical_cols.append(col)
    
    # Handle categorical variables by label encoding
    # TODO: explore other encoding methods (one hot?)
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le
    
    # Scale only numerical columns
    if numerical_cols:
        scaler = StandardScaler()
        numerical_scaled = scaler.fit_transform(df_processed[numerical_cols])
        
        # Create DataFrame with scaled numerical columns (renamed to indicate scaling)
        scaled_numerical_cols = [f"{col}_scaled" for col in numerical_cols]
        df_numerical_scaled = pd.DataFrame(numerical_scaled, 
                                         columns=scaled_numerical_cols, 
                                         index=df.index)
        
        # Combine scaled numerical columns with encoded categorical columns
        df_scaled = pd.concat([df_numerical_scaled, df_processed[categorical_cols]], axis=1)
    else:
        # No numerical columns to scale
        df_scaled = df_processed[categorical_cols]
        scaler = None
    
    # Create preprocessing info
    preprocessing_info = {
        'numerical_columns': numerical_cols,
        'categorical_columns': categorical_cols,
        'scaled_numerical_columns': scaled_numerical_cols if numerical_cols else [],
        'total_columns': len(df.columns),
        'original_shape': df.shape,
        'processed_shape': df_scaled.shape,
        'categorical_encodings': {col: list(le.classes_) for col, le in label_encoders.items()}
    }
    
    return df_scaled, label_encoders, scaler, preprocessing_info

def print_preprocessing_summary(preprocessing_info):
    """
    Print a summary of the preprocessing steps performed.
    
    Args:
        preprocessing_info (dict): Information about preprocessing steps
    """
    print("=" * 50)
    print("PREPROCESSING SUMMARY")
    print("=" * 50)
    print(f"Original dataset shape: {preprocessing_info['original_shape']}")
    print(f"Processed dataset shape: {preprocessing_info['processed_shape']}")
    print(f"Total columns: {preprocessing_info['total_columns']}")
    print(f"Numerical columns: {len(preprocessing_info['numerical_columns'])}")
    print(f"Categorical columns: {len(preprocessing_info['categorical_columns'])}")
    
    if preprocessing_info['numerical_columns']:
        print(f"\nNumerical columns scaled:")
        for col in preprocessing_info['scaled_numerical_columns']:
            print(f"  - {col}")
    
    if preprocessing_info['categorical_columns']:
        print(f"\nCategorical columns processed:")
        for col in preprocessing_info['categorical_columns']:
            unique_values = len(preprocessing_info['categorical_encodings'][col])
            print(f"  - {col}: {unique_values} unique values")
    
    print("=" * 50)

def get_feature_importance_scores(df_scaled, feature_names=None):
    """
    Calculate feature importance scores based on variance.
    
    Args:
        df_scaled (pandas.DataFrame): Scaled data DataFrame with mixed column types
        feature_names (list, optional): List of feature names. If None, uses DataFrame columns.
        
    Returns:
        dict: Dictionary mapping feature names to importance scores
    """
    # Use DataFrame columns if feature_names not provided
    if feature_names is None:
        feature_names = df_scaled.columns
    
    # Calculate variance for each feature
    variances = df_scaled.var()
    
    # Normalize to get importance scores
    importance_scores = variances / variances.sum()
    
    # Create feature importance dictionary
    feature_importance = dict(zip(feature_names, importance_scores))
    
    # Sort by importance
    feature_importance = dict(sorted(feature_importance.items(), 
                                   key=lambda x: x[1], reverse=True))
    
    return feature_importance

def get_column_type_info(df_scaled):
    """
    Get information about which columns are scaled numerical vs encoded categorical.
    
    Args:
        df_scaled (pandas.DataFrame): Processed DataFrame
        
    Returns:
        dict: Dictionary with column type information
    """
    scaled_cols = [col for col in df_scaled.columns if col.endswith('_scaled')]
    categorical_cols = [col for col in df_scaled.columns if not col.endswith('_scaled')]
    
    return {
        'scaled_numerical': scaled_cols,
        'encoded_categorical': categorical_cols,
        'total_processed': len(df_scaled.columns)
    }

if __name__ == "__main__":
    df = helpers.load_census_data()
    df_scaled, label_encoders, scaler, preprocessing_info = preprocess_census_data(df)
    print_preprocessing_summary(preprocessing_info)

    
    # Save the processed data
    df_scaled.to_csv('data/census-bureau-scaled.csv', index=False)
    print(f"\nProcessed data saved to 'census-bureau-scaled.csv'")
