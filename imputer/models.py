import impyute as impy
import numpy as np
import pandas as pd
from .gain import gain
from tqdm import tqdm


# Default GAIN Configuations
batch_size=64
hint_rate=0.9,
alpha=100
iterations=1000


def knn_impute(df,k) -> pd.DataFrame():
    print("Imputing with KNN")
    imputed_ls = list() # for storing imputed time series record
    for i in tqdm(range(df[3].size)): # Iterate through n time series values in df
        row_val = df.iloc[i,:] #selects one time series for imputation
        row_np = row_val.values.reshape(10, 5) # reshapes a one dimensional row of values into 2d(10x5) matrix for imputing 
        imputed_row = impy.fast_knn(row_np, k=k).reshape(1,50) #Implements KNN Imputation functions from impyute module
        imputed_ls.append(imputed_row.tolist()) 
    imputed_df = pd.DataFrame(imputed_ls[i][0] for i in range(df[3].size)) # turns into Dataframe object again
    return imputed_df.copy()

def mice_impute(df) -> pd.DataFrame():
    print("Imputing with MICE")
    imputed_ls = list() # for storing imputed time series record
    for i in tqdm(range(df[0].size)): # Iterate through n time series values in df
        row_val = df.iloc[i,:] #selects one time series for imputation
        row_np = row_val.values.reshape(10, 5) # reshapes a one dimensional row of values into 2d(10x5) matrix for imputing 
        imputed_row = impy.mice(row_np).reshape(1,50) #Implements MICE Imputation functions from impyute module
        imputed_ls.append(imputed_row.tolist()) 
    imputed_df = pd.DataFrame(imputed_ls[i][0] for i in range(df[3].size)) # turns into Dataframe object again
    return imputed_df.copy()

def gain_impute(df) -> pd.DataFrame():
    print("Imputing with GAIN")
    gain_parameters = {'batch_size': batch_size,
                     'hint_rate': hint_rate,
                     'alpha': alpha,
                     'iterations': iterations}
    imputed_df, _ = gain(data_x= df, gain_parameters=gain_parameters)
    return imputed_df