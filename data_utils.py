#Loads the dataset and returns the DataFrame Object for further processing

#importing dependencies
import pandas as pd #Handles Python DataFrame (Special Data Structure Object)
import numpy as np #For mathematical operations


data_path = './data/pressure-data.xlsx'

def load_data(sheet_name, n_rows=1000)->pd.DataFrame():
    df = pd.read_excel(data_path, header=None, sheet_name=sheet_name)
    #the tabular data have labels like column names and co-ordinate positions 
    #which are not required for our imputation
    df = df.drop(labels=0, axis='columns')
    df = df.drop(labels=[0,1,2], axis='rows').reset_index()
    return df[:n_rows] #returns the dataframe object


def introduce_nan(df, nan_position=[3,7,24,18,29,42,47]) -> pd.DataFrame():
    nan_df = df.copy() #Deep Copy of DataFrame df
    nan_position = [3,7,24,18,29,42,47] #Specifies the Position whose values has to be NaN
    #Making the values in those position as NaN
    for i in nan_position:
        nan_df[i] = pd.DataFrame([np.NaN for _ in range(nan_df[i].size)])
    return nan_position #NaN introduced dataframe

def save_inference(imputed_df, error_df, model, path='./results/'):
    save_df = pd.concat([imputed_df, error_df], ignore_index=True, axis=1)
    save_df.to_excel(path+model+"imputed_inference.xlsx", index=False, header=None)

