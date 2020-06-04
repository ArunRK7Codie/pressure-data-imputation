import pandas as pd
import numpy as np 
from sklearn.metrics import mean_squared_error as mse 
from sklearn.metrics import r2_score


def evaluate(actual_df, imputed_df, nan_positions=[3,7,24,18,29,42,47]) -> pd.DataFrame():
    mse_ls = list()
    r_sqr = list()
    for i in range(actual_df[0].size):
        acc_row = actual_df.iloc[i,:].tolist()
        imp_row = imputed_df.iloc[i,:].tolist()
        mse_val = mse(acc_row, imp_row)
        r2_val = r2_score(acc_row, imp_row)
        mse_ls.append(mse_val)
        r_sqr.append(r2_val)
    mse_series = pd.Series(mse_ls, name='MSE')
    r2_series = pd.Series(r_sqr, name='r2 Score')
    err_df = pd.concat([mse_series, r2_series], axis=1)
    return err_df
