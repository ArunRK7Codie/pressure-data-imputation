from utils import load_data, introduce_nan, save_inference
from imputer.models import mice_impute, knn_impute, gain_impute
from evaluate import evaluate


########################
#########CONFIG#########
########################
DATASET_PATH = './data/pressure-data.xlsx' #Dataset to Impute
SHEET_NAME = 'back'
MODEL = 'KNN' #Specify the Imputation Model to Impute
K = 3 #If the model is KNN, sepicify the K value
SAVE_PATH = './results/'
NUM_ROWS = 1000 #Num of Rows ie. Time Series

actual_df = load_data(data_path=DATASET_PATH, sheet_name=SHEET_NAME, n_rows=1000)
nan_df = introduce_nan(actual_df)
if MODEL=='KNN':
    imputed_df = knn_impute(df=nan_df, k=K)
elif MODEL=='MICE':
    imputed_df = mice_impute(df=nan_df)
elif MODEL=='GAIN':
    imputed_df = gain_impute(df=nan_df)

error_df = evaluate(actual_df=actual_df, imputed_df=imputed_df)

save_inference(actual_df, error_df, model=MODEL, path=SAVE_PATH)