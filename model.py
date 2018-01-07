from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import processing
import pandas as pd
def predict(filename):
    dataset_imputation,hammer_price = processing.data_processing(filename)
    dataset_dummies = processing.get_alignment(dataset_imputation)
    loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
    loaded_model.predict(dataset_dummies)
    
    # Write results to csv
    predict_price = loaded_model.predict(dataset_dummies)
#     prob = loaded_model.predict_proba(dataset_dummies)[:,1]
    final_result = pd.DataFrame(predict_price)
    final_result.columns = ["Price"]
    header = ["Price"]
    final_result.to_csv("test_result.csv", columns = header, index = False)
    
    # Calculate the RMSE
    RMSE = np.sqrt(abs(mean_squared_error(predict_price, hammer_price)))
    
    return RMSE