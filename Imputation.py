from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np
import fancyimpute
def imputation(data_train):
    categorical = data_train.columns[data_train.dtypes == "object"]
    class DataFrameImputer(TransformerMixin):
        def __init__(self):
            """Impute missing categorical values.
            Columns of dtype object are imputed with the most frequent value 
            in column.
            """
        def fit(self, X, y=None):

            self.fill = pd.Series([X[c].value_counts().index[0]
                if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
                index=X.columns)
            return self
        def transform(self, X, y=None):
            return X.fillna(self.fill)
    data_train_categorical = DataFrameImputer().fit_transform(data_train[categorical])
    def imputate_continuous(data_train):
        '''  Imputation Continuous Variable 
        '''
        continuous = data_train.columns[data_train.dtypes != "object"]
        index_train = data_train.index
        X_train = data_train[continuous].as_matrix()
        try:
            X_train_fancy_mice = fancyimpute.MICE(verbose=0).complete(X_train)
            data_train_continuous = pd.DataFrame(X_train_fancy_mice,columns=continuous,index = index_train)
        except:
            data_train_continuous = pd.DataFrame(X_train,columns=continuous,index = index_train)
        return data_train_continuous
    
    data_train_continuous = imputate_continuous(data_train)
    data_train_imputation = pd.merge(data_train_categorical, data_train_continuous, left_index=True, right_index=True)
    return data_train_imputation