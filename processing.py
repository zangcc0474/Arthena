import pandas as pd
import pickle
import Imputation as Im
def data_processing(filename):
    ''' Read the dataset and conduct the data processing
    '''
    dataset = pd.read_csv(filename,encoding="latin-1")
    
    dataset = dataset.drop(dataset[dataset.hammer_price.isnull()].index)
    
    # Modify the year_of_execution column
    dataset['year_of_execution'] = dataset.year_of_execution.str.extract(r"(\d{4})")
    dataset.ix[pd.to_numeric(dataset['year_of_execution'])>2016,'year_of_execution'] = 2016
    
    # Add two new columns which are auction_year and auction_month
    dataset['auction_year'] = dataset['auction_date'].str.split('-').str[0]
    dataset['auction_month'] = dataset['auction_date'].str.split('-').str[1]
    dataset.ix[dataset.auction_month>'06','auction_month'] = 'Fall'
    dataset.ix[dataset.auction_month<='06','auction_month'] = 'Spring'
    
    # Get the artist age by calculating the difference between artist_death_year and artist_birth_year
    dataset.ix[dataset.artist_death_year.isnull(),'artist_death_year'] = 2013
    dataset['artist_age'] = dataset.artist_death_year - dataset.artist_birth_year
    hammer_price = dataset.hammer_price
    
    # Drop the columns based on dropped columns in training dataset
    with open('drop_column', 'rb') as in_pickle:
        drop_column = pickle.load(in_pickle)
    dataset = dataset.drop(drop_column,axis=1)
    
    # Do the imputation
    dataset_imputation = Im.imputation(dataset)
    
    return dataset_imputation,hammer_price
def get_alignment(dataset_imputation):
    dataset_dummies = pd.get_dummies(dataset_imputation)
    with open('data_train_columns', 'rb') as in_pickle:
        data_train_columns = pickle.load(in_pickle)
    for column in data_train_columns:
        if (column not in dataset_dummies.columns):
            dataset_dummies[column] = pd.Series(0, index=dataset_dummies.index)
    for column in dataset_dummies.columns:
        if (column not in data_train_columns):
            dataset_dummies = dataset_dummies.drop(column,axis=1)
    return dataset_dummies