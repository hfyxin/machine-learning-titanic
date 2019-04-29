# sklearn pipeline for preprocessing

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

def load_data(csv_path):
    data_features = pd.read_csv(csv_path)
    data_labels = data_features['Survived']
    data_features.drop(columns=['Survived'], inplace=True)

    return data_features, data_labels


class FeatureSelector(BaseEstimator, TransformerMixin):
    '''
    Select features from input DataFrame
    '''
    def __init__(self, feature_names):
        self.feature_names = feature_names
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X:'DataFrame') -> 'np.array':
        return X[self.feature_names].values
        # same:  return X.loc[:, self.feature_names].values


def titanic_pipeline(
    cols_num=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'], # numerical
    cols_cat=['Embarked'],  # categorical
    cols_bin=['Sex']):
    '''
    define the preprocessing pipeline.
    '''

    pipe_num = Pipeline([
        ('feature_selector', FeatureSelector(cols_num)),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    pipe_cat = Pipeline([
        ('feature_selector', FeatureSelector(cols_cat)),
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('one_hot_encoder', OneHotEncoder()),    
    ])
    pipe_bin = Pipeline([
        ('feature_selector', FeatureSelector(cols_bin)),
        # No imputer
        ('binarizer', OrdinalEncoder()),
    ])
    pipe_full = FeatureUnion([
        ('pipe_num', pipe_num),
        ('pipe_cat', pipe_cat),
        ('pipe_bin', pipe_bin),
    ])
    return pipe_full