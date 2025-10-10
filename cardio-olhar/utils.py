
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

# Classes para pipeline

class DropFeatures(BaseEstimator,TransformerMixin):
    def __init__(self,feature_to_drop = ['index', 'id']):
        self.feature_to_drop = feature_to_drop
    def fit(self,df, y=None):
        return self
    def transform(self,df):
        if (set(self.feature_to_drop).issubset(df.columns)):
            df.drop(self.feature_to_drop,axis=1,inplace=True)
            return df
        else:
            print('Uma ou mais features não estão no DataFrame')
            return df

class OrdinalFeature(BaseEstimator,TransformerMixin):
    def __init__(self,ordinal_feature = ['Colesterol', 'Glicose']):
        self.ordinal_feature = ordinal_feature
    def fit(self,df, y=None):
        return self
    def transform(self,df):
        if (set(self.ordinal_feature ).issubset(df.columns)):
            ordinal_encoder = OrdinalEncoder()
            df[self.ordinal_feature] = ordinal_encoder.fit_transform(df[self.ordinal_feature])
            return df
        else:
            print('Grau_escolaridade não está no DataFrame')
            return df

class MinMaxWithFeatNames(BaseEstimator,TransformerMixin):
    def __init__(self,min_max_scaler_ft = ['Idade', 'Peso', 'Altura', 'PressaoArterialSistolica',
                                         'PressaoArterialDiastolica']):
        self.min_max_scaler_ft = min_max_scaler_ft
    def fit(self,df, y=None):
        return self
    def transform(self,df):
        if (set(self.min_max_scaler_ft).issubset(df.columns)):
            min_max_enc = MinMaxScaler()
            df[self.min_max_scaler_ft] = min_max_enc.fit_transform(df[self.min_max_scaler_ft])
            return df
        else:
            print('Uma ou mais features não estão no DataFrame')
            return df