from model.sseLogic import define_target_features

from sklearn.ensemble import RandomForestRegressor
import numpy as np

DEP_FEATURE_NAME = 'surf_temp_sqerror'

def train_model(df):
    define_target_features(df, dep_feature_name=DEP_FEATURE_NAME)
    model = train_feature_model(df=df, dep_feature_name=DEP_FEATURE_NAME)
    return model

def train_feature_model(df, dep_feature_name, ind_feature_names=ST.INDEPENDENT_FEATURES, params=RFR_PARAMS):
    included = np.array(df[ind_feature_names + [dep_feature_name]].notnull().all(axis=1))
    rfr = RandomForestRegressor(**params)
    rfr.fit(df.loc[included, ind_feature_names].values.astype(np.float),
            df.loc[included, dep_feature_name].values.astype(np.float))
    return rfr

