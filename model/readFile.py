import os
import logging

import pandas as pd
import h5py
import threading
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)

DEP_FEATURE_NAME = 'surf_temp_sqerror'
HDF_LOCK = threading.Lock()
DATE_PATTERN = 'date%Y%m%d'
TIME_PATTERN = 'time%H%M'
KEY_PATTERN = '/{}/{}'.format(DATE_PATTERN, TIME_PATTERN)
STORE_NAME = 'metro_error_data_pro_swe2018.h5'

start_time = datetime(2018, 9, 29, 5)
end_time = datetime(2018, 9, 29, 6)

def define_target_features(df, dep_feature_name):
    if dep_feature_name == 'surf_temp_sqerror':
        df[dep_feature_name] = (df['surf_temp'] - df['stn_surf_temp']) ** 2

def store_path(store_name):
    cwd = os.getcwd()
    return os.path.join(cwd, '{}.h5'.format(store_name))

def read(key, store_name=STORE_NAME):
    key = key.strftime(KEY_PATTERN)
    path = store_name
    with HDF_LOCK:
        with pd.HDFStore(path, 'r') as store:
            if key in store:
                try:
                    df = store.get(key)
                except Exception as e:
                    log.warning(e)
                    df = None
                if df is None:
                    raise ValueError('No data stored as {} in {}'.format(key, store_path))
                else:
                    return df
            else:
                raise ValueError('No data stored as {} in {}'.format(key, store_path))

def gather_training_data(start_time, end_time):
    log.info('gathering training data from {} until {}'.format(start_time, end_time))
    ref_time = start_time
    dfs = []
    dfs.append(read(ref_time))
    # while ref_time <= end_time:
    #     dfs.append(read(ref_time))
    #     ref_time += timedelta(minutes=15)
    df = pd.concat(dfs)
#    print(df.memory_usage(index=False).sum())
    return dfs

def train_test_splitter(df):
    define_target_features(df, dep_feature_name=DEP_FEATURE_NAME)
    return model


df = gather_training_data(start_time, end_time)
y = df['station_id']          # Split off classifications
X = df.ix[:, 'road_cond':] # Split off features

test = df.concat()
print(test)
# [print(key) for key in df[0].keys()]
# print(gather_training_data(start_time, end_time))