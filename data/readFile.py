import os
import logging

import pandas as pd
import h5py
import threading
from datetime import datetime, timedelta

log = logging.getLogger(__name__)

HDF_LOCK = threading.Lock()
DATE_PATTERN = 'date%Y%m%d'
TIME_PATTERN = 'time%H%M'
KEY_PATTERN = '/{}/{}'.format(DATE_PATTERN, TIME_PATTERN)
STORE_NAME = 'error_data_athena_t_1_1_v_1_0.h5'

start_time = datetime(2018, 12, 7, 12, 45)
end_time = datetime(2018, 12, 20)

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
    while ref_time <= end_time:
        dfs.append(read(ref_time))
        ref_time += timedelta(minutes=15)
    df = pd.concat(dfs)
    log.info('Training data memory usage: {}Mb'.format(df.memory_usage().sum() / 10**6))
#    print(df.memory_usage(index=False).sum())
    return df

# print(read('/date20181207/time1345/meta/values_block_1/meta', 'error_data_athena_t_1_1_v_1_0.h5'))
print(gather_training_data(start_time, end_time))