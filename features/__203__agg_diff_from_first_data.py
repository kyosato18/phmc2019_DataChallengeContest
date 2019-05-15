import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from __000__data_road import road_raw_data


def agg_diff_from_first_first_some_data(df, features):
    f_agg_diff_from_first_some_data = pd.DataFrame()
    for i, grp in tqdm(df.groupby('EngineID')):
        diff_from_first_some_data_1engine = pd.DataFrame()
        for c in features:
            diff_from_first_some_data_1engine[c + '_rolling3std'] = grp[c].rolling(3).std().fillna(10)
            diff_from_first_some_data_1engine[c + '_rolling10max'] = grp[c].rolling(10).max().fillna(10)
        f_agg_diff_from_first_some_data = pd.concat([f_agg_diff_from_first_some_data,
                                                     diff_from_first_some_data_1engine])

    return f_agg_diff_from_first_some_data


if __name__ == '__main__':
    df_train = pd.read_pickle('train__200__important_columns.pkl')
    df_test = pd.read_pickle('test__200__important_columns.pkl')

    df_202_train = pd.read_pickle('train__202__diff_from_first_some_data.pkl')
    df_202_test = pd.read_pickle('test__202__diff_from_first_some_data.pkl')

    df_202_train = pd.concat([df_202_train, df_train[['Flight Regime', 'EngineID', 'EngineIndex']]], axis=1)
    df_202_test = pd.concat([df_202_test, df_test[['Flight Regime', 'EngineID', 'EngineIndex']]], axis=1)

    features = [c for c in df_202_train.columns if c not in ['Flight Regime', 'EngineID', 'EngineIndex']]

    f_train = agg_diff_from_first_first_some_data(df_202_train.sort_index(), features)
    f_test = agg_diff_from_first_first_some_data(df_202_test.sort_index(), features)

    f_train.to_pickle('train__203__agg_diff_from_first_some_data.pkl')
    f_test.to_pickle('test__203__agg_diff_from_first_some_data.pkl')
