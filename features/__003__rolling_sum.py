import pandas as pd
import pickle
from tqdm import tqdm
from __000__data_road import road_raw_data


def rolling_sum(df, features):
    feature_rolling_sum = pd.DataFrame()
    for i, grp in tqdm(df.groupby('EngineID')):
        rolling_sum_1engine = pd.DataFrame()
        for c in features:
            rolling_sum_1engine[c + '_rolling_sum_3'] = grp[c].rolling(3, min_periods=1).sum()
            rolling_sum_1engine[c + '_rolling_sum_5'] = grp[c].rolling(5, min_periods=1).sum()
            rolling_sum_1engine[c + '_rolling_sum_10'] = grp[c].rolling(10, min_periods=1).sum()
            rolling_sum_1engine[c + '_rolling_sum_20'] = grp[c].rolling(20, min_periods=1).sum()
            rolling_sum_1engine[c + '_rolling_sum_50'] = grp[c].rolling(50, min_periods=1).sum()
            rolling_sum_1engine[c + '_rolling_sum_100'] = grp[c].rolling(100, min_periods=1).sum()
        feature_rolling_sum = pd.concat([feature_rolling_sum, rolling_sum_1engine])

    return feature_rolling_sum


if __name__ == '__main__':
    df_train, df_test, target = road_raw_data()
    features = [c for c in df_train.columns if c not in ['Flight Regime', 'EngineID', 'EngineIndex']]

    f_train = rolling_sum(df_train, features)
    f_test = rolling_sum(df_test, features)

    f_train.to_pickle('train__003__rolling_sum.pkl')
    f_test.to_pickle('test__003__rolling_sum.pkl')
