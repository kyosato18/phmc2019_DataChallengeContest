import pandas as pd
import pickle
from tqdm import tqdm
from __000__data_road import road_raw_data


def rolling_mean(df, features):
    feature_rolling_mean = pd.DataFrame()
    for i, grp in tqdm(df.groupby('EngineID')):
        rolling_mean_1engine = pd.DataFrame()
        for c in features:
            rolling_mean_1engine[c + '_rolling_mean_3'] = grp[c].rolling(3, min_periods=1).mean()
            rolling_mean_1engine[c + '_rolling_mean_5'] = grp[c].rolling(5, min_periods=1).mean()
            rolling_mean_1engine[c + '_rolling_mean_10'] = grp[c].rolling(10, min_periods=1).mean()
            rolling_mean_1engine[c + '_rolling_mean_20'] = grp[c].rolling(20, min_periods=1).mean()
            rolling_mean_1engine[c + '_rolling_mean_50'] = grp[c].rolling(50, min_periods=1).mean()
            rolling_mean_1engine[c + '_rolling_mean_100'] = grp[c].rolling(100, min_periods=1).mean()
        feature_rolling_mean = pd.concat([feature_rolling_mean, rolling_mean_1engine])

    return feature_rolling_mean


if __name__ == '__main__':
    df_train, df_test, target = road_raw_data()
    features = [c for c in df_train.columns if c not in ['Flight Regime', 'EngineID', 'EngineIndex']]

    f_train = rolling_mean(df_train, features)
    f_test = rolling_mean(df_test, features)

    f_train.to_pickle('train__002__rolling_mean.pkl')
    f_test.to_pickle('test__002__rolling_mean.pkl')
