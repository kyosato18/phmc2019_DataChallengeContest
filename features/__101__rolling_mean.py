import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from __000__data_road import road_raw_data


def rolling_mean_per_FlightRegime(df, features):
    f_rolling_mean_per_FlightRegime = pd.DataFrame()
    for fr in tqdm(np.sort(df['Flight Regime'].unique())):
        print('Flight Regime', fr)
        df_fr = df[df['Flight Regime'] == fr]
        for i, grp in tqdm(df_fr.groupby('EngineID')):
            rolling_mean_1engine = pd.DataFrame()
            for c in features:
                rolling_mean_1engine[c + '_rolling_mean_3'] = grp[c].rolling(3, min_periods=1).mean()
                rolling_mean_1engine[c + '_rolling_mean_5'] = grp[c].rolling(5, min_periods=1).mean()
                rolling_mean_1engine[c + '_rolling_mean_10'] = grp[c].rolling(10, min_periods=1).mean()
                rolling_mean_1engine[c + '_rolling_mean_20'] = grp[c].rolling(20, min_periods=1).mean()
                rolling_mean_1engine[c + '_rolling_mean_30'] = grp[c].rolling(30, min_periods=1).mean()
                rolling_mean_1engine[c + '_rolling_mean_40'] = grp[c].rolling(40, min_periods=1).mean()
            f_rolling_mean_per_FlightRegime = pd.concat([f_rolling_mean_per_FlightRegime, rolling_mean_1engine])

    return f_rolling_mean_per_FlightRegime


if __name__ == '__main__':
    df_train, df_test, target = road_raw_data()
    features = [c for c in df_train.columns if c not in ['Flight Regime', 'EngineID', 'EngineIndex']]

    f_train = rolling_mean_per_FlightRegime(df_train, features)
    f_test = rolling_mean_per_FlightRegime(df_test, features)

    f_train.to_pickle('train__101__rolling_mean_per_FlightRegime.pkl')
    f_test.to_pickle('test__101__rolling_mean_per_FlightRegime.pkl')
