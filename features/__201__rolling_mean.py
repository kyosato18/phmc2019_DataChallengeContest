import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from __000__data_road import road_raw_data


def rolling_mean_per_FlightRegime(df, features):
    f_rolling_mean_per_FlightRegime = pd.DataFrame()
    for i, grp in tqdm(df.groupby('EngineID')):
        for fr in np.sort(df['Flight Regime'].unique()):
            grp_fr = grp[grp['Flight Regime'] == fr]
            rolling_mean_1engine = pd.DataFrame()
            for c in features:
                rolling_mean_1engine[c + '_rolling_mean_3'] = grp_fr[c].rolling(3, min_periods=1).mean()
                rolling_mean_1engine[c + '_rolling_mean_5'] = grp_fr[c].rolling(5, min_periods=1).mean()
                rolling_mean_1engine[c + '_rolling_mean_10'] = grp_fr[c].rolling(10, min_periods=1).mean()
                rolling_mean_1engine[c + '_rolling_mean_20'] = grp_fr[c].rolling(20, min_periods=1).mean()
                rolling_mean_1engine[c + '_rolling_mean_30'] = grp_fr[c].rolling(30, min_periods=1).mean()
            f_rolling_mean_per_FlightRegime = pd.concat([f_rolling_mean_per_FlightRegime, rolling_mean_1engine])

    return f_rolling_mean_per_FlightRegime


if __name__ == '__main__':
    df_train = pd.read_pickle('train__200__important_columns.pkl')
    df_test = pd.read_pickle('test__200__important_columns.pkl')

    features = [c for c in df_train.columns if c not in ['Flight Regime', 'EngineID', 'EngineIndex']]

    f_train = rolling_mean_per_FlightRegime(df_train, features)
    f_test = rolling_mean_per_FlightRegime(df_test, features)

    f_train.to_pickle('train__201__rolling_mean_per_FlightRegime.pkl')
    f_test.to_pickle('test__201__rolling_mean_per_FlightRegime.pkl')
