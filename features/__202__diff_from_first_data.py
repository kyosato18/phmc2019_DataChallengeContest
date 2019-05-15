import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from __000__data_road import road_raw_data


def diff_from_first_first_some_data_per_FlightRegime(df, features):
    f_diff_from_first_some_data_per_FlightRegime = pd.DataFrame()
    for i, grp in tqdm(df.groupby('EngineID')):
        for fr in np.sort(df['Flight Regime'].unique()):
            grp_fr = grp[grp['Flight Regime'] == fr]
            diff_from_first_some_data_1engine = pd.DataFrame()
            for c in features:
                # mean_first_some_data = grp_fr[grp_fr['lifespan'] < 100][c].mean()
                mean_first_some_data = grp_fr.iloc[0:5][c].mean()
                diff_from_first_some_data_1engine[c + '_diff_from_first_5_data'] = grp_fr[c] - mean_first_some_data
            f_diff_from_first_some_data_per_FlightRegime = pd.concat([f_diff_from_first_some_data_per_FlightRegime,
                                                                      diff_from_first_some_data_1engine])

    return f_diff_from_first_some_data_per_FlightRegime


if __name__ == '__main__':
    df_train = pd.read_pickle('train__200__important_columns.pkl')
    df_test = pd.read_pickle('test__200__important_columns.pkl')

    df_rol_mean_train = pd.read_pickle('train__201__rolling_mean_per_FlightRegime.pkl')
    df_rol_mean_test = pd.read_pickle('test__201__rolling_mean_per_FlightRegime.pkl')

    df_rol_mean_train = pd.concat([df_rol_mean_train, df_train[['Flight Regime', 'EngineID', 'EngineIndex']]], axis=1)
    df_rol_mean_test = pd.concat([df_rol_mean_test, df_test[['Flight Regime', 'EngineID', 'EngineIndex']]], axis=1)

    features = [c for c in df_rol_mean_train.columns if c not in ['Flight Regime', 'EngineID', 'EngineIndex']]

    f_train = diff_from_first_first_some_data_per_FlightRegime(df_rol_mean_train, features)
    f_test = diff_from_first_first_some_data_per_FlightRegime(df_rol_mean_test, features)

    f_train.to_pickle('train__202__diff_from_first_some_data.pkl')
    f_test.to_pickle('test__202__diff_from_first_some_data.pkl')
