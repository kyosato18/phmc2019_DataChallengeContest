import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from __000__data_road import road_raw_data


def ewa_per_FlightRegime(df, features):
    f_ewa_per_FlightRegime = pd.DataFrame()
    for fr in tqdm(np.sort(df['Flight Regime'].unique())):
        df_fr = df[df['Flight Regime'] == fr]
        for i, grp in tqdm(df_fr.groupby('EngineID')):
            ewa_1engine = pd.DataFrame()
            for c in features:
                ewa_1engine[c + '_ewa_3'] = grp[c].ewm(span=3, min_periods=1).mean()
                ewa_1engine[c + '_ewa_5'] = grp[c].ewm(span=5, min_periods=1).mean()
                ewa_1engine[c + '_ewa_10'] = grp[c].ewm(span=10, min_periods=1).mean()
                ewa_1engine[c + '_ewa_20'] = grp[c].ewm(span=20, min_periods=1).mean()
                ewa_1engine[c + '_ewa_30'] = grp[c].ewm(span=30, min_periods=1).mean()
                ewa_1engine[c + '_ewa_40'] = grp[c].ewm(span=40, min_periods=1).mean()
            f_ewa_per_FlightRegime = pd.concat([f_ewa_per_FlightRegime, ewa_1engine])

    return f_ewa_per_FlightRegime


if __name__ == '__main__':
    df_train, df_test, target = road_raw_data()
    features = [c for c in df_train.columns if c not in ['Flight Regime', 'EngineID', 'EngineIndex']]

    f_train = ewa_per_FlightRegime(df_train, features)
    f_test = ewa_per_FlightRegime(df_test, features)

    f_train.to_pickle('train__102__ewa_per_FlightRegime.pkl')
    f_test.to_pickle('test__102__ewa_per_FlightRegime.pkl')
