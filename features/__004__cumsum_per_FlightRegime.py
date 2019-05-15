import pandas as pd
import pickle
from tqdm import tqdm
from __000__data_road import road_raw_data


def cumsum_per_FlightRegime(df, features):
    feature_FlightRegime_cumsum = pd.DataFrame()
    for i, grp in tqdm(df.groupby('EngineID')):
        FR_c_1engine = pd.DataFrame()
        for flight_state in df['Flight Regime'].unique():
            df_1fs = grp.copy()
            df_1fs.loc[~(df_1fs['Flight Regime'] == flight_state), features] = 0
            df_1fs_cumsum = df_1fs[features].cumsum()
            df_1fs_cumsum = df_1fs_cumsum.rename(columns=lambda s: s + '_FlightRegime' + str(flight_state) + '_cumsum')
            FR_c_1engine = pd.concat([FR_c_1engine, df_1fs_cumsum], axis=1)

        feature_FlightRegime_cumsum = pd.concat([feature_FlightRegime_cumsum, FR_c_1engine])

    return feature_FlightRegime_cumsum


if __name__ == '__main__':
    df_train, df_test, target = road_raw_data()
    features = [c for c in df_train.columns if c not in ['Flight Regime', 'EngineID', 'EngineIndex']]

    f_train = cumsum_per_FlightRegime(df_train, features)
    f_test = cumsum_per_FlightRegime(df_test, features)

    f_train.to_pickle('train__004__cumsum_per_FlightRegime.pkl')
    f_test.to_pickle('test__004__cumsum_per_FlightRegime.pkl')
