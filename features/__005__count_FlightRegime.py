import pandas as pd
import pickle
from tqdm import tqdm
from __000__data_road import road_raw_data


def cumsum_per_FlightRegime(df, features):
    feature_FlightRegime = pd.DataFrame()
    for i, grp in tqdm(df.groupby('EngineID')):
        df_FlightRegime_one_hot = pd.get_dummies(grp['Flight Regime'])
        FR_1engine = pd.DataFrame()
        for flight_state in df_FlightRegime_one_hot.columns:
            FR_1engine['FlightRegime_' + str(flight_state) + '_count'] = df_FlightRegime_one_hot[flight_state].cumsum()
            FR_1engine['FlightRegime_' + str(flight_state) + '_ratio'] = df_FlightRegime_one_hot[flight_state].cumsum() \
                                                                         / (df_FlightRegime_one_hot.index + 1)

        feature_FlightRegime = pd.concat([feature_FlightRegime, FR_1engine], sort=False)

    feature_FlightRegime = feature_FlightRegime.fillna(0)
    return feature_FlightRegime


if __name__ == '__main__':
    df_train, df_test, target = road_raw_data()
    features = [c for c in df_train.columns if c not in ['Flight Regime', 'EngineID', 'EngineIndex']]

    f_train = cumsum_per_FlightRegime(df_train, features)
    f_test = cumsum_per_FlightRegime(df_test, features)

    f_train.to_pickle('train__005__count_FlightRegime.pkl')
    f_test.to_pickle('test__005__count_FlightRegime.pkl')
