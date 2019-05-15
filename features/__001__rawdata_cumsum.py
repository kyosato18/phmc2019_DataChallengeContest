import pandas as pd
import pickle
from tqdm import tqdm
from __000__data_road import road_raw_data


def raw_data_cumsum(df, features):
    feature_cumsum = pd.DataFrame()
    for i, grp in tqdm(df.groupby('EngineID')):
        cumsum_1engine = pd.DataFrame()
        for c in features:
            cumsum_1engine[c + '_cumsum'] = grp[c].cumsum()
        feature_cumsum = pd.concat([feature_cumsum, cumsum_1engine])

    return feature_cumsum


if __name__ == '__main__':
    df_train, df_test, target = road_raw_data()
    features = [c for c in df_train.columns if c not in ['Flight Regime', 'EngineID', 'EngineIndex']]

    f_train = raw_data_cumsum(df_train, features)
    f_test = raw_data_cumsum(df_test, features)

    f_train.to_pickle('train__001__rawdata_cumsum.pkl')
    f_test.to_pickle('test__001__rawdata_cumsum.pkl')
