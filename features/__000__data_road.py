import pandas as pd
import glob
from tqdm import tqdm


def road_raw_data():
    train_files = glob.glob('../input/Train Files/Train Files/*.csv')
    test_files = glob.glob('../input/Test Files/Test Files/*.csv')

    df_train = pd.DataFrame()
    for file_path in tqdm(train_files):
        df_tmp = pd.read_csv(file_path, encoding="shift-jis")
        df_tmp['EngineID'] = str(file_path)[33:-4]
        df_tmp['lifespan'] = df_tmp.index[::-1]
        df_tmp = df_tmp.drop('Unnamed: 25', axis=1)
        df_train = pd.concat([df_train, df_tmp])
    df_train = df_train.reset_index()
    df_train = df_train.rename(columns={'index': 'EngineIndex'})

    df_test = pd.DataFrame()
    for file_path in tqdm(test_files):
        df_tmp = pd.read_csv(file_path, encoding="shift-jis")
        df_tmp['EngineID'] = str(file_path)[31:-4]
        df_tmp = df_tmp.drop('Unnamed: 25', axis=1)
        df_test = pd.concat([df_test, df_tmp])
    df_test = df_test.reset_index()
    df_test = df_test.rename(columns={'index': 'EngineIndex'})

    target = df_train[['lifespan', 'EngineID', 'Flight Regime', 'EngineIndex']]
    del df_train['lifespan']

    return df_train, df_test, target


if __name__ == '__main__':
    df_train, df_test, target = road_raw_data()
    df_train.to_pickle('train__000__raw_data.pkl')
    df_test.to_pickle('test__000__raw_data.pkl')
    target.to_pickle('target__000__raw_data.pkl')

