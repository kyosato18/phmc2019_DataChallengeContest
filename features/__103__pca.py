import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from __000__data_road import road_raw_data
from sklearn.decomposition import PCA


"""
trainで作ったpca.componentにtestデータを食わせたいので、
この特徴量はdf_train, df_testを同時に計算する
移動平均か指数平均とってノイズとってからのほうがよかったかもしれないが、
とりあえずこれで。
"""


def pca_per_FlightRegime(df_train, df_test, features):
    f_pca_per_FlightRegime_train = pd.DataFrame()
    f_pca_per_FlightRegime_test = pd.DataFrame()
    for fr in tqdm(np.sort(df_train['Flight Regime'].unique())):
        df_train_fr = df_train[df_train['Flight Regime'] == fr]
        pca = PCA(n_components=2)
        pca.fit(df_train_fr[features])

        df_pca_train = pd.DataFrame({'PCA_X': pca.transform(df_train_fr[features])[:, 0],
                                     'PCA_Y': pca.transform(df_train_fr[features])[:, 1]},
                                    index=df_train_fr.index)

        df_test_fr = df_test[df_test['Flight Regime'] == fr]
        df_pca_test = pd.DataFrame({'PCA_X': pca.transform(df_test_fr[features])[:, 0],
                                    'PCA_Y': pca.transform(df_test_fr[features])[:, 1]},
                                   index=df_test_fr.index)

        f_pca_per_FlightRegime_train = pd.concat([f_pca_per_FlightRegime_train, df_pca_train])
        f_pca_per_FlightRegime_test = pd.concat([f_pca_per_FlightRegime_test, df_pca_test])

    return f_pca_per_FlightRegime_train, f_pca_per_FlightRegime_test


if __name__ == '__main__':
    df_train, df_test, _ = road_raw_data()
    important_features = ['T24 Total temperature at LPC outlet ｰR',
                          'T30 Total temperature at HPC outlet ｰR',
                          'T50 Total temperature at LPT outlet ｰR',
                          'Nc Physical core speed rpm',
                          'Ps30 Static pressure at HPC outlet psia',
                          'NRc Corrected core speed rpm',
                          'BPR Bypass Ratio --',
                          'farB Burner fuel-air ratio --',
                          'htBleed (Bleed Enthalpy)',
                          'W31 HPT coolant bleed lbm/s',
                          'W32 LPT coolant bleed lbm/s']

    f_train, f_test = pca_per_FlightRegime(df_train, df_test, important_features)

    f_train.to_pickle('train__103__pca_per_FlightRegime.pkl')
    f_test.to_pickle('test__103__pca_per_FlightRegime.pkl')
