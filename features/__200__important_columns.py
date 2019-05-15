import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from __000__data_road import road_raw_data

"""
pandas_profilingで表示した相関図で、
lifespanと関係ありそうなもののみ抽出する
"""

if __name__ == '__main__':
    df_train, df_test, target = road_raw_data()
    features = ['T24 Total temperature at LPC outlet ｰR',
                'T30 Total temperature at HPC outlet ｰR',
                'T50 Total temperature at LPT outlet ｰR',
                'Ps30 Static pressure at HPC outlet psia',
                'phi Ratio of fuel flow to Ps30 pps/psi',
                'BPR Bypass Ratio --',
                'htBleed (Bleed Enthalpy)',
                'W31 HPT coolant bleed lbm/s',
                'W32 LPT coolant bleed lbm/s',
                'EngineID',
                'Flight Regime',
                'EngineIndex']

    f_train = df_train[features]
    f_test = df_test[features]

    f_train.to_pickle('train__200__important_columns.pkl')
    f_test.to_pickle('test__200__important_columns.pkl')
