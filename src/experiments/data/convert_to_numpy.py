# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Converts the datasets from various formats to numpy.
Data is read in and saved into [X,y] format with X standardised.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

datasets = {
    'superconductor' : {
        'path' : 'superconductor.csv',
        'url'  : 'https://archive.ics.uci.edu/ml/datasets/Superconductivty+Data',
        'target' : -1
        },

    'covertype' : {
        'path' : 'covtype.data',
        'url'  : 'https://archive.ics.uci.edu/ml/datasets/covertype',
        'target' : -1
        },

    'yearpredictions' : {
        'path' : 'YearPredictionMSD.txt',
        'url'  : 'https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD',
        'target' : 0
        },

    'w8a' : {
        'path' : 'w8a',
        'url'  : 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#w8a',
    }   

    } 



def main():
    from sklearn.datasets import load_svmlight_file
    for d in datasets:
        try:
            # read the numpy file -- Numpy file exists
            dat = np.load(d+'.npy')
            print(f'{d} data already in numpy format.')
        except:
            # Numpy file does not exists so load via pandas
            if d in ['w8a']:
                X,y =  load_svmlight_file('w8a')
                X = X.toarray()
                df_arr = np.c_[X,y]
            else:
                print(f'{d} : obtaining .npy from input format')
                data_path = datasets[d]['path']
                df = pd.read_csv(data_path, header=0)
                print(df.head())
                df_arr = df.to_numpy()

            # Save the data in format data.npy
            df_arr = StandardScaler().fit_transform(df_arr)
            print(df_arr.shape)
            fname = d+'.npy'
            np.save(fname, df_arr)

if __name__ == '__main__':
    main()