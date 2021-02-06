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

from data_factory import DataFactory
from frequent_directions_aistats_ridge_regression import FDRidge
from random_projections_aistats_ridge_regression import RPRidge
from sklearn.linear_model import Ridge,LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.kernel_approximation import RBFSampler
from plot_config import fd_params, rfd_params, sjlt_rp_params, gauss_rp_params, gauss_hs_params, sjlt_hs_params 
import numpy as np
from math import floor
import json
import matplotlib.pyplot as plt
from utils import get_errors, get_x_opt
from timeit import default_timer as timer

def real_data_error_profile(data_name,sketch_size):
    '''
    Use polynomial feature map which generates {feature_size + degree \choose degree} new 
    features
    '''
    # * Experimental parameters
    n = 20000
    trials = 1
    iterations = 10
    ds = DataFactory(n=n)
    if data_name == 'CoverType':
        _X,_y = ds.fetch_forest_cover()
        feature_expansion = 'Polynomial'
        features = [3]
    elif data_name == 'w8a':
        _X,_y= ds.fetch_w8a()
        feature_expansion = 'RBF'
        features = [7500, 15000]
    elif data_name == 'CaliforniaHousing':
        feature_size = 18000
        _X,_y = fetch_california_housing(return_X_y=True)
        feature_expansion = 'Polynomial'
        features = [5,6,7,8]
    else:
        _X,_y= ds.fetch_year_predictions()
        feature_expansion = 'RBF'
        features = [7500, 15000]


    # * Results data structures
    exact_results = {
        'solve time' : { _ : np.zeros(trials,dtype=float) for _ in features}
    }
    fd_results =  { 
        'errors'          : { _ : np.zeros((iterations+1, trials),dtype=float) for _ in features},
        'build times'     : { _ : np.zeros(trials,dtype=float) for _ in features},
        'iteration times' : { _ : np.zeros(trials,dtype=float) for _ in features},
        'all_times'       : { _ : np.zeros((iterations+1, trials),dtype=float) for _ in features}
    }
    rfd_results = { 
        'errors'          : { _ : np.zeros((iterations+1, trials),dtype=float) for _ in features},
        'build times'     : { _ : np.zeros(trials,dtype=float) for _ in features},
        'iteration times' : { _ : np.zeros(trials,dtype=float) for _ in features},
        'all_times'       : { _ : np.zeros((iterations+1, trials),dtype=float) for _ in features}
    }
    rpg_results = { 
        'errors'          : { _ : np.zeros((iterations+1, trials),dtype=float) for _ in features},
        'build times'     : { _ : np.zeros(trials,dtype=float) for _ in features},
        'iteration times' : { _ : np.zeros(trials,dtype=float) for _ in features},
        'all_times'       : { _ : np.zeros((iterations+1, trials),dtype=float) for _ in features}
    }
    rps_results = { 
        'errors'          : { _ : np.zeros((iterations+1, trials),dtype=float) for _ in features},
        'build times'     : { _ : np.zeros(trials,dtype=float) for _ in features},
        'iteration times' : { _ : np.zeros(trials,dtype=float) for _ in features},
        'all_times'       : { _ : np.zeros((iterations+1, trials),dtype=float) for _ in features}
    }
    ihs_g_results=  { 
        'errors'          : { _ : np.zeros((iterations+1, trials),dtype=float) for _ in features},
        'build times'     : { _ : np.zeros(trials,dtype=float) for _ in features},
        'iteration times' : { _ : np.zeros(trials,dtype=float) for _ in features},
        'all_times'       : { _ : np.zeros((iterations+1, trials),dtype=float) for _ in features}
    }

    ihs_s_results =  { 
        'errors'          : { _ : np.zeros((iterations+1, trials),dtype=float) for _ in features},
        'build times'     : { _ : np.zeros(trials,dtype=float) for _ in features},
        'iteration times' : { _ : np.zeros(trials,dtype=float) for _ in features},
        'all_times'       : { _ : np.zeros((iterations+1, trials),dtype=float) for _ in features}
    }


    mean_iter_time_single = lambda a : np.mean(a['all_times'][1:] - a['sketch time'])
    mean_iter_time_multi = lambda a : np.mean(a['all_times'][1:] - a['sketch time']/iterations)

    for t in range(trials):
        print('*'*10, '\t TRIAL ', t, '\t', '*'*10)
        np.random.seed(t)
        sample = np.random.choice(_X.shape[0],size=n, replace=False)
        X_sample, y = _X[sample], _y[sample]
        
        for i, feature_hyper in enumerate(features):
            
            print('######### FEATURIZING #########')
            if feature_expansion == 'Polynomial':
                X_poly = PolynomialFeatures(degree=feature_hyper).fit_transform(X_sample)
                if X_poly.shape[1] > X_poly.shape[0]:
                    print(X_poly.shape)
                    nkeep = int(1.5*(X_poly.shape[0] - X_poly.shape[1]))
                    X_poly = X_poly[:,:nkeep]
                    print(X_poly.shape)
            else:
                X_poly = RBFSampler(gamma=0.0025, random_state=t,n_components=feature_hyper).fit_transform(X_sample)
            X = StandardScaler().fit_transform(X_poly)
            g = 2*(np.linalg.norm(X,ord='fro')**2)/sketch_size
            print('#'*10, f'\t GAMMA: G={g}, i={i} d={feature_hyper}\t', '#'*10)

            # # ! Optimal solution
            print('#'*60)
            print('Solving exactly: Data shape: ', X.shape)
            solve_start = timer()
            x_opt = get_x_opt(X,y,g)
            solve_time = timer() - solve_start
            exact_results['solve time'][feature_hyper][t] = solve_time


            # # ! FD Sketching 
            print('#'*10, '\t FREQUENT DIRECTIONS \t', '#'*10)
            fdr = FDRidge(fd_dim=sketch_size,gamma=g)
            _, all_x,fd_measured = fdr.fast_iterate(X,y,iterations)

            fd_results['errors'][feature_hyper][:,t]             =  get_errors(all_x,x_opt)
            fd_results['build times'][feature_hyper][t]          =  fd_measured['sketch time']
            fd_results['iteration times'][feature_hyper][t]      =  mean_iter_time_single(fd_measured)
            fd_results['all_times'][feature_hyper][:,t]          =  fd_measured['all_times']


            # # ! RFD Sketching 
            print('#'*10, '\t ROBUST FREQUENT DIRECTIONS \t', '#'*10)
            rfdr = FDRidge(fd_dim=sketch_size,fd_mode='RFD',gamma=g)
            _, rfd_all_x,rfd_measured = rfdr.fast_iterate(X,y,iterations)
            
            rfd_results['errors'][feature_hyper][:,t]             =  get_errors(rfd_all_x,x_opt)
            rfd_results['build times'][feature_hyper][t]          =  rfd_measured['sketch time']
            rfd_results['iteration times'][feature_hyper][t]      =  mean_iter_time_single(rfd_measured)
            rfd_results['all_times'][feature_hyper][:,t]          =  rfd_measured['all_times']

            # ! Single Random sketches
            print('#'*10, '\t GAUSS SINGLE \t', '#'*10)
            gauss_single = RPRidge(rp_dim=sketch_size,rp_mode='Gaussian',gamma=g)
            _, gauss_single_all_x, gauss_single_measured = gauss_single.iterate_single_timing(X,y)

            rpg_results['errors'][feature_hyper][:,t]             =  get_errors(rfd_all_x,x_opt)
            rpg_results['build times'][feature_hyper][t]          =  gauss_single_measured['sketch time']
            rpg_results['iteration times'][feature_hyper][t]      =  mean_iter_time_single(gauss_single_measured)
            rpg_results['all_times'][feature_hyper][:,t]          =  gauss_single_measured['all_times']

            print('#'*10, '\t SJLT SINGLE \t', '#'*10)
            sjlt_single = RPRidge(rp_dim=sketch_size,rp_mode='SJLT',gamma=g)
            _, sjlt_single_all_x, sjlt_single_measured = sjlt_single.iterate_single_timing(X,y)

            rps_results['errors'][feature_hyper][:,t]             =  get_errors(sjlt_single_all_x,x_opt)
            rps_results['build times'][feature_hyper][t]          =  sjlt_single_measured['sketch time']
            rps_results['iteration times'][feature_hyper][t]      =  mean_iter_time_single(sjlt_single_measured)
            rps_results['all_times'][feature_hyper][:,t]          =  sjlt_single_measured['all_times']

            # ! Multi Random sketches
            print('#'*10, '\t GAUSS IHS \t', '#'*10)
            ihs_gauss = RPRidge(rp_dim=sketch_size,rp_mode='Gaussian',gamma=g)
            _, ihs_gauss_all_x, ihs_gauss_measured = ihs_gauss.iterate_multiple_timing(X,y)

            ihs_g_results['errors'][feature_hyper][:,t]              =  get_errors(ihs_gauss_all_x,x_opt)
            ihs_g_results['build times'][feature_hyper][t]           =  ihs_gauss_measured['sketch time']
            ihs_g_results['iteration times'][feature_hyper][t]       =  mean_iter_time_multi(ihs_gauss_measured)
            ihs_g_results['all_times'][feature_hyper][:,t]           =  ihs_gauss_measured['all_times']

            print('#'*10, '\t SJLT IHS \t', '#'*10)
            ihs_sjlt = RPRidge(rp_dim=sketch_size,rp_mode='SJLT',gamma=g)
            _, ihs_sjlt_all_x, ihs_sjlt_measured = ihs_sjlt.iterate_multiple_timing(X,y)
             
            ihs_s_results['errors'][feature_hyper][:,t]              =  get_errors(ihs_sjlt_all_x,x_opt)
            ihs_s_results['build times'][feature_hyper][t]           =  ihs_sjlt_measured['sketch time']
            ihs_s_results['iteration times'][feature_hyper][t]       =  mean_iter_time_multi(ihs_sjlt_measured)
            ihs_s_results['all_times'][feature_hyper][:,t]           =  ihs_sjlt_measured['all_times']


    # ! Prepare and save the results in json format 
    results_file_name = 'results/real-iterations' + data_name + '.json'
    for d in [exact_results, fd_results, rfd_results, rpg_results, rps_results, ihs_g_results, ihs_s_results]:
        for k,v in d.items():
            for v_key, v_val in v.items():
                if type(v_val) == np.ndarray:
                    d[k][v_key] = v_val.tolist()
    all_results = {
        'Exact'     : exact_results,
        'FD'        : fd_results,
        'RFD'       : rfd_results,
        'Gauss'     : rpg_results,
        'SJLT'      : rps_results,
        'ihs:Gauss' : ihs_g_results,
        'ihs:SJLT'  : ihs_s_results
    }


    with open(results_file_name, 'w') as fp:
        json.dump(all_results, fp,sort_keys=True, indent=4)  


def main():
    datasets = ['CoverType']#,'', 'YearPredictions']['w8a']: #
    for d in datasets:
        real_data_error_profile(d,300) 


if __name__ =='__main__':
    main()