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

import numpy as np
from math import exp, floor
from scipy.fftpack import dct
from sklearn.datasets import make_low_rank_matrix
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import fetch_20newsgroups_vectorized 
import pandas as pd
from utils import multivariate_t_rvs, make_covariance


class DataFactory:
    """
    Class to pull various types of data for the experiment scripts.

    Types of data:

    Synthetic:

    Real:

    """

    def __init__(self,n=None,d=None,effective_rank=None,tail_strength=None,random_seed=10):
        self.n = n
        self.d = d
        if self.n == None:
            self.n = 10000

        if self.d == None:
            self.d = 250
        self.effective_rank = effective_rank
        self.tail_strength = tail_strength
        self.rng = random_seed

    def gaussian_design(self, variance=None):
        A = np.random.standard_normal(size=(self.n, self.d))
        x = np.random.randn(self.d)
        x /= np.linalg.norm(x)
        if variance == None:
            variance = 1.0
        y = A@x + variance*np.random.randn(self.n)
        return A,y, x

    def shi_phillips_synthetic(self):
        '''
        Generates the low rank data from the shi-phillips paper.
        '''
        A = np.zeros((self.n,self.d),dtype=float)
        r = self.effective_rank
        for col in range(self.d):
            arg = (col)**2/ r**2
            std = exp(-arg) #exp()
            A[:,col] = np.random.normal(loc=0.0,scale=std**2,size=(self.n,))
        x_star = np.zeros(self.d)
        x_star[:r] = np.random.randn(r,)
        x_star /= np.linalg.norm(x_star,2)
        noise = np.random.normal(loc=0.0,scale=4.0,size=(self.n,))
        X = dct(A)
        y = X@x_star + noise
        return X, y, x_star

    def mahoney_synthetic(self, noise_std, seed=10):
        '''
        Generates the mahoney synthetic data from section 4 https://arxiv.org/pdf/1702.04837.pdf
        '''
        np.random.seed(seed)
        C = 2*(0.5**np.fromfunction(lambda i, j: abs(i-j), (self.d,self.d)))
        M = multivariate_t_rvs(np.zeros(self.d,dtype=float),S=C,df=2,n=self.n)
        U,_,_ = np.linalg.svd(M,full_matrices=False)
        s = 10.**np.linspace(0,-6,num=self.d)
        G = np.random.randn(self.d,self.d)
        q,_ = np.linalg.qr(G)
        Vt = q.T
        X = (U*s)@Vt
        w0 = np.ones(self.d)
        w0[int(floor(0.2*self.d)):int(floor(0.6*self.d))] *= 0.1
        noise = np.random.normal(loc=0.0,scale=noise_std**2)
        y = X@w0 + noise 
        return X, y, w0

    def chowdury_synthetic(self):
        '''
        Generates the synthetic data from https://github.com/jiaseny/ridge-regression/blob/master/ex-synthetic.py
        in the n >> d setting
        '''
        # Generate synthetic dataset
        n = self.n  # Number of rows
        d = self.d  # Number of columns
        s = 50  # rank(A)

        alpha = .05
        gamma = 5

        M = np.random.normal(size=(n, s))
        D = np.diag(1. - (np.arange(s) - 1.) / d)
        # Random column-orthonormal matrix
        Q, _ = np.linalg.qr(np.random.normal(size=(d, s)), mode='reduced')
        E = np.random.normal(size=(n, d))
        A = M.dot(D).dot(Q.T) + alpha * E  # Design matrix (n, d)

        x = np.random.normal(size=d)  # Target vector
        e = np.random.normal(size=n)  # Noise
        b = A.dot(x) + gamma * e  # Response vector
        return A,b,x



    def fetch_low_rank_matrix(self,effective_rank=None,tail_strength=None):
        """
        Generates synthetic data using sklearn make_low_rank with self.n,self.d but with 
        variable effective_rank and tail_strength if need be.
        """
        if effective_rank == None:
            eff_rank = self.effective_rank
        else:
            eff_rank = effective_rank
        
        if tail_strength == None:
            t_strength = self.tail_strength
        else:
            t_strength = tail_strength

        X = make_low_rank_matrix(n_samples=self.n, 
                                 n_features=self.d, 
                                 effective_rank=eff_rank, 
                                 tail_strength=t_strength,
                                 random_state=self.rng)
        return X

    def fetch_bayesian_ridge_data(self, n_samples=1000, n_features=250):
        """
        Generates the data from the bayesian ridge regression example 
        https://scikit-learn.org/stable/auto_examples/linear_model/plot_bayesian_ridge.html
        """
        X = np.random.randn(n_samples, n_features)
        lambda_ = 4.
        w = np.zeros(n_features)
        # Keep n_features // 10 features for the model.
        relevant_features = np.random.randint(0,n_features,n_features//10)
        for i in relevant_features:
            w[i] = np.random.normal(loc=0., scale=1./np.sqrt(lambda_))
        
        # Noise  with precision alpha = 50
        alpha_ = 50.
        noise = np.random.normal(loc=0., scale=1./np.sqrt(alpha_),size=n_samples)
        y = X@w + noise
        return X,y

    def fetch_low_rank_ridge(self,effective_rank=None,tail_strength=None):
        """
        Generates an instance of ridge regression with low rank and sparse
        truth vector
        """
        np.random.seed(self.rng)
        n_features = self.d
        n_samples = self.n
        X = self.fetch_low_rank_matrix(effective_rank,tail_strength)
        lambda_ = 4.
        w = np.zeros(n_features)
        # Keep n_features // 10 features for the model.
        relevant_features = np.random.randint(0,n_features,n_features//10)
        for i in relevant_features:
            w[i] = np.random.normal(loc=0., scale=1./np.sqrt(lambda_))
        
        # Noise  with precision alpha = 50
        alpha_ = 50.
        noise = np.random.normal(loc=0., scale=1./np.sqrt(alpha_),size=n_samples)
        y = X@w + noise
        return X,y,w


    # Real dataset loading facilities
    def fetch_superconductor(self):
        """
        Returns the superconductor regression dataset
        """
        _ = np.load('data/superconductor.npy')
        X,y = _[:,:-1], _[:,-1]
        return X, y

    def fetch_newsgroups(self,ncols=100):
        """
        Returns a 100 column sample of the newsgroups datasets.
        """
        news = fetch_20newsgroups_vectorized().data 
        [nn,dd] = news.shape
        cols = np.random.choice(dd,size=ncols,replace=False)
        #rows = np.random.choice(nn,size=nsamples,replace=False)
        return news[:,cols].toarray()

    def fetch_adult_dataset(self):
        """
        Gets the ``Adult'' dataset
        """
        df = pd.read_csv('data/adult.data',header=None)
        print(df.head())

    def fetch_forest_cover(self):
        """
        Fetches the UCI ForestCover dataset provided that it has been downloaded into 
        the directory.
        """
        _ = np.load('data/covertype.npy')
        X,y = _[:,:-1], _[:,-1]
        return X, y

    def fetch_year_predictions(self):
        """
        Fetches the UCI YearPredictions dataset provided that it has been downloaded into 
        the directory.
        """
        _ = np.load('data/yearpredictions.npy')
        X,y = _[:,1:], _[:,0] # target column is the first see https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD
        return X, y

    def fetch_w8a(self):
        """
        Fetches the LIBSVM W8A dataset https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#w8a
        """
        _ = np.load('data/w8a.npy')
        X,y = _[:,:-1], _[:,-1]
        return X, y

    def feature_expansion(self,data, n_extra_features=50,gamma=1.,):
        """
        Given input ``data``, expand the feature space by ``n_extra_features``
        """
        g = gamma
        rbf_feature = RBFSampler(gamma=g, random_state=self.rng,n_components=n_extra_features)
        X_features = rbf_feature.fit_transform(data)
        return X_features

    def polynomial_features(self,data,degree):
        """
        Given input data ``data`` expand into the polynomial feature space.
        """
        poly = PolynomialFeatures(degree=degree)
        return poly.fit_transform(data)

    
def main():
    import matplotlib.pyplot as plt 
    from math import floor
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    plt.style.use('ggplot')
    np.set_printoptions(precision=4)

    seed = 100
    n = 10000
    d = 500
    R_lr = floor(0.05*d + 0.5)
    R_hr = floor(0.5*d + 0.5)
    tail_strength = 0.05
    REAL_DATA = True
    ds = DataFactory(n=n,d=d,random_seed=seed)
    fig, ax = plt.subplots(dpi=100)
    lw = 2.0

    low_rank_X = ds.fetch_low_rank_matrix(effective_rank=R_lr,tail_strength=tail_strength)
    high_rank_X = ds.fetch_low_rank_matrix(effective_rank=R_hr,tail_strength=tail_strength)

    if REAL_DATA:
        real_data = ['SuperConductor', 'Newsgroups', 'Cover']
        for df in real_data:
            # Tidy this for general case
            if df == 'SuperConductor':
                X,_ = ds.fetch_superconductor()
            elif df == 'Newsgroups':
                X = ds.fetch_newsgroups()
            elif df == 'Cover':
                X,_ = ds.fetch_forest_cover()
                X = X[:10000,:]

            #X = StandardScaler().fit_transform(X)
            X /= np.linalg.norm(X,ord='fro')
            _,S,_ = np.linalg.svd(X,full_matrices=False)
            ax.plot(np.arange(len(S))/len(S),MinMaxScaler().fit_transform(S[:,None]),linewidth=lw,label=df)
    

    # Evaluate the singular value profile 
    Ul, Sl, Vtl = np.linalg.svd(low_rank_X,full_matrices=False)
    Uh, Sh, Vth = np.linalg.svd(high_rank_X,full_matrices=False)

    
    ax.plot(np.arange(d)/d,Sl,linewidth=lw,label='Low rank') # Normalise the x axis
    ax.plot(np.arange(d)/d,Sh,linewidth=lw,label='High rank') 
    ax.legend()
    #ax.set_xlim(-0.05)
    ax.set_xlabel('Normalised Index')
    ax.set_ylabel('Normalised singular value')
    plt.show()







if __name__ == '__main__':
    main()