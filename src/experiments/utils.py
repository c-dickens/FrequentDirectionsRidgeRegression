import numpy as np

# Find where errors are > t
def greater_than_thld(arr, thld):
    '''
    Given an input array arr, find the first index where arr > thld.
    '''
    i_thld = np.where(arr>thld)[0][0]
    return i_thld

def linear_fit(x, a, b):
    """
    Wrapper function for scipy to fit a line of best fit to datapoints.
    """
    return a*x + b

    
def reciprocal_fit(x, a, b):
    """
    Wrapper function for scipy to fit a curve of a/x + b to datapoints.
    """
    return a/x + b

def gaussian_projection(data,sketch_size,random_seed=10):
    """
    evaluates the linear random projection SA with S sampled from a 
    standard Gaussian distribution with appropriate scaling.
    """
    rng = np.random.seed(random_seed)
    [n,d] = data.shape
    S = np.random.randn(sketch_size,n) / np.sqrt(sketch_size)
    return S@data

def sparse_projection(data,sketch_size,sparsity=1,random_seed=10):
    """
    Performs the sparse johnson lindenstrauss transform of Kane and Nelson
    """
    [n,d] = data.shape
    sketch = np.zeros((sketch_size,n),dtype=float)
    for i in range(n):
        nnz_loc = np.random.choice(sketch_size,size=sparsity,replace=False)
        nnz_sign = np.random.choice([-1,1],size=sparsity,replace=True)
        sketch[nnz_loc,i] = nnz_sign
    return (1./np.sqrt(sparsity))*sketch@data

def get_covariance_bound(X,k,sketch_dimension):
    '''
    Given input data X and rank k, evaluate tthe covariance error bound:
    ||X - Xk||_F^2 / (sketch_dimension - k)
    '''
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    Xk = U[:,:k]@(np.diag(S[:k])@Vt[:k,:])
    delta_k = np.linalg.norm(X- Xk,ord='fro')**2
    return delta_k / (sketch_dimension - k )

def get_covariance_bound_vary_sketch_size(X,k,sketch_sizes):
    '''
    Returns the range of covariance bounds as the sketch size is varied
    '''
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    Xk = U[:,:k]@(np.diag(S[:k])@Vt[:k,:])
    delta_k = np.linalg.norm(X- Xk,ord='fro')**2
    bounds = delta_k*np.ones_like(sketch_sizes,dtype=float)
    for i, sk_dim in enumerate(sketch_sizes):
        bounds[i] /= (sk_dim - k)
    return bounds

# Student's T random variable
def multivariate_t_rvs(m, S, df=np.inf, n=1):
    '''generate random variables of multivariate t distribution
    Parameters
    ----------
    m : array_like
        mean of random variable, length determines dimension of random variable
    S : array_like
        square array of covariance  matrix
    df : int or float
        degrees of freedom
    n : int
        number of observations, return random array will be (n, len(m))
    Returns
    -------
    rvs : ndarray, (n, len(m))
        each row is an independent draw of a multivariate t distributed
        random variable
    '''
    m = np.asarray(m)
    d = len(m)
    if df == np.inf:
        x = 1.
    else:
        x = np.random.chisquare(df, n)/df
    z = np.random.multivariate_normal(np.zeros(d),S,(n,))
    return m + z/np.sqrt(x)[:,None]   # same output format as random.multivariate_normal

def make_covariance(d):
    '''
    Makes the covariance matrix whose entries are 
    2*0.5^|i-j|
    '''
    dists = np.fromfunction(lambda i, j: abs(i-j), (d,d))
    return 2.*(0.5**dists)