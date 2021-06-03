"""
This get_datasets.py files has 3 functions, each generates a different toy-example dataset.
For each function:
Inputs:
N - number of datapoints to generate
num_comp - number of components. This will determine how many y's can x map to.
Returns: x, y
where x, y are both lists of N datapoints

Note: for now datasets are generated only in 1D
"""

import numpy as np
from scipy import spatial


# Dataset 1 - Localisation-task-specific custom dataset - suggested by Ignas Budvytis
def get_dataset__simplified_localisation_problem(N=1000, C=100, num_clusters=10, one_to_many=True):
    """ N - num datapoints
    C - num patterns 
    num_clusters - how many patterns should be located along the axis from which output is sampled. """

    # c - patterns. Uniformly distributed along axis between [-1, 1]
    c = np.arange(-1,1 + 1e-9, 2/C)
    epsilon = np.random.normal(0,0.1,size=N)

    # Generate the rule governing the dataset. Sample some output locations (t) and assigned them a pattern (c)
    if not one_to_many:
        assert num_clusters <= C, "In one-to-one mode, number of clusters has to be lower or equal to number of patterns"
    T_min, T_max = -10*num_clusters, 10*num_clusters
    t = np.random.uniform(low=T_min, high=T_max, size=num_clusters)
    t_patterns = np.random.choice(c, num_clusters, replace=one_to_many)

    # Generate dataset. Sample some output location (y) and assign it a pattern (t_pattern -> x) which is associated with closses (t)
    y = np.random.uniform(low=T_min, high=T_max, size=N)
    tree = spatial.cKDTree(np.reshape(t, (-1,1)))
    indices = tree.query(np.reshape(y, (-1,1)))[1]
    x = np.array(t_patterns[indices]) + epsilon
    
    # get info on which x point where generated by which cluster
    # cluster_assignment = np.empty(len(x))
    # for i in range(num_clusters):
    #     bool_arr = indices == i 
    #     output = np.where(bool_arr)[0]
    #     cluster_assignment[]
    cluster_assignment = indices

    indexes = np.arange(len(x))
    np.random.shuffle(indexes)
    x, y, cluster_assignment, epsilon  = x[indexes], y[indexes], cluster_assignment[indexes], epsilon[indexes]
    split = int(0.9*N)
    data = {'train': [x[:split], y[:split], cluster_assignment[:split], epsilon[:split]],
            'test': [x[split:], y[split:], cluster_assignment[split:], epsilon[split:]]}

    return data



# Dataset 2 - sinusoidal function where x's and y's were swaped making one-to-many mapping 
# this dataset has some underlying patern, yet is not trivial to solve with standard 
# regression or other models which support only one-to-one mapping
# the advantage of this model is that there is some continutiy between patterns. 
# i.e. the mapping from x to y is not random as in Dataset 1
def get_dataset__inverted_sinusoid(N=1000, num_sinusoids=2):

    n = int(N / num_sinusoids)  
    x = []
    y = []
    epsilon = np.random.normal(0,0.1,size=n*num_sinusoids)
    cluster_assignment = np.zeros(n*num_sinusoids, dtype=int)

    x_shifts = np.random.uniform(low=-15, high=15, size=num_sinusoids)
    y_shifts = np.random.uniform(low=-5, high=5, size=num_sinusoids)
    amplitudes = np.random.uniform(low=1, high=5, size=num_sinusoids)
    periods = np.random.uniform(low=0.5, high=2, size=num_sinusoids)

    for i, x_shift, y_shift, amp, period in zip(range(num_sinusoids), x_shifts, y_shifts, amplitudes, periods):
        y_tmp = np.random.normal(y_shift, 1, size=n)
        x_tmp = amp * np.sin(period * y_tmp) + x_shift + 0.5 * y_tmp + epsilon[n*i:n*(i+1)]
        x += list(x_tmp)
        y += list(y_tmp)
        cluster_assignment[n*i:n*(i+1)] = i
    
    indexes = np.arange(len(x))
    np.random.shuffle(indexes)
    x, y = np.array(x), np.array(y)
    x, y, cluster_assignment, epsilon  = x[indexes], y[indexes], cluster_assignment[indexes], epsilon[indexes]
    split = int(0.9*N)
    data = {'train': [x[:split], y[:split], cluster_assignment[:split], y_shifts],
            'test': [x[split:], y[split:], cluster_assignment[split:], y_shifts]}

    return data



# Dataset 3 - K linear functions. if K=1 it can be solved with simple regression.
# However if K > 1 then there could be one-to-many mapping
def get_dataset__many_linear_functions(N = 1000, num_components=3):
    n = int(N / num_components)  
    x = []
    y = []
    epsilon = np.random.normal(0,0.1,size=n*num_components)
    cluster_assignment = np.zeros(n*num_components, dtype=int)

    x_shifts = np.random.uniform(low=-20, high=20, size=num_components)
    y_shifts = np.random.uniform(low=-5, high=5, size=num_components)
    gradients = np.random.uniform(low=-5, high=5, size=num_components)
    lengths = np.random.uniform(low=1, high=5, size=num_components)

    for i, x_shift, y_shift, gradient, length in zip(range(num_components), x_shifts, y_shifts, gradients, lengths):
        x_tmp = length*np.linspace(-1, 1, n) + x_shift
        y_tmp = gradient * x_tmp + y_shift + epsilon[n*i:n*(i+1)]
        x += list(x_tmp)
        y += list(y_tmp)
        cluster_assignment[n*i:n*(i+1)] = i

    indexes = np.arange(len(x))
    np.random.shuffle(indexes)
    x, y = np.array(x), np.array(y)
    x, y, cluster_assignment, epsilon  = x[indexes], y[indexes], cluster_assignment[indexes], epsilon[indexes]
    split = int(0.9*N)
    data = {'train': [x[:split], y[:split], cluster_assignment[:split], epsilon[:split], y_shifts],
            'test': [x[split:], y[split:], cluster_assignment[split:], epsilon[split:], y_shifts]}

    return data