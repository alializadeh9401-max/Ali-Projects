import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functions import *

###--- Import spatial data ---###

### Read in the data
df = pd.read_csv('data.csv')

### Generate the arrays needed from the dataframe
data = np.array(df["bicycle.theft"])
xi = np.array(df['xi'])
yi = np.array(df['yi'])
N = len(data)
coords = [(xi[i],yi[i]) for i in range(N)]

### Subsample the original data set
subsample_factor = 3
idx = subsample(N, subsample_factor, seed=42)
G = get_G(N,idx)
c = G @ data


###--- MCMC ---####

### Set MCMC parameters
l = 0.5
n = 10000
beta = 0.2

### Set the likelihood and target, for sampling p(u|c)
log_target = log_poisson_target
log_likelihood = log_poisson_likelihood


# TODO: Complete Spatial Data questions (e), (f).
K = GaussianKernel(coords,l)
poi_samp, poi_accrate = pcn(log_likelihood,np.random.multivariate_normal(np.zeros(N), K),c, K,G,n,beta,N)
c_pred = np.mean(np.exp(poi_samp),axis = 0)
Pred_error = np.mean((c_pred - data)**2)
print('mean prediction error (MSE) = ', Pred_error)
### Plotting examples
plot_2D(data, xi, yi, title='Bike Theft Data')                   # Plot bike theft count data
plot_2D(c_pred,xi, yi, title = 'Expected counts')
plot_2D(abs(data - c_pred), xi,yi, title = 'error field')
plot_2D(c, xi[idx], yi[idx], title='Subsampled Data')      # Plot subsampled data