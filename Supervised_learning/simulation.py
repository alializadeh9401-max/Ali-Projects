import numpy as np
import matplotlib.pyplot as plt
from functions import *


###--- Data Generation ---###

### Inference grid defining {ui}i=1,Dx*Dy
Dx = 32
Dy = 32
N = Dx * Dy     # Total number of coordinates
points = [(x, y) for y in np.arange(Dx) for x in np.arange(Dy)]                # Indexes for the inference grid
coords = [(x, y) for y in np.linspace(0,1,Dy) for x in np.linspace(0,1,Dx)]    # Coordinates for the inference grid
xi, yi = np.array([c[0] for c in points]), np.array([c[1] for c in points])    # Get x, y index lists
x, y = np.array([c[0] for c in coords]), np.array([c[1] for c in coords])      # Get x, y coordinate lists

### Data grid defining {vi}i=1,N/subsample_factor - subsampled from inference grid
subsample_factor = 4
idx = subsample(N, subsample_factor)
M = len(idx)                                                                   # Total number of data points
z = np.random.randn(N, )
e = np.random.randn(M, )
### Generate K, the covariance of the Gaussian process, and sample from N(0,K) using a stable Cholesky decomposition
def get_uK(l,coords, N,z):
    K = GaussianKernel(coords, l)
    Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))
    u = Kc @ z
    return u,K

### Observation model: v = G(u) + e,   e~N(0,I)
l_true = 0.3
u,K = get_uK(l_true, coords, N,z)
G = get_G(N, idx)
v = G @ u + e
plot_3D(u,x,y)
plot_result(u,v,x,y,x[idx],y[idx])
Kc = np.linalg.cholesky(K + 1e-6*np.eye(N))
Kc_inv = np.linalg.inv(Kc)
K_inverse = np.transpose(Kc_inv) @ Kc_inv
###--- MCMC ---####

### Set MCMC parameters
n = 10000
beta = 0.2

### Set the likelihood and target, for sampling p(u|v)
log_target = log_continuous_target
log_likelihood = log_continuous_likelihood

### Sample from prior for MCMC initialisation


# TODO: Complete Simulation questions (a), (b).
#(a)

# l_array = np.linspace(0.1,2,10)
# print(l_array)
# for l in l_array:
#     u_test,K_test = get_uK(l, coords, N,z)
#     ### Plotting examples
#     plot_3D(u_test, x, y)               # Plot original u with data v
#     plt.show()
#(b)
u0 = np.random.multivariate_normal(np.zeros(N), K)
grw_samp,grw_accrate = grw(log_target,u0,v,K,G,n,beta,N)
pcn_samp, pcn_accrate = pcn(log_likelihood, u0, v, K, G, n, beta, N)
grw_samp_avg = np.mean(grw_samp,axis = 0)          #averaging 
pcn_samp_avg = np.mean(pcn_samp,axis = 0)            #averaging 
plot_result(grw_samp_avg,v, x, y, x[idx], y[idx])
plot_3D(abs(grw_samp_avg-u),x,y)
print("grw RMS:",np.mean((grw_samp_avg-u)**2)**0.5)
print("grw acceptance rate is:",grw_accrate)
plot_result(pcn_samp_avg,v, x, y, x[idx], y[idx])
plot_3D(abs(pcn_samp_avg-u),x,y)
print("pcn RMS:",np.mean((pcn_samp_avg-u)**2)**0.5)
print("pcn acceptance rate is:",pcn_accrate)
# beta_array = np.logspace(-3,0,10)
# grw_accratearr = []
# pcn_accratearr = []
# print(beta_array)
# for beta in beta_array:
#     grw_samp, grw_accrate = grw(log_target, u, v, K, G, n, beta,N)
#     grw_accratearr.append(grw_accrate)
#     pcn_samp, pcn_accrate = pcn(log_likelihood, u, v, K, G, n, beta, N)
#     pcn_accratearr.append(pcn_accrate)
# plt.plot(beta_array,grw_accratearr)
# plt.show()
# plt.plot(beta_array,pcn_accratearr)
# plt.show()
###--- Probit transform ---###
t = probit(v)       # Probit transform of data

# TODO: Complete Simulation questions (c), (d).
log_likelihood = log_probit_likelihood
samples, prob_accrate = pcn(log_likelihood, u0, v,K,G,n,0.2,N)
prob_t = predict_t(samples)
u_pred = np.mean(samples, axis = 0)
#hard assignments:
pred_class = prob_t >= 0.5
### Plotting examples
plot_2D(probit(u), xi, yi, title='Original Data')     # Plot true class assignments
plot_2D(prob_t, xi, yi, title='Predicted Data')     # Plot true class assignments
plot_2D(pred_class, xi, yi, title='Predicted Data')      # Plot Predicted Data
plot_2D(t, xi[idx], yi[idx], title='Probit Data')     # Plot data
#plot_2D(prob_t, xi[idx], yi[idx], title='Predicted Data')      # Plot Predicted Data
Pred_error = np.mean((pred_class - probit(u))**2)
print('mean prediction error (MSE) = ', Pred_error)
error_search = []
l_search = np.linspace(0.01,10,100)
for i in range(len(l_search)):
    l = l_search[i]
    u_search,K_search = get_uK(l,coords,N,z)
    samples, prob_accrate = pcn(log_likelihood,np.random.multivariate_normal(np.zeros(N), K_search) , v,K_search,G,n,0.2,N)
    prob_t = predict_t(samples)
    u_pred = np.mean(samples, axis = 0)
#hard assignments:
    pred_class = prob_t >= 0.5
    error = np.mean((pred_class - probit(u))**2)
    error_search.append(error)
l_optimal = l_search[np.argmin(error_search)]
print("optimal length scale = ", l_optimal)
plt.figure(figsize=(8, 5))
plt.plot(l_search, error_search, label="Prediction Error")
plt.axvline(x=0.3, color='r', linestyle='--', label="True Length Scale")
plt.xlabel("Length Scale (l)")
plt.ylabel("Mean Prediction Error")
plt.title("Length Scale Optimization")
plt.legend()
plt.grid(True)
plt.show()