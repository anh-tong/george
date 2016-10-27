__author__ = 'anhth'


import numpy as np
import george
from george.kernels import ExpSquaredKernel
import time
import matplotlib.pyplot as plt

time_baseline = np.zeros([15,1], dtype= 'float64')
likelihood_baseline = np.zeros([15,1], dtype= 'float64')
time_holrd = np.zeros([15,1], dtype= 'float64')
likelihood_holrd = np.zeros([15,1], dtype='float64')

num_data = [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000, 11000, 12000, 13000, 14000, 15000]

# Generate some fake noisy data.
for i in range(15):
    x = num_data[i] * np.sort(np.random.rand(num_data[i]))
    yerr = 0.2 * np.ones_like(x)
    y = np.sin(x) + yerr * np.random.randn(len(x))
    kernel = ExpSquaredKernel(1.0)
    #baseline
    start = time.time()
    gp = george.GP(kernel)
    gp.compute(x, yerr)
    likelihood_baseline[i] = gp.lnlikelihood(y)
    time_baseline[i] = time.time() - start
    print("baseline ", i)

    #HOLRD
    start = time.time()
    gp = george.GP(kernel, solver=george.HODLRSolver)
    gp.compute(x, yerr)
    likelihood_holrd[i] = gp.lnlikelihood(y)
    time_holrd[i] = time.time() - start
    print("holrd ", i)

plt.figure()
a,  = plt.plot(num_data, time_baseline, color = 'red', marker = 'o', linestyle='-' , label = 'Cholesky')
a1,  = plt.plot(num_data, time_holrd, color = 'blue', marker = 'o', linestyle='-' , label ='New method')
plt.ylabel('Running times (s)')
plt.xlabel('Number of data points')
plt.legend(handles=[a, a1])


plt.figure()
b, = plt.plot(num_data, likelihood_baseline, color = 'red', marker = 'o', linestyle='-' , label = 'Cholesky')
b1, = plt.plot(num_data, likelihood_holrd, color = 'blue', marker = 'o', linestyle='-' , label ='New method')
plt.ylabel('Log likelihood')
plt.xlabel('Number of data points')
plt.legend(handles=[b,b1])

plt.show()

