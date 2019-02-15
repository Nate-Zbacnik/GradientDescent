# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 23:39:17 2019

@author: NATE

Simple 1D gradient descent on a sample polynomial.

"""

import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython
#plt.close('all')


cur_x = float(input('initial guess: ')) 
alpha = 0.01 # step size factor
precision = 0.0000001
previous_step_size = 1 
max_iters = 100000 
iters = 0 

f = lambda x: 2*x**4 - 5 * x**3 - 10  #function to minimize
df = lambda x: 8 * x**3 - 15 * x**2 #derivative

while previous_step_size > precision and iters < max_iters:
    prev_x = cur_x
    if (iters % 10)==0 or iters < 3:
        plt.plot(cur_x,f(cur_x), 'rx')
    cur_x -= alpha * df(prev_x)
    previous_step_size = abs(cur_x - prev_x)
    iters+=1


print("A local minimum occurs at", cur_x)
print("The value at this point is", f(cur_x))

#plotting the function
f_vect = np.vectorize(f)
x_vals = np.linspace(-2,3,200)
plt.plot(x_vals, f_vect(x_vals))
plt.plot(cur_x,f(cur_x), 'kx')
plt.xlabel('x'), plt.ylabel('y')