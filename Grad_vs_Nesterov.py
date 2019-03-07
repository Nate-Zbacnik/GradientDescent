# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 21:53:15 2019

@author: NATE

Tests the speed of convergence for regular gradient descent vs Nesterov's method
"""




import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython import get_ipython

#plt.close('all')
get_ipython().run_line_magic('matplotlib', 'qt') #put image in new window

cur_xy = np.array([float(x) for x in input('initial guess(input x and y separated by a space. try 0.2 -.4): ' \
                   ).split()])
value = cur_xy
alpha = 0.01 # step size factor
precision = 0.0000001
prev_step_size = 1 
max_iters = 10000 
g_iters = 0  #keep track of regular gradient descent
n_iters = 0  #keep track of Nesterov gradient descent

f = lambda x,y:  x**4+ y**2 - x*y - 2*y - 10*x**2 
grad_fx = lambda x,y: 4*x**3 - y - 20*x
grad_fy = lambda x,y: 2*y - x - 2

fig = plt.figure() # create figure now to scatter plot the updated values
ax = fig.add_subplot(111, projection='3d') 

while prev_step_size > precision and g_iters < max_iters:
    prev_xy = cur_xy
    if (g_iters % 5)==0 or g_iters < 3:
        ax.scatter(cur_xy[0],cur_xy[1], f(cur_xy[0],cur_xy[1]), s=10, c ='r', marker = '^')
    cur_xy = cur_xy - alpha * np.array([grad_fx(cur_xy[0],cur_xy[1]), \
                                        grad_fy(cur_xy[0],cur_xy[1])])
    prev_step_size = np.linalg.norm(cur_xy-prev_xy)
    g_iters+=1

g_values = cur_xy #store the gradient values


#Nesterov Gradient Descent
cur_xy = value
prev_step_size = 1
cur_xy = cur_xy - alpha * np.array([grad_fx(cur_xy[0],cur_xy[1]), \
                                            grad_fy(cur_xy[0],cur_xy[1])]) #First step
lam = float(1)  
while prev_step_size > precision and n_iters < max_iters:
    prev_xy = cur_xy
    if (n_iters % 5)==0 or n_iters < 3:
        ax.scatter(cur_xy[0],cur_xy[1], f(cur_xy[0],cur_xy[1]), s=10, c ='k', marker = '<')
        
    lam_new = (1 + (1+4*lam)**(0.5))/2
    
    inter_xy = cur_xy - alpha * np.array([grad_fx(cur_xy[0],cur_xy[1]), \
                                            grad_fy(cur_xy[0],cur_xy[1])]) #Gradient step
    
    cur_xy = inter_xy + (lam-1)/lam_new*(inter_xy-cur_xy) #Nesterov step
    prev_step_size = np.linalg.norm(cur_xy-prev_xy)
    lam = lam_new
    n_iters+=1



print("A local minimum occurs at", cur_xy)
print("the value at this point is: ", f(cur_xy[0],cur_xy[1]))

xy_vals = np.linspace(-1,3,50)
newx_vals = np.reshape([x for x in xy_vals for y in xy_vals], (50,50))
newy_vals = np.reshape([y for x in xy_vals for y in xy_vals], (50,50))

evals = np.reshape([f(x,y) for x in xy_vals for y in xy_vals], (50,50))


#put all this into one figure        
ax.plot_wireframe(newx_vals, newy_vals, evals)
ax.scatter(cur_xy[0],cur_xy[1], f(cur_xy[0],cur_xy[1]), s=10, c ='k', marker = '^') 


print("Number of iterations for regular gradient descent", g_iters)
print('Number of iterations for Nesterov gradient descent', n_iters)
