# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 09:50:24 2019

@author: NATE

2D gradient descent with the graph and path of descent plotted
"""



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython import get_ipython

#plt.close('all')
get_ipython().run_line_magic('matplotlib', 'qt') #put image in new window

cur_xy = np.array([float(x) for x in input('initial guess: ').split()])

alpha = 0.001 # step size factor
precision = 0.0000001
previous_step_size = 1 
max_iters = 100000 
iters = 0 

f = lambda x,y: x**2*y**2 +2*x*y
grad_fx = lambda x,y: 2*x*y**2 + 2*y
grad_fy = lambda x,y: 2*x**2*y + 2*x

fig = plt.figure() # create figure now to scatter plot the updated values
ax = fig.add_subplot(111, projection='3d') 

while previous_step_size > precision and iters < max_iters:
    prev_xy = cur_xy
    if (iters % 10)==0 or iters < 3:
        ax.scatter(cur_xy[0],cur_xy[1], f(cur_xy[0],cur_xy[1]), s=5, c ='r', marker = '^')
    cur_xy = cur_xy - alpha * np.array([grad_fx(cur_xy[0],cur_xy[1]), \
                                        grad_fy(cur_xy[0],cur_xy[1])])
    previous_step_size = np.linalg.norm(cur_xy-prev_xy)
    iters+=1


print("A local minimum occurs at", cur_xy)
print("the value at this point is: ", f(cur_xy[0],cur_xy[1]))

xy_vals = np.linspace(-3,3,50)
newx_vals = np.reshape([x for x in xy_vals for y in xy_vals], (50,50))
newy_vals = np.reshape([y for x in xy_vals for y in xy_vals], (50,50))

evals = np.reshape([f(x,y) for x in xy_vals for y in xy_vals], (50,50))


#put all this into one figure        
ax.plot_wireframe(newx_vals, newy_vals, evals)
ax.scatter(cur_xy[0],cur_xy[1], f(cur_xy[0],cur_xy[1]), s=10, c ='k', marker = '^') 


#print(previous_step_size)
#print(iters)
#print(grad_fx(cur_xy[0],cur_xy[1]),grad_fy(cur_xy[0],cur_xy[1]))
