#from sympy import *
import matplotlib.pyplot as plt
import numpy as np

#init_session()

#init_printing() 



#R = symbols('theta')
#A, L, I = symbols('A L I')


#T = Matrix((
#              [cos(R), sin(R),0,0,0,0],
#              [-sin(R), cos(R),0,0,0,0],
#              [0,0,1,0,0,0],
#              [0,0,0,cos(R),sin(R),0],
#              [0,0,0,-sin(R), cos(R),0],
#              [0,0,0,0,0,1]
#))
#K = Matrix((
#              [A*L**2/I, 0,0,-A*L**2/I,0,0],
#              [0,12,6*L,0,-12,6*L],
#              [0,6*L,4*L**2,0,-6*L,2*L**2],
#              [-A*L**2/I,0,0,A*L**2/I,0,0],
#              [0,-12,-6*L,0,12,-6*L],
#              [0,6*L,2*L**2,0,-6*L,4*L**2],
#))
#Result = transpose(T)*K*T
#Result = simplify(Result)
#pprint(Result)


#init_session()

#init_printing() 



#R = symbols('theta')
#A, L, I = symbols('A L I')


#T = Matrix((
#              [cos(R), sin(R),0,0,0,0],
#              [-sin(R), cos(R),0,0,0,0],
#              [0,0,1,0,0,0],
#              [0,0,0,cos(R),sin(R),0],
#              [0,0,0,-sin(R), cos(R),0],
#              [0,0,0,0,0,1]
#))
#K = Matrix((
#              [1, 0,0,-1,0,0],
#              [0,0,0,0,0,0],
#              [0,0,0,0,0,0],
#              [-1,0,0,1,0,0],
#              [0,0,0,0,0,0],
#              [0,0,0,0,0,0],
#))
#Result = transpose(T)*K*T
##Result = simplify(Result)
#pprint(Result, use_unicode=False, wrap_line = 1000)




######### HOW TO SCALE POLYGONS
#import matplotlib.pyplot as plt
#import matplotlib.patches as patches
#import matplotlib.transforms as transform
#import numpy as np
#from numpy import radians as rad

#fig, ax = plt.subplots()


#pin = plt.Polygon([[0, 0],[0.5, 1],[1, 0]],color='red',transform=ax.transData)
#ax.add_patch(pin)
#roller = patches.Circle([4, 6], radius=0.5, color='red',transform=ax.transData)
#ax.add_patch(roller)
#ax.scatter([-10, 10],[-50, 100])

## Need to add ax.transData second so that it scales first, then translates accordingly
#t1 =  transform.Affine2D().scale(10) + ax.transData
##t1 = transform.Affine2D().scale(30)
##t2 = transform.Affine2D().scale(1)
#roller.set(transform=t1)
#pin.set(transform=t1)
#ax.axis("equal")
#plt.show()


import numpy as np

arr = np.array([[1, 2],
                [3, 4]])
arr[1,:] = 0
print(arr)