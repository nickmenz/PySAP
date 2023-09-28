from sympy import *

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



R = symbols('theta')
A, L, I = symbols('A L I')


T = Matrix((
              [cos(R), sin(R),0,0,0,0],
              [-sin(R), cos(R),0,0,0,0],
              [0,0,1,0,0,0],
              [0,0,0,cos(R),sin(R),0],
              [0,0,0,-sin(R), cos(R),0],
              [0,0,0,0,0,1]
))
K = Matrix((
              [1, 0,0,-1,0,0],
              [0,0,0,0,0,0],
              [0,0,0,0,0,0],
              [-1,0,0,1,0,0],
              [0,0,0,0,0,0],
              [0,0,0,0,0,0],
))
Result = transpose(T)*K*T
#Result = simplify(Result)
pprint(Result, use_unicode=False, wrap_line = 1000)