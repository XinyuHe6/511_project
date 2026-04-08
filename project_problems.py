# IOE 511/MATH 562, University of Michigan
# Code written by: Albert S. Berahas & Jiahao Shi

import numpy as np
import scipy.stats as stats
import scipy.sparse as sparse
import scipy.io
# Define all the functions and calculate their gradients and Hessians, those functions include:
# (1)(2)(3)(4) Quadractic function
# (5)(6) Quartic function
# (7)(8) Rosenbrock function 
# (9) Data fit
# (10)(11) Exponential

 
# Problem Number: 1
# Problem Name: quad_10_10
# Problem Description: A randomly generated convex quadratic function; the 
#                      random seed is set so that the results are 
#                      reproducable. Dimension n = 10; Condition number
#                      kappa = 10

# function that computes the function value of the quad_10_10 function

def quad_10_10_func(x):
    # set raondom seed
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(10,1))

    mat = scipy.io.loadmat('quad_10_10_Q.mat')
    Q = mat['Q']
    
    # compute function value
    return (1/2*x.T@Q@x + q.T@x)[0]

def quad_10_10_grad(x):
    # set raondom seed
    np.random.seed(12)
    # Generate random data
    q = np.random.normal(size=(10,1))
    mat = scipy.io.loadmat('quad_10_10_Q.mat')
    Q = mat['Q']
    
    return Q@x + q   
    

def quad_10_10_Hess(x):
    # set raondom seed
    np.random.seed(12)
    # Generate random data
    q = np.random.normal(size=(10,1))
    mat = scipy.io.loadmat('quad_10_10_Q.mat')
    Q = mat['Q']
    
    return Q

# Problem Number: 2
# Problem Name: quad_10_1000
# Problem Description: A randomly generated convex quadratic function; the 
#                      random seed is set so that the results are 
#                      reproducable. Dimension n = 10; Condition number
#                      kappa = 1000

# function that computes the function value of the quad_10_1000 function

 

def quad_10_1000_func(x):
    # set raondom seed
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(10,1))

    mat = scipy.io.loadmat('quad_10_1000_Q.mat')
    Q = mat['Q']
    
    # compute function value
    return (1/2*x.T@Q@x + q.T@x)[0]


# Problem Number: 3
# Problem Name: quad_1000_10
# Problem Description: A randomly generated convex quadratic function; the 
#                      random seed is set so that the results are 
#                      reproducable. Dimension n = 1000; Condition number
#                      kappa = 10

# function that computes the function value of the quad_1000_10 function

def quad_1000_10_func(x):
    # set raondom seed
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(1000,1))

    mat = scipy.io.loadmat('quad_1000_10_Q.mat')
    Q = mat['Q']
    
    # compute function value
    return (1/2*x.T@Q@x + q.T@x)[0]

# Problem Number: 4
# Problem Name: quad_1000_1000
# Problem Description: A randomly generated convex quadratic function; the 
#                      random seed is set so that the results are 
#                      reproducable. Dimension n = 1000; Condition number
#                      kappa = 1000

# function that computes the function value of the quad_10_10 function

def quad_1000_1000_func(x):
    # set raondom seed
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(1000,1))
    
    mat = scipy.io.loadmat('quad_1000_1000_Q.mat')
    Q = mat['Q']
    
    # compute function value
    return (1/2*x.T@Q@x + q.T@x)[0]




# Problem Number: 5
# Problem Name: quartic_1
# Problem Description: A quartic function. Dimension n = 4

# function that computes the function value of the quartic_1 function


def quartic_1_func(x):
    Q = np.array([[5,1,0,0.5],
     [1,4,0.5,0],
     [0,0.5,3,0],
     [0.5,0,0,2]])
    sigma = 1e-4
    
    return 1/2*(x.T @x) + sigma/4*(x.T@Q@x)**2

# Problem Number: 6
# Problem Name: quartic_2
# Problem Description: A quartic function. Dimension n = 4

# function that computes the function value of the quartic_2 function


def quartic_2_func(x):
    Q = np.array([[5,1,0,0.5],
     [1,4,0.5,0],
     [0,0.5,3,0],
     [0.5,0,0,2]])
    sigma = 1e4
    
    return 1/2*(x.T@x) + sigma/4*(x.T@Q@x)**2




# Problem Number: 12
# Problem Name: genhumps_5
# Problem Description: A multi-dimensional problem with a lot of humps.
#                      This problem is from the well-known CUTEr test set.

# function that computes the function value of the genhumps_5 function



def genhumps_5_func(x):
    f = 0
    for i in range(4):
        f = f + np.sin(2*x[i])**2*np.sin(2*x[i+1])**2 + 0.05*(x[i]**2 + x[i+1]**2)
    return f

# function that computes the gradient of the genhumps_5 function

def genhumps_5_grad(x):
    g = [4*np.sin(2*x[0])*np.cos(2*x[0])* np.sin(2*x[1])**2                  + 0.1*x[0],
         4*np.sin(2*x[1])*np.cos(2*x[1])*(np.sin(2*x[0])**2 + np.sin(2*x[2])**2) + 0.2*x[1],
         4*np.sin(2*x[2])*np.cos(2*x[2])*(np.sin(2*x[1])**2 + np.sin(2*x[3])**2) + 0.2*x[2],
         4*np.sin(2*x[3])*np.cos(2*x[3])*(np.sin(2*x[2])**2 + np.sin(2*x[4])**2) + 0.2*x[3],
         4*np.sin(2*x[4])*np.cos(2*x[4])* np.sin(2*x[3])**2                  + 0.1*x[4]]
    
    return np.array(g)

# function that computes the Hessian of the genhumps_5 function
def genhumps_5_Hess(x)
    H = np.zeros((5,5))
    H[0,0] =  8* np.sin(2*x[1])**2*(np.cos(2*x[0])**2 - np.sin(2*x[0])**2) + 0.1
    H[0,1] = 16* np.sin(2*x[0])*np.cos(2*x[0])*np.sin(2*x[1])*np.cos(2*x[1])
    H[1,1] =  8*(np.sin(2*x[0])**2 + np.sin(2*x[2])**2)*(np.cos(2*x[1])**2 - np.sin(2*x[1])**2) + 0.2
    H[1,2] = 16* np.sin(2*x[1])*np.cos(2*x[1])*np.sin(2*x[2])*np.cos(2*x[2])
    H[2,2] =  8*(np.sin(2*x[1])**2 + np.sin(2*x[3])**2)*(np.cos(2*x[2])**2 - np.sin(2*x[2])**2) + 0.2
    H[2,3] = 16* np.sin(2*x[2])*np.cos(2*x[2])*np.sin(2*x[3])*np.cos(2*x[3])
    H[3,3] =  8*(np.sin(2*x[2])**2 + np.sin(2*x[4])**2)*(np.cos(2*x[3])**2 - np.sin(2*x[3])**2) + 0.2
    H[3,4] = 16* np.sin(2*x[3])*np.cos(2*x[3])*np.sin(2*x[4])*np.cos(2*x[4])
    H[4,4] =  8* np.sin(2*x[3])**2*(np.cos(2*x[4])**2 - np.sin(2*x[4])**2) + 0.1
    H[1,0] = H[0,1]
    H[2,1] = H[1,2]
    H[3,2] = H[2,3]
    H[4,3] = H[3,4]
    return H
