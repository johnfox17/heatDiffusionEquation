import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import inv
from scipy.sparse import csc_matrix
from scipy.sparse import spdiags
from scipy.integrate import solve_ivp, RK45

import PDDOLeftDirichletRightNeumannBC as PDDO
import FDLeftDirichletRightNeumannBC as FD

#Defining constants
L = 10.0
#L = 10.2
dx = 0.1
dt = 0.01
t0 = 0
tf = 250 
numTimeSteps =int(tf/dt) 
xCoords = np.arange(-L/2,L/2+dx, dx) #create the discrete x and y grids
numNodes = len(xCoords)
kappa = 0.04
kappa2 = 0.1

def calcInitialCondition():
    initialCondition = np.zeros(numNodes)
    for iNode in range(numNodes):
        #initialCondition[iNode] = np.exp(-0.5*(xCoords[iNode])**2)
        initialCondition[iNode] = 0
    return initialCondition


def main():
    
    ##############################
    #Initial Condition
    ##############################
    initialCondition = calcInitialCondition()
    initialConditionPDDO = initialCondition
    initialConditionFD = initialCondition

    ##############################
    #PDDO Setup
    ##############################
    numNodes = len(xCoords)
    horizon = 15.015
    #horizon = 3.015
    delta = horizon * dx
    bVec = np.array([0,0,2])
    diffOrder = 2
    numBC = 1
    BC = np.array([0.01])
    diffOrderBC = np.array([1])
    bVecBC = np.array([0,1,0])
    nodesBC = np.array([numNodes-1])
    
    ###############################
    #Solving with PDDO
    ###############################
    pddo = PDDO.PDDO(numNodes, xCoords, dx, dt, kappa, kappa2, horizon, diffOrder, \
            bVec, numBC, BC, nodesBC, diffOrderBC, bVecBC)
    pddo.solve(tf, initialConditionFD)

    ###############################
    #Solving with FD
    ###############################
    fd = FD.FD(numNodes, dx, dt, kappa, kappa2, BC)
    fd.solve(tf, initialConditionFD)

    ###############################
    #Calculating absolute error
    ##############################
    absError = np.abs(np.subtract(fd.SOL_FD,pddo.SOL_PDDO))
    
    ###############################
    #Plotting
    ###############################
    figure, axis = plt.subplots(2,1)  
    axis[0].plot(xCoords[:numNodes-10], fd.SOL_FD[:numNodes-10], marker='o',label='FD')
    axis[0].plot(xCoords[:numNodes-10], pddo.SOL_PDDO[:numNodes-10], marker='*', label='PDDO')
    axis[0].legend()
    axis[0].grid()
    axis[0].set_xlabel('x-axis')
    axis[0].set_ylabel('Heat Magnitude')
    axis[0].set_title('Heat Diffusion Equation (1.5 sec)')
    
    axis[1].plot(xCoords[:numNodes-10],absError[:numNodes-10], marker='o')
    axis[1].grid()
    axis[1].set_xlabel('x-axis')
    axis[1].set_ylabel('Absolute Error')
    axis[1].set_title('Absolute Error FD vs PDDO (1.5 sec)')


    plt.show()


    #np.savetxt('/home/doctajfox/Documents/Thesis_Research/heatDiffusionEquation/data/sysMatrix.csv', sysMatrix, delimiter=",")
    #a = input('').split(" ")[0]
    print('Done')

if __name__ == "__main__":
    main()
