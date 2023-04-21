import numpy as np
import matplotlib.pyplot as plt
#from numpy.linalg import solve
from scipy.sparse.linalg import inv
from scipy.sparse import csc_matrix
import time
from scipy.sparse import spdiags
from scipy.integrate import solve_ivp, RK45

import PDDO as PDDO

#Defining constants
L = 10
dx = 0.1
dt = 0.001
t0 = 0
tf = 2
numTimeSteps =int(tf/dt) 
xCoords = np.arange(-L/2,L/2+dx, dx) #create the discrete x and y grids
numNodes = len(xCoords)
kappa = 0.04

def calcInitialCondition():
    initialCondition = np.zeros(numNodes)
    for iNode in range(numNodes):
        initialCondition[iNode] = np.exp(-0.5*(xCoords[iNode])**2)
    return initialCondition


def createKMatrix():
    data0 = np.array([(numNodes-1)*[1]])
    data1 = np.array([(numNodes)*[-2]])
    data2 = np.array([(numNodes)*[1]])
    KFD = spdiags(data0, -1, numNodes, numNodes ).toarray()\
            +spdiags(data1, 0, numNodes, numNodes ).toarray()\
            +spdiags(data2, 1, numNodes, numNodes ).toarray()
    KFD[0][0] = 1 
    KFD[0][1] = 0
    KFD[numNodes-1][numNodes-1] = 1
    KFD[numNodes-1][numNodes-2] = 0
    return KFD

def calcDuDt(t, initialConditionFD):
    kMat = createKMatrix()
    dudt = np.multiply(kappa/dx**2,np.matmul(kMat, initialConditionFD))
    return dudt

def main():
    #Peridynamics constants
    numNodes = len(xCoords)
    horizon = 3.015
    delta = horizon * dx
    bVec = np.array([0,0,2])
    diffOrder = 2
    pddo = PDDO.PDDO(numNodes, horizon, bVec, diffOrder, dx)
    #Initial Condition
    initialCondition = calcInitialCondition()
    initialConditionPDDO = initialCondition
    initialConditionFD = initialCondition

    #PDDO
    pddo.findFamilyMembers(xCoords)
    pddo.calcXis(xCoords)
    pddo.calcSysMatrix(kappa)
    pddo.adjustSysMatrixBoundaryNodes()

    identity = np.identity(numNodes)
    
    KInvPDDO = inv(csc_matrix(np.identity(pddo.numNodes)-np.multiply(dt,pddo.sysMatrix))).toarray()
    
    for iTimeStep in range(numTimeSteps):
        initialConditionPDDO = np.matmul(KInvPDDO, initialConditionPDDO)
        if iTimeStep == numTimeSteps-1:
            SOL_PDDO = initialConditionPDDO
    
    SOL_FD = solve_ivp(calcDuDt, [t0,tf], initialConditionFD, RK45)
    #Calculate error between FD and PDDO
    RK_timeSteps = len(SOL_FD.t)
    absError = np.abs(np.subtract(SOL_FD.y[:,RK_timeSteps-1],SOL_PDDO))
    figure, axis = plt.subplots(2,1)  
    axis[0].plot(xCoords, initialCondition, label='Initial Condition')
    axis[0].plot(xCoords, SOL_FD.y[:,RK_timeSteps-1], marker='o',label='FD')
    axis[0].plot(xCoords, SOL_PDDO, marker='*', label='PDDO')
    axis[0].legend()
    axis[0].grid()
    axis[0].set_xlabel('x-axis')
    axis[0].set_ylabel('Heat Magnitude')
    axis[0].set_title('Heat Diffusion Equation (1.5 sec)')
    
    axis[1].plot(xCoords,absError, marker='o')
    axis[1].grid()
    axis[1].set_xlabel('x-axis')
    axis[1].set_ylabel('Absolute Error')
    axis[1].set_title('Absolute Error FD vs PDDO (1.5 sec)')



    plt.show()


    #a = input('').split(" ")[0]
    #np.savetxt('/home/doctajfox/Documents/Thesis_Research/heatDiffusionEquation/data/sysMatrix.csv', sysMatrix, delimiter=",")
    print('Done')

if __name__ == "__main__":
    main()
