import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import inv
from scipy.sparse import csc_matrix
from scipy.sparse import spdiags
from scipy.integrate import solve_ivp, RK45

import PDDOLeftDirichletRightNeumannBC as PDDO
import FDLeftDirichletRightNeumannBC as FD

#Defining constants
L = 10.2
dx = 0.1
dt = 0.01
t0 = 0
tf = 20
numTimeSteps =int(tf/dt) 
xCoords = np.arange(-L/2,L/2+dx, dx) #create the discrete x and y grids
numNodes = len(xCoords)
kappa = 0.04
#kappa = 0.1
kappa2 = 0.1

def calcInitialCondition():
    initialCondition = np.zeros(numNodes)
    for iNode in range(numNodes):
        #initialCondition[iNode] = np.exp(-0.5*(xCoords[iNode])**2)
        initialCondition[iNode] = 0
    return initialCondition


def main():
    ##############################
    #PDDO Setup
    ##############################
    numNodes = len(xCoords)
    horizon = 8.015
    delta = horizon * dx
    bVec = np.array([0,0,2])
    diffOrder = 2
    numBC = 1
    BC = np.array([-0.01])
    diffOrderBC = np.array([1])
    bVecBC = np.array([0,1,0])
    nodesBC = np.array([numNodes-1])
    
    
    pddo = PDDO.PDDO(numNodes, horizon, bVec, diffOrder, dx, nodesBC, numBC, BC, diffOrderBC, bVecBC)
    #PDDO
    pddo.findFamilyMembers(xCoords)
    pddo.calcXis(xCoords)
    pddo.calcSysMatrix(kappa, kappa2)
    #pddo.adjustSysMatrixBoundaryNodes()

    ##############################
    #Initial Condition
    ##############################
    initialCondition = calcInitialCondition()
    initialConditionPDDO = initialCondition
    initialConditionFD = initialCondition[1:numNodes-2]

    
    ###############################
    #Solving with PDDO
    ###############################
    identity = np.identity(numNodes-2)
    pddo.sysMatrix[1:numNodes-1,0:numNodes] = np.multiply(dt,pddo.sysMatrix[1:numNodes-1,0:numNodes])
    pddo.sysMatrix[1:numNodes-1,1:numNodes-1] = identity - pddo.sysMatrix[1:numNodes-1,1:numNodes-1]
    KInvPDDO = inv(csc_matrix(pddo.sysMatrix)).toarray()
    
    for iTimeStep in range(numTimeSteps):
        initialConditionPDDO[0] = 0.0
        initialConditionPDDO[-1] = BC[0]
        initialConditionPDDO = np.matmul(KInvPDDO, initialConditionPDDO)
        if iTimeStep == numTimeSteps-1:
            SOL_PDDO = initialConditionPDDO
    
    
    
    #Solving with Finite Difference
    fd = FD.FD(numNodes, dx)

    '''np.savetxt('/home/doctajfox/Documents/Thesis_Research/heatDiffusionEquation/data/sysMatrixNeumann.csv', pddo.sysMatrix, delimiter=",")
    figure, axis = plt.subplots()
    axis.plot(xCoords[1:numNodes-1], initialCondition[1:numNodes-1], label='Initial Condition')
    axis.plot(xCoords[1:numNodes-1], SOL_PDDO[1:numNodes-1], marker='*',label='PDDO')
    axis.grid()
    axis.legend()
    axis.set_title('Neumann BC')
    axis.set_xlabel('x-axis')
    axis.set_ylabel('Heat Magnitude')
    plt.show()'''

    ###############################
    #Solving with FD
    ###############################
    SOL_FD = solve_ivp(fd.calcDuDt, [t0,tf], initialConditionFD, RK45)
    
    ###############################
    #Calculating absolute error
    ##############################
    RK_timeSteps = len(SOL_FD.t)
    
    SOL_FD_tf = np.zeros(numNodes)
    SOL_FD_tf[1:numNodes-2] = SOL_FD.y[:,RK_timeSteps-1] 

    absError = np.abs(np.subtract(SOL_FD_tf,SOL_PDDO))
    
    ###############################
    #Plotting
    ###############################
    figure, axis = plt.subplots(2,1)  
    axis[0].plot(xCoords, initialCondition, label='Initial Condition')
    axis[0].plot(xCoords, SOL_FD_tf, marker='o',label='FD')
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
