import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from numpy.linalg import solve
from scipy.sparse.linalg import inv
from scipy.sparse import csc_matrix
import time
from scipy.sparse import spdiags
from scipy.integrate import solve_ivp, RK45


#Defining constants
L = 10
dx = 0.1
dt = 0.001
t0 = 0
tf = 1.5
numTimeSteps =int(tf/dt) 
xCoords = np.arange(-L/2,L/2+dx, dx) #create the discrete x and y grids
numNodes = len(xCoords)
kappa = 0.04
#Peridynamics constants
horizon = 3.015
delta = horizon * dx
bVec = np.array([0,0,2])
diffOrder = 2
##################################################################################################
#PDDO
##################################################################################################
def findFamilyMembers():
    tree = KDTree(xCoords.reshape((numNodes,1)), leaf_size=2)
    familyMembers = tree.query_radius(xCoords.reshape((numNodes,1)), r = delta)
    return familyMembers

def calcXis(familyMembers):
    xis = []
    for iNode in range(numNodes):
        family = familyMembers[iNode]
        currentXis = []
        for iFamilyMember in range(len(family)):
            currentXis.append(xCoords[family[iFamilyMember]] - xCoords[iNode])
        xis.append(currentXis)
    return xis

def calcSysMatrix(familyMembers, xis):
    sysMatrix = np.zeros([numNodes,numNodes])
    #Differential Equation Part
    for iNode in range(numNodes):
        family = familyMembers[iNode]
        xi = xis[iNode]
        diffMat = np.zeros([3,3])
        for iFamilyMember in range(len(family)):
            currentFamilyMember = family[iFamilyMember]
            currentXi = xi[iFamilyMember]
            pList = np.array([1, currentXi, (currentXi)**2])
            weight = np.exp(-4*(np.absolute(currentXi))**2)
            diffMat += weight*np.outer(pList,pList)*dx
        for iFamilyMember in range(len(family)):
            currentFamilyMember = family[iFamilyMember]
            currentXi = xi[iFamilyMember]
            pList = np.array([1, currentXi, (currentXi)**2])
            weight = np.exp(-4*(np.absolute(currentXi))**2);
            sysMatrix[iNode][ currentFamilyMember] = kappa*weight*np.inner(solve(diffMat,bVec),pList)*dx

    #Boundary Condition 
    sysMatrix[0][0] = 1
    sysMatrix[0][1] = 0
    sysMatrix[0][2] = 0
    sysMatrix[0][3] = 0
    sysMatrix[numNodes-1][numNodes-1] = 1
    sysMatrix[numNodes-1][numNodes-2] = 0
    sysMatrix[numNodes-1][numNodes-3] = 0
    sysMatrix[numNodes-1][numNodes-4] = 0
    return sysMatrix

def calcInitialCondition():
    initialCondition = np.zeros(numNodes)
    for iNode in range(numNodes):
        initialCondition[iNode] = np.exp(-0.5*(xCoords[iNode])**2)
    return initialCondition

def adjustSysMatrixBoundaryNodes(sysMatrix):
    validValues = sysMatrix[3:10,6]
    sysMatrix[2:9,5] = validValues
    sysMatrix[1:8,4] = validValues
    sysMatrix[1:7,3] = validValues[1:7]
    sysMatrix[1:6,2] = validValues[2:7]
    sysMatrix[1:5,1] = validValues[3:7]
    sysMatrix[1:4,0] = validValues[4:7]
    
    sysMatrix[numNodes-9:numNodes-2,numNodes-6] = validValues
    sysMatrix[numNodes-8:numNodes-1,numNodes-5] = validValues
    sysMatrix[numNodes-7:numNodes-1,numNodes-4] = validValues[0:6]
    sysMatrix[numNodes-6:numNodes-1,numNodes-3] = validValues[0:5]
    sysMatrix[numNodes-5:numNodes-1,numNodes-2] = validValues[0:4]
    sysMatrix[numNodes-4:numNodes-1,numNodes-1] = validValues[0:3]
    return sysMatrix


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

    #Initial Condition
    initialCondition = calcInitialCondition()
    initialConditionPDDO = initialCondition
    initialConditionFD = initialCondition

    #PDDO
    familyMembers = findFamilyMembers()
    xis = calcXis(familyMembers)
    sysMatrix = calcSysMatrix(familyMembers, xis)
    sysMatrix = adjustSysMatrixBoundaryNodes(sysMatrix)

    identity = np.identity(numNodes)
    
    KInvPDDO = inv(csc_matrix(np.identity(numNodes)-np.multiply(dt,sysMatrix))).toarray()
    
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
