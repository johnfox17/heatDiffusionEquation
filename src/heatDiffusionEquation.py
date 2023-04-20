import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from numpy.linalg import solve
from scipy.sparse.linalg import inv
from scipy.sparse import csc_matrix

#Defining constants
L = 10
dx = 0.1
xCoords = np.arange(-L/2,L/2+dx, dx) #create the discrete x and y grids
numNodes = len(xCoords)
#Peridynamics constants
horizon = 3.015
delta = horizon * dx
bVec = np.array([0,0,2])
diffOrder = 2
dt = 0.001
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
            sysMatrix[iNode][ currentFamilyMember] = weight*np.inner(solve(diffMat,bVec),pList)*dx

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

def main():

    #Initial Condition
    initialCondition = calcInitialCondition()
    #PDDO
    familyMembers = findFamilyMembers()
    xis = calcXis(familyMembers)
    sysMatrix = calcSysMatrix(familyMembers, xis)
    sysMatrix = adjustSysMatrixBoundaryNodes(sysMatrix)

    identity = np.identity(numNodes)
    
    K_inv = inv(csc_matrix(np.identity(numNodes)-np.multiply(dt,sysMatrix))).toarray()
    
    figure, axis = plt.subplots()  
    for iTimeStep in range(100):
        axis.plot(xCoords, initialCondition)
        initialCondition = np.matmul(K_inv,initialCondition)  
    plt.show()
    a = input('').split(" ")[0]
    np.savetxt('/home/doctajfox/Documents/Thesis_Research/heatDiffusionEquation/data/sysMatrix.csv', sysMatrix, delimiter=",")
    print('Done')

if __name__ == "__main__":
    main()
