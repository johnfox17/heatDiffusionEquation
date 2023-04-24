
import numpy as np
from sklearn.neighbors import KDTree
from numpy.linalg import solve

##################################################################################################
#PDDO
##################################################################################################
class PDDO:
    def __init__(self, numNodes, horizon, bVec, diffOrder, dx, nodesBC, numBC, BC, diffOrderBC, bVecBC):
        self.numNodes = numNodes
        self.horizon = horizon
        self.bVec = bVec
        self.diffOrder = diffOrder
        self.dx = dx
        self.nodesBC = nodesBC
        self.delta = horizon * dx
        self.numBC = numBC
        self.BC = BC
        self.diffOrderBC = diffOrderBC
        self.bVecBC = bVecBC

    def findFamilyMembers(self, xCoords):
        tree = KDTree(xCoords.reshape((self.numNodes,1)), leaf_size=2)
        familyMembers = tree.query_radius(xCoords.reshape((self.numNodes,1)), r = self.delta)
        self.familyMembers = familyMembers

    def calcXis(self, xCoords):
        xis = []
        for iNode in range(self.numNodes):
            family = self.familyMembers[iNode]
            currentXis = []
            for iFamilyMember in range(len(family)):
                currentXis.append(xCoords[family[iFamilyMember]] - xCoords[iNode])
            xis.append(currentXis)
        self.xis = xis


    def calcSysMatrix(self, kappa, kappa2):
        #sysMatrix = np.zeros([self.numNodes + self.numBC,self.numNodes + self.numBC])
        sysMatrix = np.zeros([self.numNodes,self.numNodes])
        

        #Left Boundary Condition Dirichlet 
        sysMatrix[0][0] = 1
        sysMatrix[0][1:] = 0
        #Differential Equation Part
        for iNode in range(1,self.numNodes-1):
            family = self.familyMembers[iNode]
            xi = self.xis[iNode]
            diffMat = np.zeros([3,3])
            for iFamilyMember in range(len(family)):
                currentFamilyMember = family[iFamilyMember]
                currentXi = xi[iFamilyMember]
                pList = np.array([1, currentXi, (currentXi)**2])
                weight = np.exp(-4*(np.absolute(currentXi))**2)
                diffMat += weight*np.outer(pList,pList)*self.dx
            for iFamilyMember in range(len(family)):
                currentFamilyMember = family[iFamilyMember]
                currentXi = xi[iFamilyMember]
                pList = np.array([1, currentXi, (currentXi)**2])
                weight = np.exp(-4*(np.absolute(currentXi))**2);
                sysMatrix[iNode][ currentFamilyMember] = kappa*weight*np.inner(solve(diffMat,self.bVec),pList)*self.dx

        #Boundary Condition
        #iNode = iNode -1
        for iNodeBC in range(self.numBC):
            iNode = iNode + 1
            family = self.familyMembers[self.nodesBC[iNodeBC]]
            xi = self.xis[self.nodesBC[iNodeBC]]
            diffMat = np.zeros([3,3])
            for iFamilyMember in range(len(family)):
                currentFamilyMember = family[iFamilyMember]
                currentXi = xi[iFamilyMember]
                pList = np.array([1, currentXi, currentXi**2])
                weight = np.exp(-4*(np.absolute(currentXi))**2)
                diffMat += weight*np.outer(pList,pList)*self.dx
            for iFamilyMember in range(len(family)):
                currentFamilyMember = family[iFamilyMember]
                currentXi = xi[iFamilyMember]
                pList = np.array([1, currentXi, currentXi**2])
                weight = np.exp(-4*(np.absolute(currentXi))**2);
                sysMatrix[iNode][ currentFamilyMember] = kappa2*weight*np.inner(solve(diffMat,self.bVecBC),pList)*self.dx
                #sysMatrix[ currentFamilyMember][iNode] = sysMatrix[iNode][ currentFamilyMember]
                #a = input('').split(" ")[0]
        #sysMatrix[self.numNodes-1][self.numNodes-1] = 1
        #sysMatrix[0][1:] = 0 
        self.sysMatrix = sysMatrix

    def adjustSysMatrixBoundaryNodes(self):
        validValues = self.sysMatrix[3:10,6]
        
        self.sysMatrix[2:9,5] = validValues
        self.sysMatrix[1:8,4] = validValues
        self.sysMatrix[1:7,3] = validValues[1:7]
        self.sysMatrix[1:6,2] = validValues[2:7]
        self.sysMatrix[1:5,1] = validValues[3:7]
        self.sysMatrix[1:4,0] = validValues[4:7]

        self.sysMatrix[self.numNodes-9:self.numNodes-2,self.numNodes-6] = validValues
        self.sysMatrix[self.numNodes-8:self.numNodes-1,self.numNodes-5] = validValues
        self.sysMatrix[self.numNodes-7:self.numNodes-1,self.numNodes-4] = validValues[0:6]
        self.sysMatrix[self.numNodes-6:self.numNodes-1,self.numNodes-3] = validValues[0:5]
        self.sysMatrix[self.numNodes-5:self.numNodes-1,self.numNodes-2] = validValues[0:4]
        self.sysMatrix[self.numNodes-4:self.numNodes-1,self.numNodes-1] = validValues[0:3]
