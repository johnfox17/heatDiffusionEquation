import numpy as np
from sklearn.neighbors import KDTree
from numpy.linalg import solve
from scipy.sparse.linalg import inv
from scipy.sparse import csc_matrix

##################################################################################################
#PDDO
##################################################################################################
class PDDO:
    def __init__(self, numNodes, xCoords, dx, dt, kappa, kappa2, horizon, diffOrder, \
            bVec, numBC, BC, nodesBC, diffOrderBC, bVecBC):
        self.numNodes = numNodes
        self.xCoords = xCoords
        self.dx = dx
        self.dt = dt
        self.kappa = kappa
        self.kappa2 = kappa2
        self.horizon = horizon
        self.diffOrder = diffOrder
        self.bVec = bVec
        self.numBC = numBC
        self.BC = BC
        self.nodesBC = nodesBC
        self.diffOrderBC = diffOrderBC
        self.bVecBC = bVecBC
        self.delta = horizon * dx

    def findFamilyMembers(self):
        xCoords = self.xCoords
        numNodes = self.numNodes
        delta = self.delta

        tree = KDTree(xCoords.reshape((numNodes,1)), leaf_size=2)
        familyMembers = tree.query_radius(xCoords.reshape((numNodes,1)), r = delta)
        self.familyMembers = familyMembers

    def calcXis(self):
        xCoords = self.xCoords
        numNodes = self.numNodes
        familyMembers = self.familyMembers
        
        xis = []
        for iNode in range(numNodes):
            family = familyMembers[iNode]
            currentXis = []
            for iFamilyMember in range(len(family)):
                currentXis.append(xCoords[family[iFamilyMember]] - xCoords[iNode])
            xis.append(currentXis)
        self.xis = xis


    def calcSysMatrix(self):
        kappa = self.kappa
        kappa2 = self.kappa2
        numNodes = self.numNodes
        familyMembers = self.familyMembers
        xis = self.xis
        dx = self.dx
        bVec = self.bVec
        numBC = self.numBC
        nodesBC = self.nodesBC
        bVecBC = self.bVecBC

        #sysMatrix = np.zeros([self.numNodes + self.numBC,self.numNodes + self.numBC])
        sysMatrix = np.zeros([numNodes, numNodes])
        

        #Left Boundary Condition Dirichlet 
        sysMatrix[0][0] = 1
        sysMatrix[0][1:] = 0
        #Differential Equation Part
        for iNode in range(1, numNodes-1):
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
        #iNode = iNode -1
        for iNodeBC in range(numBC):
            iNode = iNode + 1
            family = familyMembers[nodesBC[iNodeBC]]
            xi = xis[nodesBC[iNodeBC]]
            diffMat = np.zeros([3,3])
            for iFamilyMember in range(len(family)):
                currentFamilyMember = family[iFamilyMember]
                currentXi = xi[iFamilyMember]
                pList = np.array([1, currentXi, currentXi**2])
                weight = np.exp(-4*(np.absolute(currentXi))**2)
                diffMat += weight*np.outer(pList,pList)*dx
            for iFamilyMember in range(len(family)):
                currentFamilyMember = family[iFamilyMember]
                currentXi = xi[iFamilyMember]
                pList = np.array([1, currentXi, currentXi**2])
                weight = np.exp(-4*(np.absolute(currentXi))**2);
                sysMatrix[iNode][ currentFamilyMember] = kappa2*weight*np.inner(solve(diffMat,bVecBC),pList)*dx
        #sysMatrix[self.numNodes-1][self.numNodes-1] = 1
        #sysMatrix[0][1:] = 0 
        self.sysMatrix = sysMatrix
    
    def solve(self, tf, initialCondition):
        numNodes = self.numNodes
        dt = self.dt
        BC = self.BC
        numTimeSteps =int(tf/dt)

        PDDO.findFamilyMembers(self)
        PDDO.calcXis(self)
        PDDO.calcSysMatrix(self)
        sysMatrix = self.sysMatrix
        
        identity = np.identity(numNodes-2)
        sysMatrix[1:numNodes-1,0:numNodes] = np.multiply(dt,sysMatrix[1:numNodes-1,0:numNodes])
        sysMatrix[1:numNodes-1,1:numNodes-1] = identity - sysMatrix[1:numNodes-1,1:numNodes-1]
        KInvPDDO = inv(csc_matrix(sysMatrix)).toarray()
        for iTimeStep in range(numTimeSteps):
            initialCondition[0] = 0.0
            initialCondition[-1] = BC[0]
            initialCondition = np.matmul(KInvPDDO, initialCondition)
            if iTimeStep == numTimeSteps-1:
                SOL_PDDO = initialCondition

        #print('Here')
        #a = input('').split(" ")[0]
        #pddo.adjustSysMatrixBoundaryNodes()

        self.SOL_PDDO = SOL_PDDO
