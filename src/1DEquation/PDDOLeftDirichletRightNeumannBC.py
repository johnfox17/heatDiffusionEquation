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
        self.BC = BC[0]
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
        numNodes = self.numNodes
        familyMembers = self.familyMembers
        dx = self.dx
        bVec = self.bVec

        sysMatrix = np.zeros([numNodes, numNodes])
         
        PDDO.calcXis(self)
        xis = self.xis
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
                weight = np.exp(-4*(np.absolute(currentXi))**2)
                sysMatrix[iNode][ currentFamilyMember] = kappa*weight*np.inner(solve(diffMat,bVec),pList)*dx
        self.sysMatrix = sysMatrix
    
    def enforceLeftBoundaryConditions(self):
        sysMatrix = self.sysMatrix
        sysMatrix[0,0] = 1
        self.sysMatrix = sysMatrix

    def enforceRightBoundaryConditions(self):
        familyMembers = self.familyMembers
        xis = self.xis
        dx = self.dx
        numBC = self.numBC
        nodesBC = self.nodesBC
        bVecBC = self.bVecBC
        sysMatrix = self.sysMatrix
        kappa2 = self.kappa2

        for iNodeBC in range(numBC):
            iNode = nodesBC[iNodeBC]
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
                weight = np.exp(-4*(np.absolute(currentXi))**2)
                sysMatrix[iNode][ currentFamilyMember] = kappa2*weight\
                        *np.inner(solve(diffMat,bVecBC),pList)*dx
        
        self.sysMatrix = sysMatrix


    def enforceBoundaryConditionsRHS(self, initialCondition):
        dt = self.dt
        BC = self.BC
        initialCondition[0] = 0 
        initialCondition[-1] = BC
        self.RHS = initialCondition
    
    def calcDuDt(self):
        sysMatrix = self.sysMatrix
        
        numNodes = self.numNodes
        dt = self.dt
        dx = self.dx
        kappa = self.kappa
        PDDO.calcSysMatrix(self)
        PDDO.enforceRightBoundaryConditions(self)
        sysMatrixAux = np.zeros([numNodes,numNodes])
        identity = np.identity(numNodes-2)
        sysMatrixAux[1:numNodes-1,:] = np.multiply(dt,sysMatrix[1:numNodes-1,:])
        sysMatrixAux[1:numNodes-1,1:numNodes-1] = identity - sysMatrixAux[1:numNodes-1,1:numNodes-1]
        sysMatrix[1:numNodes-1,:] = sysMatrixAux[1:numNodes-1,:]
        sysMatrix[1:numNodes-2,0] = -sysMatrixAux[1:numNodes-2,0]
        sysMatrix[1:numNodes-1:,numNodes-1] = -sysMatrixAux[1:numNodes-1:,numNodes-1]
        self.dudt = sysMatrix

    def solve(self, tf, initialCondition):
        numNodes = self.numNodes
        dt = self.dt
        BC = self.BC
        numTimeSteps =int(tf/dt)
        PDDO.findFamilyMembers(self)
        PDDO.calcSysMatrix(self)
        PDDO.enforceLeftBoundaryConditions(self)
        PDDO.enforceRightBoundaryConditions(self)
        PDDO.calcDuDt(self)
        PDDO.enforceBoundaryConditionsRHS(self, initialCondition)
        
        RHS = self.RHS
        invDUDT = inv(csc_matrix(self.dudt)).toarray()
        
        for iTimeStep in range(numTimeSteps):
            RHS = np.matmul(invDUDT, initialCondition)
            PDDO.enforceBoundaryConditionsRHS(self, RHS)
            if iTimeStep == numTimeSteps-1:
                SOL_PDDO = RHS
            initialCondition = self.RHS
        self.SOL_PDDO = SOL_PDDO
