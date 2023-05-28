import numpy as np
from sklearn.neighbors import KDTree

##################################################################################################
#PDDO
##################################################################################################
class PDDO:
    def __init__(self, numNodes, coords, dx, dy, dt, horizon, diffOrder, \
            bVec, numBC, BC, nodesBC, diffOrderBC, bVecBC):
        self.numNodes = numNodes
        self.coords = coords
        self.dx = dx
        self.dy = dy
        self.dt = dt
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
        coords = self.coords
        numNodes = self.numNodes
        delta = self.delta
        tree = KDTree(coords, leaf_size=2)
        familyMembers = tree.query_radius(coords, r = delta)
        self.familyMembers = familyMembers

    def calcXis(self):
        coords = self.coords
        numNodes = self.numNodes
        familyMembers = self.familyMembers

        xXis = []
        yXis = []
        for iNode in range(numNodes):
            family = familyMembers[iNode]
            currentXXis = []
            currentYXis = []
            for iFamilyMember in range(len(family)):
                currentXXis.append(coords[family[iFamilyMember]][0] - coords[iNode][0])
                currentYXis.append(coords[family[iFamilyMember]][1] - coords[iNode][1])
            xXis.append(currentXXis)
            yXis.append(currentYXis)
        self.xXis = xXis
        self.yXis = yXis


    def calcSysMatrix(self):
        numNodes = self.numNodes
        familyMembers = self.familyMembers
        dx = self.dx
        bVec = self.bVec
        sysMatrix = np.zeros([numNodes, numNodes])
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

    def solve(self, tf, initialCondition):
        numNodes = self.numNodes
        dt = self.dt
        BC = self.BC
        numTimeSteps =int(tf/dt)
        PDDO.findFamilyMembers(self)
        PDDO.calcXis(self)
        #PDDO.calcSysMatrix(self)
        #a = input('').split(" ")[0]
        '''PDDO.calcSysMatrix(self)
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
        self.SOL_PDDO = SOL_PDDO'''
