import numpy as np
from scipy.sparse.linalg import inv
from scipy.sparse import csc_matrix
from scipy.sparse import spdiags
##################################################################################################
#FD
##################################################################################################

class FD:
    def __init__(self, numNodes, dx, dt, kappa, kappa2, BC):
        self.numNodes = numNodes
        self.dx = dx
        self.dt = dt
        self.kappa = kappa
        self.kappa2 = kappa2
        self.BC = BC[0]

    def createKMatrix(self):
        numNodes = self.numNodes
        data0 = np.array([(numNodes-1)*[1]])
        data1 = np.array([(numNodes)*[-2]])
        data2 = np.array([(numNodes)*[1]])
        kMat = spdiags(data0, -1, numNodes, numNodes ).toarray()\
            +spdiags(data1, 0, numNodes, numNodes ).toarray()\
            +spdiags(data2, 1, numNodes, numNodes ).toarray()
        self.kMat = kMat

    def enforceLeftBoundaryConditions(self):
        kMat = self.kMat
        kMat[0,0] = 1 #Dirichlet
        kMat[0,1] = 0
        self.kMat = kMat

    def enforceRightBoundaryConditions(self): 
        numNodes = self.numNodes
        dx = self.dx
        kMat = self.kMat
        kMat[numNodes-1, -2] = -1
        kMat[numNodes-1, -1] = 1
        self.kMat = kMat

    def calcDuDt(self):
        numNodes = self.numNodes
        dt = self.dt
        dx = self.dx
        kappa = self.kappa
        kMatAux = np.zeros([numNodes,numNodes])
        identity = np.identity(numNodes-2)
        kMatAux[1:numNodes-1,:] = np.multiply(kappa/dx**2, np.multiply(dt, self.kMat[1:numNodes-1,:]))
        kMatAux[1:numNodes-1,1:numNodes-1] = identity - kMatAux[1:numNodes-1,1:numNodes-1]
        kMatAux[0,:] = self.kMat[0,:]
        kMatAux[numNodes-1,:] = self.kMat[numNodes-1,:]
        kMatAux[1,0] = - kMatAux[1,0]
        kMatAux[numNodes-2,numNodes-1] = - kMatAux[numNodes-2,numNodes-1]
        
        self.dudt = kMatAux

    def enforceBoundaryConditionsRHS(self, initialCondition):
        dt = self.dt
        dx = self.dx
        BC = self.BC
        kappa = self.kappa
        kappa2 = self.kappa2
        initialCondition[0] = 0
        initialCondition[-1] = BC*dx/kappa2
        self.RHS = initialCondition


    def solve(self, tf, initialCondition):
        numNodes = self.numNodes
        dt = self.dt
        numTimeSteps =int(tf/dt)
        FD.createKMatrix(self)

        FD.enforceLeftBoundaryConditions(self)
        FD.enforceRightBoundaryConditions(self)
        FD.calcDuDt(self)
        FD.enforceBoundaryConditionsRHS(self, initialCondition)
        RHS = self.RHS
        invDUDT = inv(csc_matrix(self.dudt)).toarray()
        
        for iTimeStep in range(numTimeSteps):
            RHS = np.matmul(invDUDT, RHS)
            FD.enforceBoundaryConditionsRHS(self, RHS)
            if iTimeStep == numTimeSteps-1:
                SOL_FD = RHS
            RHS = self.RHS

        self.SOL_FD = SOL_FD
