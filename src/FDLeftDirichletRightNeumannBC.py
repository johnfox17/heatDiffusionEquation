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

    def enforceLeftBoundaryConditions(self, kMat):
        kMat[0,0] = 1 #Dirichlet
        kMat[0,1] = 0
        self.kMat = kMat

    def enforceRightBoundaryConditions(self): 
        numNodes = self.numNodes
        dx = self.dx
        dt = self.dt
        kMat = self.kMat
        kMat = kMat.astype(np.float)
        kappa = self.kappa
        kappa2 = self.kappa2
        kMat[numNodes-1, -3] = -2*kappa*dt/(kappa2*dx**2)
        kMat[numNodes-1, -2] = 0
        kMat[numNodes-1, -1] = 2*kappa*dt/(kappa2*dx**2)
        self.kMat = kMat

    def calcDuDt(self):
        numNodes = self.numNodes 
        dt = self.dt
        dx = self.dx
        kappa = self.kappa
        FD.createKMatrix(self)
        kMat = np.zeros([numNodes,numNodes])
        FD.enforceRightBoundaryConditions(self)
        identity = np.identity(numNodes-2)
        kMat[1:numNodes,0:numNodes] = np.multiply(kappa/dx**2, np.multiply(dt,self.kMat[1:numNodes,:]))
        kMat[1:numNodes-1,1:numNodes-1] = identity - kMat[1:numNodes-1,1:numNodes-1]
        kMat[1,0] = -kMat[1,0]
        kMat[numNodes-2,numNodes-1] = -kMat[numNodes-2,numNodes-1]
        FD.enforceLeftBoundaryConditions(self, kMat)
        self.dudt = self.kMat

    def enforceBoundaryConditionsRHS(self, initialCondition, timeIntegrationFlag):
        numNodes = self.numNodes
        dt = self.dt
        dx = self.dx
        BC = self.BC
        kappa = self.kappa
        kappa2 = self.kappa2
        if timeIntegrationFlag:
            RHS = initialCondition
            RHS[-1] = 2*dt*BC*kappa/(kappa2*dx)
        else:
            RHS = np.zeros([numNodes])
            RHS[0:numNodes]=  initialCondition[:]
            RHS[-1] = 2*dt*BC*kappa/(kappa2*dx)
        self.RHS = RHS


    def solve(self, tf, initialCondition):
        numNodes = self.numNodes
        dt = self.dt
        numTimeSteps =int(tf/dt)

        FD.calcDuDt(self)
        #np.savetxt('C:\\Users\\docta\\Documents\\Thesis\\heatDiffusionEquation\\data\\dudt.csv', self.dudt, delimiter=",")
        #print('Here')
        #a = input('').split(" ")[0]
        invDUDT = inv(csc_matrix(self.dudt)).toarray() 
        FD.enforceBoundaryConditionsRHS(self, initialCondition, False)
        RHS = self.RHS
        for iTimeStep in range(numTimeSteps):
            RHS = np.matmul(invDUDT, RHS)
            FD.enforceBoundaryConditionsRHS(self, RHS, True)
            if iTimeStep == numTimeSteps-1:
                SOL_FD = RHS
            RHS = self.RHS

        #np.savetxt('/home/doctajfox/Documents/Thesis_Research/heatDiffusionEquation/data/solFD.csv', SOL_FD, delimiter=",")
        #a = input('').split(" ")[0]


        self.SOL_FD = SOL_FD
