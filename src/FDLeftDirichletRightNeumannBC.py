import numpy as np

from scipy.sparse import spdiags
##################################################################################################
#FD
##################################################################################################

class FD:
    def __init__(self, numNodes, dx):
        self.numNodes = numNodes
        self.dx = dx

    def createKMatrix(self):
        numNodes = self.numNodes
        data0 = np.array([(numNodes-1)*[1]])
        data1 = np.array([(numNodes)*[-2]])
        data2 = np.array([(numNodes)*[1]])
        kMat = spdiags(data0, -1, numNodes, numNodes ).toarray()\
            +spdiags(data1, 0, numNodes, numNodes ).toarray()\
            +spdiags(data2, 1, numNodes, numNodes ).toarray()
        return kMat

    def enforceBoundaryConditions(self, kMat):
        kMat[0,:] = 0
        kMat[0,0] = 1
        kMat[0,0] = 1
        print(kMat)
        a = input('').split(" ")[0]
        return kMat
    
    def calcDuDt(self, t, initialConditionFD):
        kMat = FD.createKMatrix(self)
        kMat = FD.enforceBoundaryConditions(self, kMat)

        print(self.numNodes)
        print(np.shape(kMat))
        print(kMat)
        a = input('').split(" ")[0]
        dudt = np.multiply(kappa/dx**2,np.matmul(kMat, initialConditionFD))
        return dudt

