

##################################################################################################
#FD
##################################################################################################

class FD:
    def __init__(self, numNodes, dx):
        self.numNodes = numNodes
        self.dx = dx

    def createKMatrix():
         data0 = np.array([(numNodes-1)*[1]])
    data1 = np.array([(numNodes)*[-2]])
    data2 = np.array([(numNodes)*[1]])
    KFD = spdiags(data0, -1, numNodes, numNodes ).toarray()\
            +spdiags(data1, 0, numNodes, numNodes ).toarray()\
            +spdiags(data2, 1, numNodes, numNodes ).toarray()
    return KFD[1:numNodes-2,1:numNodes-2]

    def calcDuDt(t, initialConditionFD):
        kMat = createKMatrix()
        dudt = np.multiply(kappa/dx**2,np.matmul(kMat, initialConditionFD))
    return dudt

