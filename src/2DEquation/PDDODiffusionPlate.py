import numpy as np
from sklearn.neighbors import KDTree
from numpy.linalg import solve
from scipy.sparse.linalg import inv
from scipy.sparse import csc_matrix
##################################################################################################
#PDDO
##################################################################################################
class PDDO:
    def __init__(self, numNodes, coords, dx, dy, dt, horizon, diffOrder, \
            bVec00, bVec20, bVec02, boundaries):
        self.numNodes = numNodes
        self.coords = coords
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.horizon = horizon
        self.diffOrder = diffOrder
        self.bVec00 = bVec00
        self.bVec20 = bVec20
        self.bVec02 = bVec02
        self.boundaries = boundaries
        self.deltax = horizon * dx
        self.deltay = horizon * dy

    def findFamilyMembers(self):
        coords = self.coords
        numNodes = self.numNodes
        deltax = self.deltax
        tree = KDTree(coords, leaf_size=2)
        familyMembers = tree.query_radius(coords, r = deltax)
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


    def calcDiffEqSysMatrix(self, t):
        numNodes = self.numNodes
        familyMembers = self.familyMembers
        dx = self.dx
        dy = self.dy
        bVec00 = self.bVec00
        bVec20 = self.bVec20
        bVec02 = self.bVec02
        xXis = self.xXis
        yXis = self.yXis
        deltax = self.deltax 
        deltay = self.deltay
        deltaMag = np.sqrt(deltax**2 + deltay**2)
        sysMatrix = np.zeros([numNodes, numNodes])
        
        #Differential Equation Part
        for iNode in range(numNodes):
            family = familyMembers[iNode]
            xXi = xXis[iNode]
            yXi = yXis[iNode]
            diffMat = np.zeros([6,6])
            for iFamilyMember in range(len(family)):
                currentFamilyMember = family[iFamilyMember]
                #if currentFamilyMember != iNode:
                currentXXi = xXi[iFamilyMember]
                currentYXi = yXi[iFamilyMember]
                xiMag = np.sqrt(currentXXi**2+currentYXi**2) 
                pList = np.array([1, currentXXi/deltaMag, currentYXi/deltaMag, \
                            (currentXXi/deltaMag)**2, (currentYXi/deltaMag)**2, \
                            (currentXXi/deltaMag)*(currentYXi/deltaMag)]) 
                weight = np.exp(-4*(xiMag/deltaMag)**2)
                diffMat += weight*np.outer(pList,pList)*dx*dy
            for iFamilyMember in range(len(family)):
                currentFamilyMember = family[iFamilyMember]
                #if currentFamilyMember != iNode:
                currentXXi = xXi[iFamilyMember]
                currentYXi = yXi[iFamilyMember]
                xiMag = np.sqrt(currentXXi**2+currentYXi**2)
                pList = np.array([1, currentXXi/deltaMag, currentYXi/deltaMag, \
                            (currentXXi/deltaMag)**2, (currentYXi/deltaMag)**2, \
                            (currentXXi/deltaMag)*(currentYXi/deltaMag)])
                weight = np.exp(-4*(xiMag/deltaMag)**2)
                sysMatrix[iNode][ currentFamilyMember] = \
                            weight*(np.inner(solve(diffMat,bVec20), pList) + \
                            np.inner(solve(diffMat,bVec02),pList) + \
                            (1+t**2)*np.inner(solve(diffMat,bVec00),pList))*((dx*dy)/deltaMag**4)
        
        self.sysMatrixDiffEq = sysMatrix

    def findBoundaryNodes(self):
        boundaries = self.boundaries
        coords = self.coords
        xBoundaryNodes = np.array([np.where(coords[:,0]==boundaries[0])[0], \
                np.where(coords[:,0]==boundaries[1])[0]])
        yBoundaryNodes = np.array([np.where(coords[:,1]==boundaries[2])[0], \
                np.where(coords[:,1]==boundaries[3])[0]])
        self.xBoundaryNodes = xBoundaryNodes
        self.yBoundaryNodes = yBoundaryNodes

    def enforceXBoundaryConditions(self):
        xBoundaryNodes = self.xBoundaryNodes
        sysMatrix = self.sysMatrix
        for iXBoundary in range(2):
            sysMatrix[xBoundaryNodes[iXBoundary],:] = 0
            sysMatrix[xBoundaryNodes[iXBoundary],xBoundaryNodes[iXBoundary]] = 1
        self.sysMatrix = sysMatrix

    def calcBCSysMatrix(self, t):
        numNodes = self.numNodes
        xBoundaryNodes = self.xBoundaryNodes
        yBoundaryNodes = self.yBoundaryNodes
        familyMembers = self.familyMembers
        bVec00 = self.bVec00
        xXis = self.xXis
        yXis = self.yXis
        dx = self.dx
        dy = self.dy
        deltax = self.deltax
        deltay = self.deltay
        deltaMag = np.sqrt(deltax**2 + deltay**2)
        numBCX, numBCNodesX = np.shape(xBoundaryNodes)
        numBCY, numBCNodesY = np.shape(yBoundaryNodes)
        sysMatrix = np.zeros([numBCX*numBCNodesX+numBCY*numBCNodesY, numNodes]) 
        iRow = 0
        #X BC
        for iBC in range(numBCX):
            iBCNodes = xBoundaryNodes[iBC]
            for iNode in iBCNodes:
                family = familyMembers[iNode]
                xXi = xXis[iNode]
                yXi = yXis[iNode]
                diffMat = np.zeros([6,6])
                for iFamilyMember in range(len(family)):
                    currentFamilyMember = family[iFamilyMember]
                    #if currentFamilyMember != iNode:
                    currentXXi = xXi[iFamilyMember]
                    currentYXi = yXi[iFamilyMember]
                    xiMag = np.sqrt(currentXXi**2+currentYXi**2)
                    pList = np.array([1, currentXXi/deltaMag, currentYXi/deltaMag, \
                                (currentXXi/deltaMag)**2, (currentYXi/deltaMag)**2, \
                                (currentXXi/deltaMag)*(currentYXi/deltaMag)])
                    weight = np.exp(-4*(xiMag/deltaMag)**2)
                    diffMat += weight*np.outer(pList,pList)*dx*dy
                for iFamilyMember in range(len(family)):
                    currentFamilyMember = family[iFamilyMember]
                    #if currentFamilyMember != iNode:
                    currentXXi = xXi[iFamilyMember]
                    currentYXi = yXi[iFamilyMember]
                    xiMag = np.sqrt(currentXXi**2+currentYXi**2)
                    pList = np.array([1, currentXXi/deltaMag, currentYXi/deltaMag, \
                                (currentXXi/deltaMag)**2, (currentYXi/deltaMag)**2, \
                                (currentXXi/deltaMag)*(currentYXi/deltaMag)])
                    weight = np.exp(-4*(xiMag/deltaMag)**2)
                    sysMatrix[iRow][ currentFamilyMember] = \
                             weight*(np.inner(solve(diffMat,bVec00) ,pList))*dx*dy
                iRow = iRow + 1
        #Y BC
        for iBC in range(numBCY):
            iBCNodes = yBoundaryNodes[iBC]
            for iNode in iBCNodes:
                family = familyMembers[iNode]
                xXi = xXis[iNode]
                yXi = yXis[iNode]
                diffMat = np.zeros([6,6])
                for iFamilyMember in range(len(family)):
                    currentFamilyMember = family[iFamilyMember]
                    #if currentFamilyMember != iNode:
                    currentXXi = xXi[iFamilyMember]
                    currentYXi = yXi[iFamilyMember]
                    xiMag = np.sqrt(currentXXi**2+currentYXi**2)
                    pList = np.array([1, currentXXi/deltaMag, currentYXi/deltaMag, \
                                (currentXXi/deltaMag)**2, (currentYXi/deltaMag)**2, \
                                (currentXXi/deltaMag)*(currentYXi/deltaMag)])
                    weight = np.exp(-4*(xiMag/deltaMag)**2)
                    diffMat += weight*np.outer(pList,pList)*dx*dy
                for iFamilyMember in range(len(family)):
                    currentFamilyMember = family[iFamilyMember]
                    #if currentFamilyMember != iNode:
                    currentXXi = xXi[iFamilyMember]
                    currentYXi = yXi[iFamilyMember]
                    xiMag = np.sqrt(currentXXi**2+currentYXi**2)
                    pList = np.array([1, currentXXi/deltaMag, currentYXi/deltaMag, \
                                (currentXXi/deltaMag)**2, (currentYXi/deltaMag)**2, \
                                (currentXXi/deltaMag)*(currentYXi/deltaMag)])
                    weight = np.exp(-4*(xiMag/deltaMag)**2)
                    sysMatrix[iRow][ currentFamilyMember] = \
                            weight*(np.inner(solve(diffMat,bVec00) ,pList))*dx*dy 
                iRow = iRow + 1
        self.sysMatrixBC = sysMatrix


    def enforceBoundaryConditionsRHS(self, RHS, t):
        numNodes = self.numNodes
        coords = self.coords
        xBoundaryNodes = self.xBoundaryNodes
        yBoundaryNodes = self.yBoundaryNodes
        xBoundaryNodes = xBoundaryNodes.flatten()
        
        for iNode in range(numNodes):
            RHS[iNode] = (2*np.pi**2-t**2-2)*np.exp(-t)*np.sin(np.pi*coords[iNode,0])*\
                    np.cos(np.pi*coords[iNode,1])

        RHS[xBoundaryNodes] = 0       
        RHS[yBoundaryNodes[0]] = np.exp(t)*np.sin(np.pi*coords[yBoundaryNodes[0],0])
        RHS[yBoundaryNodes[1]] = -np.exp(t)*np.sin(np.pi*coords[yBoundaryNodes[1],0])
        RHS[-1] = 0
        RHS[-2] = 0
        self.RHS = RHS

    def calcDuDt(self):
        sysMatrix = self.sysMatrix
        numNodes = self.numNodes
        dt = self.dt
        xBoundaryNodes = self.xBoundaryNodes
        yBoundaryNodes = self.yBoundaryNodes
        boundaryNodes = np.concatenate((xBoundaryNodes,yBoundaryNodes))
        boundaryNodes = boundaryNodes.flatten()
        mask = np.ones(numNodes, dtype=bool)
        mask[boundaryNodes] = False
        sysMatrix[mask,:] = -sysMatrix[mask,:] 
        sysMatrix[mask, mask] = 1 + sysMatrix[mask, mask]
        self.dudt = sysMatrix


    def calcICSysMatrix(self):
        numNodes = self.numNodes
        familyMembers = self.familyMembers
        dx = self.dx
        dy = self.dy
        bVec00 = self.bVec00
        xXis = self.xXis
        yXis = self.yXis
        deltax = self.deltax
        deltay = self.deltay
        deltaMag = np.sqrt(deltax**2 + deltay**2)
        sysMatrix = np.zeros([numNodes, numNodes])

        #Differential Equation Part
        for iNode in range(numNodes):
            family = familyMembers[iNode]
            xXi = xXis[iNode]
            yXi = yXis[iNode]
            diffMat = np.zeros([6,6])
            for iFamilyMember in range(len(family)):
                currentFamilyMember = family[iFamilyMember]
                #if currentFamilyMember != iNode:
                currentXXi = xXi[iFamilyMember]
                currentYXi = yXi[iFamilyMember]
                xiMag = np.sqrt(currentXXi**2+currentYXi**2)
                pList = np.array([1, currentXXi/deltaMag, currentYXi/deltaMag, \
                            (currentXXi/deltaMag)**2, (currentYXi/deltaMag)**2, \
                            (currentXXi/deltaMag)*(currentYXi/deltaMag)])
                weight = np.exp(-4*(xiMag/deltaMag)**2)
                diffMat += weight*np.outer(pList,pList)*dx*dy
            for iFamilyMember in range(len(family)):
                currentFamilyMember = family[iFamilyMember]
                #if currentFamilyMember != iNode:
                currentXXi = xXi[iFamilyMember]
                currentYXi = yXi[iFamilyMember]
                xiMag = np.sqrt(currentXXi**2+currentYXi**2)
                pList = np.array([1, currentXXi/deltaMag, currentYXi/deltaMag, \
                            (currentXXi/deltaMag)**2, (currentYXi/deltaMag)**2, \
                            (currentXXi/deltaMag)*(currentYXi/deltaMag)])
                weight = np.exp(-4*(xiMag/deltaMag)**2)
                sysMatrix[iNode][ currentFamilyMember] = \
                            weight*(np.inner(solve(diffMat,bVec00),pList))*((dx*dy)/deltaMag**0)

        self.sysMatrixIC = sysMatrix

    def constructGlobalSysMatrix(self):
        sysMatrixDiffEq = self.sysMatrixDiffEq
        sysMatrixIC = self.sysMatrixIC
        sysMatrixBC = self.sysMatrixBC
        rowsDiffEq, colsDiffEq = np.shape(sysMatrixDiffEq)
        rowsIC, colsIC = np.shape(sysMatrixIC)
        rowsBC, colsBC = np.shape(sysMatrixBC)
        globalSysMatrix = np.zeros([rowsDiffEq + rowsIC + rowsBC, rowsDiffEq + rowsIC + rowsBC])
        conditionsAppended = np.append(sysMatrixIC,sysMatrixBC, axis=0)
        conditionsAppendedTransposed = np.transpose(conditionsAppended)
        rowsConditionsAppendedTransposed, colsConditionsAppendedTransposed = np.shape(conditionsAppendedTransposed)
        
        #sysMatrixDiffEq
        globalSysMatrix[0:rowsDiffEq, 0:colsDiffEq] = sysMatrixDiffEq
        #sysMatrixIC and sysMatrixBC
        globalSysMatrix[rowsDiffEq:rowsDiffEq + rowsIC + rowsBC, 0:colsBC] = conditionsAppended
        globalSysMatrix[0:rowsConditionsAppendedTransposed, colsDiffEq:colsDiffEq + colsConditionsAppendedTransposed] =\
                conditionsAppendedTransposed
        self.globalSysMatrix = globalSysMatrix
    
    def contructRHS(self, initialCondition, t):
        coords = self.coords
        sysMatrixDiffEq = self.sysMatrixDiffEq
        sysMatrixIC = self.sysMatrixIC
        sysMatrixBC = self.sysMatrixBC
        numNodes = self.numNodes
        xBoundaryNodes = self.xBoundaryNodes
        yBoundaryNodes = self.yBoundaryNodes
        xBoundaryNodes = xBoundaryNodes.flatten()

        rowsDiffEq, colsDiffEq = np.shape(sysMatrixDiffEq)
        rowsIC, colsIC = np.shape(sysMatrixIC)
        rowsBC, colsBC = np.shape(sysMatrixBC)
        conditionRHS = np.zeros([rowsBC, 1])

        globalRHS = np.zeros([rowsDiffEq + rowsIC + rowsBC, 1])
        
        #RHS of differential equation 
        for iNode in range(rowsDiffEq):
            globalRHS[iNode,0] = (2*np.pi**2-t**2-2)*np.exp(-t)*np.sin(np.pi*coords[iNode,0])*\
                    np.cos(np.pi*coords[iNode,1])
        
        #RHS IC
        globalRHS[rowsDiffEq:rowsDiffEq + rowsIC,0] = initialCondition
        
        #RHS BC
        condition = np.exp(t)*np.sin(np.pi*coords[yBoundaryNodes[0],0]) 
        conditionRHS[int(rowsBC/2):int(3*rowsBC/4)] = condition.reshape((100,1))
        condition = -np.exp(t)*np.sin(np.pi*coords[yBoundaryNodes[1],0])
        conditionRHS[int(3*rowsBC/4):rowsBC] = condition.reshape((100,1))
        globalRHS[rowsDiffEq + rowsIC:rowsDiffEq + rowsIC + rowsBC,0] = conditionRHS[:,0]

        self.globalRHS = globalRHS
    
    def solve(self, tf, initialCondition):
        numNodes = self.numNodes
        dt = self.dt
        numTimeSteps =int(tf/dt)
        PDDO.findFamilyMembers(self)
        PDDO.calcXis(self)
        PDDO.findBoundaryNodes(self)
        SOL_PDDO = []
        time = []
        t = 0
        for i in range(numTimeSteps):
            print(t)
            PDDO.calcDiffEqSysMatrix(self, t)
            PDDO.calcICSysMatrix(self)
            PDDO.calcBCSysMatrix(self,t)
            #np.savetxt('C:\\Users\\docta\\Documents\\Thesis\\heatDiffusionEquation\\data\\sysMatrix.csv', self.sysMatrixBC, delimiter=",")
            PDDO.constructGlobalSysMatrix(self)
            PDDO.contructRHS(self, initialCondition, t)
            if i==0:
                SOL_PDDO.append(initialCondition)
                time.append(t)
            else:
                SOL  =  np.multiply(dt,np.matmul(self.globalSysMatrix, self.globalRHS))
                initialCondition = initialCondition + SOL[numNodes:2*numNodes,0]
                SOL_PDDO.append(initialCondition)
                time.append(t)
                #print(np.shape(SOL))
                #a = input('').split(" ")[0]
            t = t + dt
        SOL_PDDO = np.array(SOL_PDDO)
        time = np.array(time)
        #print('Done')
        #a = input('').split(" ")[0]
        self.SOL_PDDO = SOL_PDDO
        self.time = time
