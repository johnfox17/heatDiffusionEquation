import numpy as np
from sklearn.neighbors import KDTree
from numpy.linalg import solve
from scipy.sparse.linalg import inv
from scipy.sparse import csc_matrix
##################################################################################################
#PDDO
##################################################################################################
class PDDO:
    def __init__(self, numNodes, coords, dx, dy, dt, deltaX, deltaY, horizon, diffOrder, \
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
        self.deltaX = deltaX
        self.deltaY = deltaY
    
    def findFamilyMembers(self):
        coords = self.coords
        numNodes = self.numNodes
        deltaX = self.deltaX
        tree = KDTree(coords, leaf_size=2)
        familyMembers = tree.query_radius(coords, r = deltaX)
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
        coords = self.coords
        dx = self.dx
        dy = self.dy
        bVec00 = self.bVec00
        bVec20 = self.bVec20
        bVec02 = self.bVec02
        xXis = self.xXis
        yXis = self.yXis
        deltaX = self.deltaX 
        deltaY = self.deltaY
        deltaMag = np.sqrt(deltaX**2 + deltaY**2)
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
                            (1+t**2)*np.inner(solve(diffMat,bVec00),pList))*((dx*dy)/deltaMag**0)#4

        self.sysMatrix = sysMatrix + np.identity(numNodes)

    def findBoundaryNodes(self):
        boundaries = self.boundaries
        coords = self.coords
        xBoundaryNodes = np.array([np.where(coords[:,0]==boundaries[0])[0], \
                np.where(coords[:,0]==boundaries[1])[0]])
        yBoundaryNodes = np.array([np.where(coords[:,1]==boundaries[2])[0], \
                np.where(coords[:,1]==boundaries[3])[0]])
        self.xBoundaryNodes = xBoundaryNodes
        self.yBoundaryNodes = yBoundaryNodes

    def enforceBoundaryConditions(self, initialCondition, t):
        numNodes = self.numNodes
        coords = self.coords
        xBoundaryNodes = self.xBoundaryNodes
        yBoundaryNodes = self.yBoundaryNodes
        xBoundaryNodes = xBoundaryNodes.flatten()
        
        initialCondition[xBoundaryNodes] = 0
        initialCondition[yBoundaryNodes[0]] = np.exp(-t)*np.sin(np.pi*coords[yBoundaryNodes[0],0]).reshape((21,1))
        initialCondition[yBoundaryNodes[1]] = -np.exp(-t)*np.sin(np.pi*coords[yBoundaryNodes[0],0]).reshape((21,1))
        return initialCondition 

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


    def calcNonHomogenousArray(self, t):
        coords = self.coords
        numNodes = self.numNodes
        nonHomogeneousPart = np.zeros([numNodes, 1])
        for iNode in range(numNodes):
            nonHomogeneousPart[iNode,0] = (2*np.pi**2-t**2-2)*np.exp(-t)*np.sin(np.pi*coords[iNode,0])*\
                np.cos(np.pi*coords[iNode,1])
        self.nonHomogeneousPart = nonHomogeneousPart

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
        
        for i in range(numTimeSteps+3):
            print(t)
            if i==0:
                SOL_PDDO.append(np.transpose(initialCondition))
                time.append(t)
            else:
                SOL_PDDO.append(np.transpose(initialCondition))
                time.append(t)
            
            PDDO.calcDiffEqSysMatrix(self, t)
            PDDO.calcNonHomogenousArray(self, t)
            initialCondition = np.multiply(dt,np.matmul(self.sysMatrix,initialCondition)+self.nonHomogeneousPart) 
            t = t + dt
        self.SOL_PDDO = np.squeeze(np.array(SOL_PDDO))
        self.time = np.array(time)
        #print('Done')
        #a = input('').split(" ")[0]
