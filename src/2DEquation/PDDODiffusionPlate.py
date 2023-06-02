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
        dy = self.dy
        bVec00 = self.bVec00
        bVec20 = self.bVec20
        bVec02 = self.bVec02
        xXis = self.xXis
        yXis = self.yXis
        delta = self.delta 
        sysMatrix = np.zeros([numNodes + 2, numNodes + 2])#added two BC and 1 IC
        
        #Differential Equation Part
        for iNode in range(numNodes):
            family = familyMembers[iNode]
            xXi = xXis[iNode]
            yXi = yXis[iNode]
            diffMat = np.zeros([6,6])
            for iFamilyMember in range(len(family)):
                currentFamilyMember = family[iFamilyMember]
                if currentFamilyMember != iNode:
                    currentXXi = xXi[iFamilyMember]
                    currentYXi = yXi[iFamilyMember]
                    pList = np.array([1, currentXXi, currentYXi, (currentXXi)**2, (currentYXi)**2,\
                        currentXXi*currentYXi]) 
                    weight = np.exp(-4*(np.sqrt(currentXXi**2+currentYXi**2)))
                    diffMat += weight*np.outer(pList,pList)*dx*dy
            for iFamilyMember in range(len(family)):
                currentFamilyMember = family[iFamilyMember]
                if currentFamilyMember != iNode:
                    currentXXi = xXi[iFamilyMember]
                    currentYXi = yXi[iFamilyMember]
                    pList = np.array([1, currentXXi, currentYXi, (currentXXi)**2, (currentYXi)**2,\
                        currentXXi*currentYXi])
                    weight = np.exp(-4*(np.sqrt(currentXXi**2+currentYXi**2)))
                    sysMatrix[iNode][ currentFamilyMember] = weight*(np.inner(solve(diffMat,bVec20)\
                        ,pList)+ np.inner(solve(diffMat,bVec02),pList))*dx*dy
        self.sysMatrix = sysMatrix

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

    def enforceYBoundaryConditions(self):
        numNodes = self.numNodes
        yBoundaryNodes = self.yBoundaryNodes
        sysMatrix = self.sysMatrix 
        familyMembers = self.familyMembers
        bVec00 = self.bVec00
        xXis = self.xXis
        yXis = self.yXis
        dx = self.dx
        dy = self.dy
        delta = self.delta
        numBC, numBCNodes = np.shape(yBoundaryNodes)
        
        for iBC in range(numBC):
            iBCNodes = yBoundaryNodes[iBC]
            for iNode in iBCNodes:
                family = familyMembers[iNode]
                xXi = xXis[iNode]
                yXi = yXis[iNode]
                diffMat = np.zeros([6,6])
                for iFamilyMember in range(len(family)):
                    currentFamilyMember = family[iFamilyMember]
                    if currentFamilyMember != iNode:
                        currentXXi = xXi[iFamilyMember]
                        currentYXi = yXi[iFamilyMember]
                        pList = np.array([1, currentXXi, currentYXi, (currentXXi)**2, (currentYXi)**2,\
                            currentXXi*currentYXi])
                        weight = np.exp(-4*(np.sqrt(currentXXi**2+currentYXi**2)))
                        diffMat += weight*np.outer(pList,pList)*dx*dy

                for iFamilyMember in range(len(family)):
                    currentFamilyMember = family[iFamilyMember]
                    if currentFamilyMember != iNode:
                        currentXXi = xXi[iFamilyMember]
                        currentYXi = yXi[iFamilyMember]
                        pList = np.array([1, currentXXi, currentYXi, (currentXXi)**2, (currentYXi)**2,\
                        currentXXi*currentYXi])
                        weight = np.exp(-4*(np.sqrt(currentXXi**2+currentYXi**2)))
                        sysMatrix[numNodes+iBC][ currentFamilyMember] = \
                            weight*(np.inner(solve(diffMat,bVec00) ,pList))*dx*dy
                        sysMatrix[currentFamilyMember][numNodes+iBC] = \
                            sysMatrix[numNodes+iBC][ currentFamilyMember]
        self.sysMatrix = sysMatrix


    def enforceBoundaryConditionsRHS(self, RHS, t):
        coords = self.coords
        xBoundaryNodes = self.xBoundaryNodes
        yBoundaryNodes = self.yBoundaryNodes
        xBoundaryNodes = xBoundaryNodes.flatten()
        RHS[xBoundaryNodes] = 0       
        RHS[yBoundaryNodes[0]] = np.exp(t)*np.sin(np.pi*coords[yBoundaryNodes[0],0])
        RHS[yBoundaryNodes[1]] = -np.exp(t)*np.sin(np.pi*coords[yBoundaryNodes[1],0])
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


    def solve(self, tf, RHS):
        numNodes = self.numNodes
        dt = self.dt
        numTimeSteps =int(tf/dt)
        PDDO.findFamilyMembers(self)
        PDDO.calcXis(self)
        PDDO.calcSysMatrix(self)
        PDDO.findBoundaryNodes(self)
        PDDO.enforceXBoundaryConditions(self)
        PDDO.enforceYBoundaryConditions(self)
        

        RHS = np.append(RHS,0) 
        RHS = np.append(RHS,0)
        t = 0
        for i in range(30):
            t = t + dt
            PDDO.enforceBoundaryConditionsRHS(self, RHS, t)
            RHS = self.RHS
            RHS  = RHS + np.multiply(dt,np.matmul(self.sysMatrix, RHS))
            #np.savetxt('/home/doctajfox/Documents/Thesis_Research/heatDiffusionEquation/data/aux'+str(i)+'.csv', RHS, delimiter=",")
        #invDUDT = inv(csc_matrix(self.dudt)).toarray()
        #a = input('').split(" ")[0]
        self.SOL_PDDO = RHS
