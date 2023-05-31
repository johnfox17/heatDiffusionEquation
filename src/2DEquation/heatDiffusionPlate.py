import numpy as np
import matplotlib.pyplot as plt
import PDDODiffusionPlate as PDDO


#Defining constants
l1=1
l2=1
N = 100
dx = l1/N
dy = l2/N
dt = dx
t0 = 0
tf = 5
numTimeSteps =int(tf/dt)
xCoords = np.arange(dx/2,l1, dx) #create the discrete x and y grids
yCoords = np.arange(dy/2,l2, dy) #create the discrete x and y grids
indexing = 'xy'

xCoords, yCoords = np.meshgrid(xCoords, yCoords, indexing=indexing)
xCoords = xCoords.reshape(-1, 1)
yCoords = yCoords.reshape(-1, 1)
coords = np.array([xCoords[:,0], yCoords[:,0]]).T
numNodes = len(xCoords)

def calcInitialCondition():
    initialCondition = np.zeros(numNodes)
    for iNode in range(numNodes):
        initialCondition[iNode] = 0
    return initialCondition


def main():
    
    ##############################
    #Initial Condition
    ##############################
    initialCondition =  calcInitialCondition() 
    
    ##############################
    #PDDO Setup
    ##############################
    horizon = 3.015
    delta = horizon * dx
    bVec = np.array([0,0,2])
    diffOrder = 2
    numBC = 1
    BC = np.array([0.01])
    diffOrderBC = np.array([1])
    bVecBC = np.array([0,1,0])
    nodesBC = np.array([numNodes-1])

    ###############################
    #Solving with PDDO
    ###############################
    pddo = PDDO.PDDO(numNodes, coords, dx, dy, dt, horizon, diffOrder, \
        bVec, numBC, BC, nodesBC, diffOrderBC, bVecBC)
    pddo.solve(tf, initialCondition)

    print(numNodes)



    #np.savetxt('/home/doctajfox/Documents/Thesis_Research/heatDiffusionEquation/data/sysMatrix.csv', sysMatrix, delimiter=",")
    #a = input('').split(" ")[0]
    print('Done')

if __name__ == "__main__":
    main()
