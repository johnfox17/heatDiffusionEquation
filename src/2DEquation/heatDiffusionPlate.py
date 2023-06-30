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
tf = 0.7
numTimeSteps =int(tf/dt)
xCoords = np.arange(dx/2,l1, dx) #create the discrete x and y grids
yCoords = np.arange(dy/2,l2, dy) #create the discrete x and y grids
indexing = 'xy'

xCoords, yCoords = np.meshgrid(xCoords, yCoords, indexing=indexing)
xCoords = xCoords.reshape(-1, 1)
yCoords = yCoords.reshape(-1, 1)
coords = np.array([np.round(xCoords[:,0],3), np.round(yCoords[:,0],3)]).T
numNodes = len(xCoords)

def calcInitialCondition():
    initialCondition = np.zeros(numNodes)
    for iNode in range(numNodes):
        #initialCondition[iNode] = np.exp(-20*((coords[iNode][0]-0.5)**2+(coords[iNode][1]-0.5)**2))
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
    bVec00 = np.array([1,0,0,0,0,0])
    bVec20 = np.array([0,0,0,2,0,0])
    bVec02 = np.array([0,0,0,0,2,0])
    diffOrder = 2
    numBC = 1
    boundaries = np.array([0.005, 0.995, 0.005, 0.995])
    #boundaries = np.array([0.05, 0.95, 0.05, 0.95])
    
    BCY = np.array([dy/2,l2-dy/2])
    diffOrderBC = np.array([1])
    bVecBC = np.array([0,1,0])
    nodesBC = np.array([numNodes-1])

    ###############################
    #Solving with PDDO
    ###############################
    pddo = PDDO.PDDO(numNodes, coords, dx, dy, dt, horizon, diffOrder, \
        bVec00, bVec20, bVec02, boundaries)
    pddo.solve(tf, initialCondition)
    

    np.savetxt('C:\\Users\\docta\\Documents\\Thesis\\heatDiffusionEquation\\data\\SOL_PDDO.csv', pddo.SOL_PDDO, delimiter=",")
    np.savetxt('C:\\Users\\docta\\Documents\\Thesis\\heatDiffusionEquation\\data\\time.csv', pddo.time, delimiter=",")
    #a = input('').split(" ")[0]
    print('Done')

if __name__ == "__main__":
    main()
