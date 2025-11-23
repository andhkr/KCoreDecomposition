#ifndef FIXEDKCORE_CUH
#define FIXEDKCORE_CUH

#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>

typedef struct KCoreAlgoResult{
    double timeTaken; // in second
    int* KCoreNodes ; // nodes which are in Kcore graph
}KCoreAlgoResult;

// we will have a csr format graph
// device global variables
extern __device__  int dnnodes, dnedges, K,fnempty; // Fixed K-core,  // dfrontier is not empty;
// device pointers on host
extern int *dedges, *dindex, *dfrontier,  *dnextFrontier,  *dactive,  *ddegs;

// main kernel funciton for K core computation in parrallel
__global__ void fixedKcoreComputation(int* dedges, int* dindex, int* dfrontier, int* dnextFrontier, int* dactive, int* ddegs);

// updating the frontier list after each iteration of algorithm
__global__ void updateFrontier(int* dfrontier, int* dnextFrontier, int* dactive);

// initialising the frontier list and calculating the degree
__global__ void initFrontier(int* dedges, int* dindex, int* dfrontier, int* dnextFrontier, int* dactive, int* ddegs);

// allocating the csr format graph on device and initialising the device global variables
void DeviceSetup(int* hedges, int* hindex, int hnnodes, int hnedges,int hK);

// this function will launch fixedKcoreComputation for calculating the K
// Core graph and return the nodes present in that.
KCoreAlgoResult* launcKernelAndKcoreNodes(int hnnodes);

// singel threaded K core implementation on CPU
KCoreAlgoResult* cpufixedKcoreComputation(int* hedges, int* hindex,int nnodes, int nedges, int hK);
#endif
