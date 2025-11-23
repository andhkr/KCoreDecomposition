#include <stdio.h>
#include <cuda.h>
#include "FixedKCore.cuh"

// helper function to calculating the time
double rtclock();

// device global variables
__device__  int dnnodes ; // number of nodes in graph 
__device__  int dnedges ; // number of edges in graph
__device__  int K       ; // The K core : minimun degree node should have to be present in K core Graph
__device__  int fnempty ; // This is basically a flag from a host function
// to tell that there is no node whose degree is less <= K , 
// i.e, K core graph is calculated.

// device pointers on host, of array allocated on device memory
/*
int* dedges        : array of edges of nodes
int* dindex        : array of index of nodes for their neigbours in dedges
int* dfrontier     : bitmap of of nodes of current graph whose degree <= K
int* dnextFrontier : nodes whose degree comes down K in current iteration
int* dactive       : nodes who are still in graph (nodes whose degree <= K)
int* ddegs         : degree of nodes
*/
int *dedges, *dindex, *dfrontier,  *dnextFrontier,  *dactive,  *ddegs;   

/* 
main kernel function which is are launching threads per thread
and doing parallel scaning of neighbours of node who are in 
frontier list (nodes whose degree <= K) and decreasing their degree 
if degree comes down <= K then added to nextFrontier list (used for next iteration)
*/

__global__ void fixedKcoreComputation(int* dedges, int* dindex, int* dfrontier, int* dnextFrontier, int* dactive, int* ddegs){
    // node number
    int v = blockDim.x*blockIdx.x + threadIdx.x;

    // if thread is not corresponding to node or not in frontier list then return
    if(v>=dnnodes || dfrontier[v] == 0) return;

    dactive[v] = 0;
    dfrontier[v] = 0;

    int start = dindex[v];
    int end = dindex[v+1];

    // scanning the neighbour and if present in graph then 
    // decreasing the degree
    for(int i = start; i < end ; ++i) {
        if(dactive[dedges[i]]){
            atomicSub(&ddegs[dedges[i]],1);

            if(ddegs[dedges[i]] < K){
                dnextFrontier[dedges[i]] = 1;
            }
        }
    }

};

// after every iteration of degree reduction (scaning the neighbours of nodes in frontier list)
// updating the frontier list by nodes who are in nextFrontier list (whose degree become <= K
// in just previous iteration.)  and updating the flag that whether we reached the K core graph or not
// i.e fontier list is now empty
__global__ void updateFrontier(int* dfrontier, int* dnextFrontier, int* dactive){
    
    int v = blockDim.x*blockIdx.x + threadIdx.x;

    if( v >= dnnodes || dactive[v] == 0) return;

    if(dnextFrontier[v]){
        dfrontier[v] = 1;
        fnempty = 1;
    }
    dnextFrontier[v] = 0;
}

// before launching the kernel for K core computation
// initialising the frontier and nextFrontier list and calculating degree.
__global__ void initFrontier(int* dedges, int* dindex, int* dfrontier, int* dnextFrontier, int* dactive, int* ddegs){

    int v = blockDim.x * blockIdx.x + threadIdx.x;

    if(v >= dnnodes) return;

    dnextFrontier[v] = 0;
    dfrontier[v] = 0;
    dactive[v] = 1;
    // by CSR format property
    ddegs[v] = dindex[v+1] - dindex[v];

    // if degree is less than K add to frontier list
    if(ddegs[v] < K){
        dfrontier[v] = 1;
        fnempty = 1;
    }
    
}

// allocating the csr format graph on device by allocating memory on device and copying
// the graph from host
// and copying the number of nodes, number of edges and value of K on device from host.
void DeviceSetup(int* hedges, int* hindex, int hnnodes, int hnedges,int hK){
    
    // allocating dedges array on device 
    cudaMalloc(&dedges, sizeof(int)*hnedges);
    // allocating dindex array on device
    cudaMalloc(&dindex, sizeof(int)*(hnnodes+1));
    // allocating dfrontier array on device
    cudaMalloc(&dfrontier, sizeof(int)*hnnodes);
    // allocating dnextFrontier on device
    cudaMalloc(&dnextFrontier, sizeof(int)*hnnodes);
    // allocating  dactive on device
    cudaMalloc(&dactive, sizeof(int)*hnnodes);
    // allocating ddegs on device
    cudaMalloc(&ddegs, sizeof(int)*hnnodes);

    // copying the csr format edges and index array from host to device
    cudaMemcpy(dedges,hedges,sizeof(int)*hnedges, cudaMemcpyHostToDevice);
    cudaMemcpy(dindex,hindex,sizeof(int)*(hnnodes+1), cudaMemcpyHostToDevice);
    // copying the number of nodes, number of edges and K Core value to device
    cudaMemcpyToSymbol(dnedges,&hnedges,sizeof(int));
    cudaMemcpyToSymbol(dnnodes,&hnnodes,sizeof(int));
    cudaMemcpyToSymbol(K,&hK,sizeof(int));

}

// give the nodes which are in k-core
KCoreAlgoResult* launcKernelAndKcoreNodes(int hnnodes){
    

    int done = 0; // flag for complition of algorithm i.e, frontier list is empty now
    int TPB = 512; // threads per block
    int blocksct = (hnnodes + TPB-1) / TPB; // blocks per grid 

    double start = rtclock();
    // initialising the fnempty to zero
    int zero = 0;
    cudaMemcpyToSymbol(fnempty,&zero,sizeof(int));
    // kernel launch for frontier list initialisation and degree calucaltion
    initFrontier<<<blocksct,TPB>>> (dedges,dindex,dfrontier,dnextFrontier,dactive,ddegs);
    cudaDeviceSynchronize(); 
    // flag for frontier list is empty or not if empty that is graph is already in K core
    cudaMemcpyFromSymbol(&done, fnempty,sizeof(int));
    
    // start iteration until Graph reached in K core
    while(done){
        // kernel launch
        fixedKcoreComputation<<<blocksct,TPB>>> (dedges,dindex,dfrontier,dnextFrontier,dactive,ddegs);
        // setting fnempty to zero to check after updating the frontier list
        cudaMemcpyToSymbol(fnempty,&zero,sizeof(int));
        // updating the frontier list
        updateFrontier<<<blocksct,TPB>>> (dfrontier,dnextFrontier,dactive);
        cudaDeviceSynchronize(); 
        // copying the flag value i.e, frontier list is empty or not
        cudaMemcpyFromSymbol(&done,fnempty,sizeof(int));
        cudaDeviceSynchronize();
    }

    double end = rtclock();
    // now copying the bitmap of node which are left in K core graph.
    KCoreAlgoResult* dKcoreinfo = (KCoreAlgoResult*) malloc(sizeof(KCoreAlgoResult));
    dKcoreinfo->KCoreNodes = (int*) malloc(sizeof(int)*hnnodes);

    cudaMemcpy(dKcoreinfo->KCoreNodes, dactive, sizeof(int)*hnnodes, cudaMemcpyDeviceToHost);
    dKcoreinfo->timeTaken = (end - start) ;
    
    return dKcoreinfo;

}

