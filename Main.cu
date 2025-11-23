#include <cuda.h>
#include <stdio.h>
#include <sys/time.h>
#include "FixedKCore.cuh"
#include <assert.h>

int *hedges, *hindex; // host allocations
// hedges → |E|     (neighbours of nodes)
// hindex → |V| + 1 (index for neigbours of node in hedges)

int hnedges, hnnodes; // number of edges , number of nodes

FILE *FP; // file pointer of graph file 

// helper funcion for calculating time
double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat = gettimeofday(&Tp, &Tzp);
    if (stat != 0)
        printf("Error return from gettimeofday: %d", stat);
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

// creating the graph in CSR format
void allocate_graph_host(int numnodes, int numedges)
{
    hedges = (int *)malloc(numedges * sizeof(int));
    hindex = (int *)malloc((numnodes + 1) * sizeof(int));

    char line[256];
    int edgi = 0, indi = 1, prev = 0, count = 0;
    hindex[0] = 0;

    while (fgets(line, sizeof(line), FP))
    {
        int u, v;
        if (sscanf(line, "%d %d", &u, &v) == 2)
        {
            while (prev < u)
            {
                hindex[indi++] = count;
                prev++;
            }
            hedges[edgi++] = v;
            count++;
        }
    }

    while (indi <= numnodes)
        hindex[indi++] = count;
}

/*
 Main function which will call the fixed K core decomposition for CPU and GPU
 and verify the results and give time difference
*/
int main(int argc, char *argv[])
{
    if (argc < 5){
        printf("Usage: %s <No of Nodes> <No of Edges> <Filepath> no K\n", argv[0]);
        return 1;
    }

    hnnodes = atoi(argv[1]);
    hnedges = atoi(argv[2]);


    printf("No of nodes=%d\t No of edges=%d\n", hnnodes, hnedges);

    FP = fopen(argv[3], "r");
    int hK = atoi(argv[4]);

    if (FP == NULL){

        printf("Opening file %s failed\n", argv[3]);
        exit(0);
    }
    else{
        printf("File %s opened\n", argv[3]);
    }

    // csr format graph creation
    allocate_graph_host(hnnodes, hnedges);

    // fixed K core decompostion on CPU
    KCoreAlgoResult* cpuKcoreinfo = cpufixedKcoreComputation(hedges,hindex,hnnodes,hnedges,hK);

    // Now on GPU
    // allocating the sapce for edges and index array on device 
    // and copying : device setup for K-Core Computation
    DeviceSetup(hedges, hindex, hnnodes, hnedges, hK);

    // calling function for Kcore decomposition on GPU. 
    KCoreAlgoResult* gpuKcoreinfo = launcKernelAndKcoreNodes(hnnodes);
    cudaDeviceSynchronize();

    // checking the correctness for GPU calculation for K-Core graph
    // by asserting that same nodes are present in KCore graph 
    // calculated by both algorithm cpu and gpu based.

    for(int i = 0;i<hnnodes;++i){
        assert(cpuKcoreinfo->KCoreNodes[i] == gpuKcoreinfo->KCoreNodes[i]);
    }

    // after verification , printing the time taken by both algorithm
    printf("CPU time: %lf msec\n", (cpuKcoreinfo->timeTaken) * 1000.0);
    printf("GPU time: %lf msec\n", (gpuKcoreinfo->timeTaken) * 1000.0);

}
