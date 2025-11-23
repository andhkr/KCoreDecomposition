#include "FixedKCore.cuh"
/*
    This is single threaded fixed K core decomposition of a graph i.e;
    recursively delete all nodes from graph whose degree is less than K.
    recursively means : first remove the all nodes whose degree is less than k
    and then again start removing the nodes with degree less than K from updated graph
    till there is no one to remove i.e, everyone's degree >= K
*/

// helper funcion for calculating time
double rtclock();

/*
    graph input is in CSR format
    1) hedges : array of degree of node
    2) hindex : array of index of neigbours of node in hedges
    3) nnodes : number of nodes in graph
    3) nedges : number of edges in graph
    4) hK     : minimum degree the node should have to present in subgraph 
    return : bitmap of nodes which is present in hK core subgraph
*/
KCoreAlgoResult* cpufixedKcoreComputation(int* hedges, int* hindex,int nnodes, int nedges, int hK){

    int* queue = (int*) malloc(sizeof(int) * nnodes);
    int* degrees = (int*) malloc(sizeof(int) * nnodes);
    int* deleted = (int*) malloc(sizeof(int) * nnodes);

    int head = 0, tail = 0;

    double start = rtclock();
    // initialisation
    for (int i = 0; i < nnodes; ++i) {
        deleted[i] = 0;

        int start = hindex[i];
        int end   = hindex[i + 1];
        degrees[i] = end - start;

        if (degrees[i] < hK) {
            deleted[i] = 1;
            queue[tail++] = i;
        }
    }

    // bfs style iterative removal of nodes whose degree is less than hK
    /*
        basically every node will pushed and poped in queue atmost one time
        so there is no need of head = (head + 1) % nnodes similarly for tail.
    */
    while (head < tail) {
        int v = queue[head++];

        int start = hindex[v];
        int end   = hindex[v + 1];

        // going to neighbour of each node whose degree is less than hK
        // reducing its degree and if degree is less than hK then delete it from graph
        // and put it in queue
        for (int i = start; i < end; ++i) {
            int neigh = hedges[i];

            // if present in graph then decreaing the degree
            if (!deleted[neigh]) {
                degrees[neigh]--;

                if (degrees[neigh] < hK) {
                    queue[tail++] = neigh;
                    deleted[neigh] = 1;
                }
            }
        }
    }
    double end = rtclock();

    KCoreAlgoResult* Kcoreinfo = (KCoreAlgoResult*) malloc(sizeof(KCoreAlgoResult));
    Kcoreinfo->KCoreNodes = (int*) malloc(sizeof(int)*nnodes);

    // creating the bitmap of node which are part of K Core Graph
    for (int i = 0; i < nnodes; ++i) {
        Kcoreinfo->KCoreNodes[i] = (deleted[i] == 0); // not deleted -> in K core
    }

    Kcoreinfo->timeTaken = (end-start);
    return Kcoreinfo;
}
