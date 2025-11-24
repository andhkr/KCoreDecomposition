#!/bin/bash

K=10
EXEC=./fixedKCoreComputation

# Create profiling directory
PROFILE_DIR="profiling_results"
mkdir -p $PROFILE_DIR

echo "=========================================="
echo "CUDA Profiling for Fixed K-Core (K=$K)"
echo "=========================================="
echo ""
echo "Profiling graph: 1,000,000 nodes, 4,999,985 edges"
echo ""

GRAPH="1000000 4999985 testGraph_1000000_4999985.txt"

echo "Running basic profiling..."
nvprof $EXEC $GRAPH $K > /dev/null 2> $PROFILE_DIR/BasicProfile.txt

echo "Creating detailed kernel trace..."
nvprof --print-gpu-trace $EXEC $GRAPH $K > /dev/null 2> $PROFILE_DIR/KernelTrace.txt

echo "Creating CUDA API trace..."
nvprof --print-api-trace $EXEC $GRAPH $K > /dev/null 2> $PROFILE_DIR/APITrace.txt

echo "Creating summary analysis..."
nvprof --log-file $PROFILE_DIR/FullProfile.txt $EXEC $GRAPH $K > /dev/null 2>&1

echo ""
echo "=========================================="
echo "Profiling Complete!"
echo "=========================================="
echo ""
echo "Generated files in $PROFILE_DIR/:"
echo ""
echo "  1. BasicProfile.txt      - Overall GPU time breakdown"
echo "  2. KernelTrace.txt       - Timeline of each kernel call"
echo "  3. APITrace.txt          - CUDA API call timeline"
echo "  4. FullProfile.txt       - Complete profiling data"
echo ""

