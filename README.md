# Fixed K-Core Computation: CPU vs GPU Implementation

A high-performance parallel CUDA implementation of the Batagelj–Zaveršnik k-core decomposition algorithm, with comparative analysis against a single-threaded CPU implementation.

## Project Overview

This project implements the classical Batagelj–Zaveršnik k-core peeling algorithm in two forms:

- A single-threaded CPU version, serving as a baseline.
- A parallel CUDA version, designed to accelerate k-core computation using a frontier-based degree-reduction approach.
- The k-core of a graph is the maximal subgraph in which every vertex has degree at least K, after iteratively removing all vertices with degree < K.

## Features

- **CPU Implementation**: Single-threaded baseline implementation
- **GPU Implementation**: Parallel CUDA implementation with significant speedup
- **Automated Testing**: Script for testing multiple graph sizes
- **Performance Analysis**: Detailed speedup metrics across different input sizes
- **CUDA Profiling**: Comprehensive GPU performance analysis
- **Graph Generation**: Barabási-Albert scale-free graph generator for realistic test cases

## Directory Structure

```
.
├── CPUfixedKCore.cu           # CPU implementation
├── GPUcudaFixedKCore.cu       # GPU/CUDA implementation
├── FixedKCore.cuh             # Header file with function declarations
├── Main.cu                    # Main driver with correctness verification
├── Makefile                   # Build configuration
├── comparesGraph.sh           # Automated testing script
├── profiling.sh               # CUDA profiling script
├── graphGen.py                # Test graph generator
├── README.md                  # This file
├── testGraph_*.txt            # Test graph files (gitignored)
└── profiling_results/         # Profiling output directory
```

## Prerequisites

- NVIDIA CUDA Toolkit (tested with CUDA 11.0+)
- NVIDIA GPU with compute capability 3.5 or higher
- GCC/G++ compiler
- Python 3.x (for graph generation)
- `bc` calculator (for shell script)

## Building the Project

### Compile the executable:

```bash
make
```

This creates the `fixedKCoreComputation` executable.

## Running the Code

### Basic Usage

```bash
./fixedKCoreComputation <nodes> <edges> <graph_file> <k_value>
```

**Parameters:**
- `nodes`: Number of nodes in the graph
- `edges`: Number of edges in the graph
- `graph_file`: Path to the graph file (edge list format)
- `k_value`: K value for K-Core computation

### Example:

```bash
make run ARGS="100000 499985 testGraph_100000_499985.txt 8"
```

Or directly:

```bash
./fixedKCoreComputation 100000 499985 testGraph_100000_499985.txt 8
```

### Output Format:

```
CPU time: 2.000093 ms
GPU time: 0.073910 ms
```

### Clean build artifacts:

```bash
make clean
```

## Generating Test Cases

### Using the Graph Generator

The project includes a Barabási-Albert (BA) graph generator that creates scale-free networks suitable for K-Core analysis.

```bash
python3 graphGen.py
```

**Interactive prompts:**
1. Enter number of nodes (e.g., 100000)
2. Enter initial fully connected nodes (m0) (e.g., 5)
3. Enter edges per new node (e.g., 5)

**Why Barabási-Albert Graphs?**

BA graphs exhibit scale-free properties with power-law degree distributions, making them excellent test cases for K-Core algorithms because:
- They contain natural hub nodes (high-degree vertices)
- They mimic real-world networks (social networks, web graphs)
- They provide challenging test cases with varying core structures

### Graph File Format

Each line represents an undirected edge:
```
node1 node2
node3 node4
...
```

## Test Cases Used

The following test cases were generated and evaluated:

| Nodes      | Edges      | File Name                          |
|------------|------------|------------------------------------|
| 1,000      | 2,995      | testGraph_1000_2995.txt           |
| 10,000     | 49,985     | testGraph_10000_49985.txt         |
| 100,000    | 499,985    | testGraph_100000_499985.txt       |
| 1,000,000  | 4,999,985  | testGraph_1000000_4999985.txt     |
| 10,000,000 | 49,999,985 | testGraph_10000000_49999985.txt   |

All graphs were generated using the BA model with m0=5 and edges_per_node=5, resulting in approximately 5n edges for n nodes.

## Performance Results

### Comparative Analysis (K=8)

| Nodes      | Edges      | CPU (ms)   | GPU (ms)   | Speedup |
|------------|------------|------------|------------|---------|
| 1,000      | 2,995      | 0.033      | 0.072      | 0.46x   |
| 10,000     | 49,985     | 0.228      | 0.074      | 3.07x   |
| 100,000    | 499,985    | 2.000      | 0.074      | 27.06x  |
| 1,000,000  | 4,999,985  | 19.921     | 0.149      | 133.69x |
| 10,000,000 | 49,999,985 | 194.521    | 0.866      | 224.64x |

### Key Observations:

1. **Small Graphs (< 1K nodes)**: GPU overhead dominates, resulting in slower performance than CPU
2. **Medium Graphs (100K nodes)**: GPU begins showing significant advantage (~27x speedup)
3. **Large Graphs (1M+ nodes)**: GPU demonstrates massive speedup (133x - 224x)
4. **Scalability**: GPU performance scales much better with increasing graph size

### Performance Scaling:
- **CPU**: O(n) growth - 10x nodes ≈ 10x time
- **GPU**: Near-constant time for medium to large graphs due to parallelization

## CUDA Profiling Analysis

Detailed profiling was performed on a representative graph (1,000,000 nodes, 4,999,985 edges, K=10) using NVIDIA nvprof to analyze GPU performance characteristics.

### GPU Time Breakdown

| Component | Time (ms) | Percentage | Description |
|-----------|-----------|------------|-------------|
| Memory Transfer (HtoD) | 3.67 | 70.2% | Host to Device data transfer |
| Memory Transfer (DtoH) | 1.47 | 28.1% | Device to Host result transfer |
| initFrontier kernel | 0.039 | 0.74% | Initialize computation frontier |
| fixedKcoreComputation kernel | 0.034 | 0.65% | Main K-Core computation |
| updateFrontier kernel | 0.015 | 0.28% | Update active vertex set |

**Total GPU Time**: ~5.23 ms (kernels + memory transfers)  
**Total API Overhead**: ~160 ms (cudaMalloc, API calls)

### Kernel Execution Details

From detailed kernel trace analysis:

```
Kernel                          Duration    Grid Size      Block Size    Registers
-----------------------------------------------------------------------------------
initFrontier                    38.35 μs    (1954, 1, 1)   (512, 1, 1)   18
fixedKcoreComputation          34.45 μs    (1954, 1, 1)   (512, 1, 1)   20
updateFrontier                  14.68 μs    (1954, 1, 1)   (512, 1, 1)   16
```

### Key Profiling Findings

1. **Memory-Bound Algorithm**: 98.3% of GPU time spent in memory transfers, only 1.7% in computation
2. **Kernel Efficiency**: All three kernels execute in <40μs, demonstrating highly efficient parallel processing
3. **Primary Bottleneck**: Memory transfer dominates execution time - optimization should focus on:
   - Using pinned (page-locked) memory for faster transfers
   - Overlapping computation with memory transfers using CUDA streams
   - Reducing data movement between host and device
4. **Optimal Configuration**: Grid configuration (1954, 1, 1) with block size (512, 1, 1) provides good occupancy
5. **Resource Usage**: Low register usage (16-20 per thread) and zero shared memory allows high occupancy

### Performance Characteristics

- **Memory Throughput**: 
  - Host to Device: 5.7 GB/s
  - Device to Host: 2.8 GB/s
- **Register Usage**: 16-20 registers per thread (efficient)
- **Shared Memory**: 0B (algorithm doesn't require shared memory)
- **Launch Overhead**: Minimal (<1μs between kernel launches)
- **API Overhead**: cudaMalloc dominates with 96% of API call time

### Running Profiling Analysis

To generate profiling data:

```bash
bash profilingScript.sh
```

This creates the `profiling_results/` directory with:
- **BasicProfile.txt**: Overall GPU time breakdown
- **KernelTrace.txt**: Detailed timeline of each kernel execution
- **APITrace.txt**: CUDA API call analysis
- **FullProfile.txt**: Complete profiling data

### Manual Profiling Commands

```bash
# Basic timing breakdown
nvprof ./fixedKCoreComputation 1000000 4999985 testGraph_1000000_4999985.txt 10

# Detailed kernel trace with timeline
nvprof --print-gpu-trace ./fixedKCoreComputation 1000000 4999985 testGraph_1000000_4999985.txt 10

# CUDA API call analysis
nvprof --print-api-trace ./fixedKCoreComputation 1000000 4999985 testGraph_1000000_4999985.txt 10
```

## Automated Testing

Run comprehensive tests across all graph sizes:

```bash
bash comparesGraph.sh
```

This script:
1. Executes the program on all test graphs
2. Extracts CPU and GPU timings
3. Calculates speedup metrics
4. Generates a formatted results table in `DiffgraphResults`

**Output File**: `DiffgraphResults`

## Correctness Verification

The `Main.cu` file includes assertions to verify that CPU and GPU implementations produce identical results - both implementations should generate exactly the same set of nodes. Each run performs:

1. CPU computation of K-Core
2. GPU computation of K-Core
3. Element-wise comparison of results
4. Assertion failure if any mismatch is detected

**Status**: All test cases pass correctness verification ✓

## Implementation Details

### CPU Algorithm
- Single-threaded iterative peeling approach
- Degree-based vertex removal
- Time Complexity: O(|V| + |E|)

### GPU Algorithm
- Parallel frontier-based computation
- Three main CUDA kernels:
  - **initFrontier**: Initialize active vertex set
  - **fixedKcoreComputation**: Parallel degree reduction
  - **updateFrontier**: Update vertices for next iteration
- Efficient memory coalescing for graph traversal
- Concurrent vertex processing
- Optimized for large-scale graphs

### CUDA Optimizations Applied
- Coalesced global memory access patterns
- Efficient thread block configuration (512 threads per block)
- Low register pressure (16-20 registers per thread)
- Minimized host-device memory transfers
- No shared memory overhead (not required for this algorithm)

## Comparison with CUDA Libraries

**Note**: Standard CUDA graph libraries (like cuGraph/RAPIDS) do not provide direct Fixed K-Core computation primitives. The algorithm is specialized for iterative subgraph extraction, which is not a standard primitive in general-purpose graph libraries.

**Current Comparison**: Custom CUDA implementation vs. single-threaded CPU baseline (shown in performance results above).

### Why Standard Libraries Weren't Used:

1. **cuGraph**: Provides k-core decomposition but not fixed k-core extraction with the specific peeling algorithm used here
2. **Thrust**: Could accelerate individual operations but doesn't provide graph algorithm primitives for this specific variant
3. **Custom Implementation**: Required for this specialized algorithm with frontier-based iterative computation

## References

- Barabási-Albert Model: Barabási, A. L., & Albert, R. (1999). Science, 286(5439), 509-512.
- K-Core Decomposition: Batagelj, V., & Zaversnik, M. (2003). arXiv preprint cs/0310049.
- CUDA Programming Guide: https://docs.nvidia.com/cuda/
- CUDA Profiling Tools: https://docs.nvidia.com/cuda/profiler-users-guide/

## License

This project is provided for educational purposes.

---

**Last Updated**: November 2025