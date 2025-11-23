# Fixed K-Core Computation: CPU vs GPU Implementation

A high-performance parallel implementation of the Fixed K-Core algorithm using CUDA, with comparative analysis against single-threaded CPU implementation.

## Project Overview

This project implements the Fixed K-Core graph decomposition algorithm in both CPU (single-threaded) and GPU (CUDA) versions. The K-Core of a graph is the largest subgraph where each vertex has at least K neighbors within that subgraph.

## Features

- **CPU Implementation**: Single-threaded baseline implementation
- **GPU Implementation**: Parallel CUDA implementation with significant speedup
- **Automated Testing**: Script for testing multiple graph sizes
- **Performance Analysis**: Detailed speedup metrics across different input sizes
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
├── graphGen.py                # Test graph generator
├── README.md                  # This file
└── testGraph_*.txt            # Test graph files (gitignored)
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

The `Main.cu` file includes assertions to verify that CPU and GPU implementations produce identical results that is both implementation should generate exactly same set of nodes . Each run performs:

1. CPU computation of K-Core
2. GPU computation of K-Core
3. Element-wise comparison of results 
4. Assertion failure if any mismatch is detected

**Status**: All test cases pass correctness verification ✓

## Comparison with CUDA Libraries

**Note**: Standard CUDA graph libraries (like cuGraph/RAPIDS) do not provide direct K-Core computation primitives. The most relevant comparison would be:

1. **cuGraph K-Core (if available)**: Would require RAPIDS installation

### Future Work:
- Integrate cuGraph for comparative benchmarking


## References

- Barabási-Albert Model: Barabási, A. L., & Albert, R. (1999). Science, 286(5439), 509-512.
- K-Core Decomposition: Batagelj, V., & Zaversnik, M. (2003). arXiv preprint cs/0310049.
- CUDA Programming Guide: https://docs.nvidia.com/cuda/

## License

This project is provided for educational purposes.
